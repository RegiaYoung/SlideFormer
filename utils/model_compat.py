# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0
"""Text-model loading and layer inputs for the public layer-streaming runtime.

Attention, rotary embeddings, expert computation and router loss are provided
by Hugging Face Transformers. These helpers prepare their ordinary inputs.
"""

import copy
import inspect
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


HYBRID_TYPES = {"qwen3_5_text", "qwen3_5_moe_text"}
MODERN_TYPES = HYBRID_TYPES | {"gemma4_text"}


def text_config(config):
    return getattr(config, "text_config", config)


def validate_text_config(config):
    config = text_config(config)
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("Only decoder-only text training is supported")
    if config.model_type == "gemma4_text":
        if (getattr(config, "hidden_size_per_layer_input", 0)
                or getattr(config, "num_kv_shared_layers", 0)):
            raise ValueError("Gemma 4 PLE and cross-layer KV sharing are not supported")
    if getattr(config, "output_router_logits", False) and config.model_type != "qwen3_5_moe_text":
        raise ValueError("Router auxiliary loss is supported for Qwen3.6 MoE only")
    return config


def load_text_model(model_path, *, torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa", use_liger=False,
                    trust_remote_code=False, **kwargs):
    """Load text weights, including the text portion of a composite checkpoint.

    A composite checkpoint may contain vision/audio and prediction heads that
    are not part of its text CausalLM. Missing or mismatched text weights are
    errors. Saved text-only checkpoints load through the same entry point.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config = validate_text_config(config)
    modern = config.model_type in MODERN_TYPES
    loader = AutoModelForCausalLM
    if use_liger and not modern:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        loader = AutoLigerKernelForCausalLM
    config.use_cache = False
    if modern:
        kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
        if config.model_type == "qwen3_5_moe_text" or getattr(config, "enable_moe_block", False):
            kwargs["experts_implementation"] = "eager"
    model, info = loader.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype,
        attn_implementation=attn_implementation, trust_remote_code=trust_remote_code,
        output_loading_info=True, **kwargs,
    )
    ignored = ("model.visual.", "model.vision_tower.", "model.audio_tower.",
               "model.multi_modal_projector.", "model.embed_vision.",
               "model.embed_audio.", "mtp.")
    unexpected = [key for key in info.get("unexpected_keys", [])
                  if not (modern and key.startswith(ignored))]
    if (info.get("missing_keys") or info.get("mismatched_keys")
            or info.get("error_msgs") or unexpected):
        raise ValueError(f"Incomplete text checkpoint: {info}")
    validate_model(model)
    return model


def save_tokenizer(source, output_dir):
    AutoTokenizer.from_pretrained(source).save_pretrained(output_dir)


def validate_model(model):
    config = validate_text_config(model.config)
    if config.model_type == "qwen3_5_moe_text" or (
            config.model_type == "gemma4_text" and getattr(config, "enable_moe_block", False)):
        # Native eager experts accept the unpadded parameter views used by this
        # runtime. The automatic grouped-MM backend requires aligned pointers.
        config._experts_implementation = "eager"
    decoder = model.get_decoder()
    if not hasattr(decoder, "layers") or not hasattr(decoder, "norm"):
        raise ValueError("The text decoder must expose layers and norm")
    managed = {id(p) for module in (model.get_input_embeddings(), *decoder.layers,
                                    decoder.norm, model.get_output_embeddings())
               for p in module.parameters()}
    extra = [name for name, p in model.named_parameters() if id(p) not in managed]
    if extra:
        raise ValueError(f"Parameters outside the text layer layout: {extra}")
    return config


def parameter_sizes(model):
    """Return the largest physical layer size and total unique parameter count."""
    decoder = model.get_decoder()
    groups = [model.get_input_embeddings(), *decoder.layers]
    sizes = [sum(p.numel() for p in module.parameters()) for module in groups]
    # The output window also holds the tied embedding's read-only weight copy.
    sizes.append(sum(p.numel() for p in decoder.norm.parameters())
                 + sum(p.numel() for p in model.get_output_embeddings().parameters()))
    return max(sizes), sum(p.numel() for p in model.parameters())


def build_decoder_attention_mask(decoder, hidden_states, attention_mask=None,
                                 position_ids=None, cache_position=None,
                                 past_key_values=None):
    legacy = getattr(decoder, "_update_causal_mask", None)
    if legacy is not None:
        if "output_attentions" in inspect.signature(legacy).parameters:
            return legacy(attention_mask, hidden_states, cache_position, past_key_values, False)
        return legacy(attention_mask, hidden_states, cache_position, past_key_values)
    from transformers.masking_utils import create_causal_mask
    return create_causal_mask(config=decoder.config, inputs_embeds=hidden_states,
                              attention_mask=attention_mask, position_ids=position_ids,
                              cache_position=cache_position, past_key_values=past_key_values)


def prepare_layer_inputs(decoder, hidden_states, attention_mask=None, position_ids=None):
    """Mirror the supported HF text backbones' per-layer input preparation."""
    config = validate_text_config(decoder.config)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    rotary = getattr(decoder, "rotary_emb", None)
    if rotary is not None:
        rotary.to(hidden_states.device)
    if config.model_type in HYBRID_TYPES:
        from transformers.masking_utils import create_causal_mask
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
        text_positions = position_ids[0] if position_ids.shape[0] == 4 else None
        rope_positions = position_ids[1:] if position_ids.shape[0] == 4 else position_ids
        full_mask = create_causal_mask(config=config, inputs_embeds=hidden_states,
                                      attention_mask=attention_mask, past_key_values=None,
                                      position_ids=text_positions)
        linear_mask = decoder._update_linear_attn_mask(attention_mask, None)
        positions = rotary(hidden_states, rope_positions)
        return [dict(attention_mask=linear_mask if kind == "linear_attention" else full_mask,
                     position_embeddings=positions, position_ids=text_positions,
                     past_key_values=None, use_cache=False)
                for kind in config.layer_types]
    if config.model_type == "gemma4_text":
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        common = dict(config=config, inputs_embeds=hidden_states, attention_mask=attention_mask,
                      past_key_values=None, position_ids=position_ids)
        masks = {"full_attention": create_causal_mask(**common),
                 "sliding_attention": create_sliding_window_causal_mask(**common)}
        positions = {kind: rotary(hidden_states, position_ids, kind)
                     for kind in set(config.layer_types)}
        return [dict(attention_mask=masks[kind], position_embeddings=positions[kind],
                     position_ids=position_ids, past_key_values=None, shared_kv_states={})
                for kind in config.layer_types]
    mask = build_decoder_attention_mask(decoder, hidden_states, attention_mask,
                                        position_ids, cache_position)
    positions = rotary(hidden_states, position_ids) if rotary is not None else None
    return [dict(attention_mask=mask, position_ids=position_ids, cache_position=cache_position,
                 position_embeddings=positions, output_attentions=False)
            for _ in decoder.layers]


def call_decoder_layer(layer, hidden_states, layer_kwargs, collect_router=False):
    """Expose native router outputs as checkpoint outputs when aux loss is enabled."""
    captured = []
    hook = None
    if collect_router:
        hook = layer.mlp.gate.register_forward_hook(lambda module, args, out: captured.append(out[0]))
    try:
        output = layer(hidden_states, **layer_kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        return (hidden, captured[0]) if collect_router else hidden
    finally:
        if hook is not None:
            hook.remove()


def router_aux_loss(model, router_outputs, attention_mask=None):
    if not router_outputs:
        return None
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func
    config = model.config
    return load_balancing_loss_func(tuple(router_outputs), config.num_experts,
                                    config.num_experts_per_tok, attention_mask)


def save_text_weights(model, output_dir, dtype):
    """Serialize without replacing the offloader's live FP32 parameter views."""
    converted, state = {}, {}
    for name, value in model.state_dict(keep_vars=True).items():
        if id(value) not in converted:
            tensor = value.detach()
            converted[id(value)] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        state[name] = converted[id(value)]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, state_dict=state)
    # save_pretrained infers config dtype from the live FP32 master weights.
    # Record the serialized dtype so a standalone dtype="auto" load is correct.
    saved_config = copy.deepcopy(model.config)
    if hasattr(saved_config, "dtype"):
        saved_config.dtype = dtype
    else:
        saved_config.torch_dtype = dtype
    saved_config.save_pretrained(output_dir)
