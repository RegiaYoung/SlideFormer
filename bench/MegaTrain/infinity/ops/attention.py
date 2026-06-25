"""Attention operations for MegaTrain.

NOTE: As of v0.2, MegaTrain uses HuggingFace's native attention implementations
(flash_attention_2, sdpa, eager) by setting attn_implementation when loading the model.
This module is retained for backward compatibility but is no longer used by default.

The recommended approach is to set attn_implementation in your YAML config:

    model:
      attn_implementation: "flash_attention_2"  # or "sdpa" or "eager"
"""

import logging

logger = logging.getLogger(__name__)
logger.info(
    "infinity.ops.attention: This module is deprecated. "
    "Use attn_implementation='flash_attention_2' in model config instead."
)
