# utils/param_reuse.py
import torch

def fp32_tensors_to_bf16_tensors(int32_tensors: list[torch.Tensor], bf16_fp32_tensors: list[torch.Tensor]):
    """
    Converts a list of FP32 tensors (viewed as int32 and bf16) in-place to a BF16+Residual format.
    The first half of each bf16_fp32_tensor will contain BF16 data, the second half residual data.
    Args:
        int32_tensors: List of int32 views of original FP32 tensors.
        bf16_fp32_tensors: List of bfloat16 views of original FP32 tensors. These tensors
                           have 2x the number of elements compared to the conceptual FP32 tensor.
    """
    for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            continue
        if int32_tensor.numel() * 2 != bf16_fp32_tensor.numel():
            raise ValueError("Mismatch in tensor elements for conversion. "
                             "BF16 tensor should have 2x elements of the conceptual FP32 tensor (or int32 view).")
        
        int32_tensor.add_(32768)  # Apply rounding offset for FP32 -> BF16
        
        # Rearrange memory:
        # Original FP32 elements (N): [fp32_1, fp32_2, ..., fp32_N]
        # bf16_fp32_tensor view (2N elements): [upper_word1, lower_word1, upper_word2, lower_word2, ...]
        # view(-1, 2) makes it N rows, 2 cols: [[uw1, lw1], [uw2, lw2], ...]
        # transpose(0, 1) makes it 2 rows, N cols: [[uw1, uw2, ...], [lw1, lw2, ...]] (BF16 parts, then Residual parts)
        # reshape(-1) flattens to: [uw1, uw2, ..., lw1, lw2, ...]
        # This means the first N elements of bf16_fp32_tensor are the BF16 data,
        # and the next N elements are the residuals.
        temp_view = bf16_fp32_tensor.view(-1, 2).transpose(0, 1).reshape(-1).contiguous()
        bf16_fp32_tensor.copy_(temp_view)


def bf16_tensors_to_fp32_tensors(int32_tensors: list[torch.Tensor], bf16_fp32_tensors: list[torch.Tensor]):
    """
    Converts a list of BF16+Residual formatted tensors back to FP32 tensors in-place.
    Args:
        int32_tensors: List of int32 views of target FP32 tensors.
        bf16_fp32_tensors: List of bfloat16 views currently holding BF16 data in the first half
                           and residual data in the second half.
    """
    for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            continue
        if int32_tensor.numel() * 2 != bf16_fp32_tensor.numel():
            raise ValueError("Mismatch in tensor elements for conversion.")

        # Rearrange memory:
        # bf16_fp32_tensor (2N elements): [bf16_1, ..., bf16_N, res_1, ..., res_N]
        # view(2, -1) makes it 2 rows, N cols: [[bf16_1, ..., bf16_N], [res_1, ..., res_N]]
        # transpose(0, 1) makes it N rows, 2 cols: [[bf16_1, res_1], [bf16_2, res_2], ...]
        # reshape(-1) flattens to: [bf16_1, res_1, bf16_2, res_2, ...] (interleaved for FP32 reconstruction)
        temp_view = bf16_fp32_tensor.view(2, -1).transpose(0, 1).reshape(-1).contiguous()
        bf16_fp32_tensor.copy_(temp_view)
        
        int32_tensor.sub_(32768) # Remove rounding offset to restore original FP32 bit pattern