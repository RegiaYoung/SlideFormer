"""Test gradient correctness by comparing manual gradients with autograd."""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_linear_backward():
    """Verify linear backward matches autograd."""
    torch.manual_seed(42)
    B, T, H_in, H_out = 2, 4, 8, 6

    x = torch.randn(B, T, H_in, requires_grad=True, dtype=torch.float32)
    weight = torch.randn(H_out, H_in, requires_grad=True, dtype=torch.float32)

    # Autograd version
    y_auto = torch.nn.functional.linear(x, weight)
    loss_auto = y_auto.sum()
    loss_auto.backward()
    grad_x_auto = x.grad.clone()
    grad_w_auto = weight.grad.clone()

    # Manual version
    x.grad = None
    weight.grad = None
    y_manual = x @ weight.t()
    grad_out = torch.ones_like(y_manual)
    grad_x_manual = grad_out @ weight
    grad_w_manual = grad_out.reshape(-1, H_out).t() @ x.reshape(-1, H_in)

    print(f"Linear grad_x diff: {(grad_x_auto - grad_x_manual).abs().max().item():.2e}")
    print(f"Linear grad_w diff: {(grad_w_auto - grad_w_manual).abs().max().item():.2e}")
    assert torch.allclose(grad_x_auto, grad_x_manual, atol=1e-5)
    assert torch.allclose(grad_w_auto, grad_w_manual, atol=1e-5)
    print("✓ Linear backward correct")

def test_rmsnorm_backward():
    """Verify RMSNorm backward matches autograd."""
    torch.manual_seed(42)
    B, T, H = 2, 4, 8
    eps = 1e-6

    x = torch.randn(B, T, H, requires_grad=True, dtype=torch.float32)
    weight = torch.randn(H, requires_grad=True, dtype=torch.float32)

    # Autograd version
    var = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    y_auto = x * inv_rms * weight
    loss_auto = y_auto.sum()
    loss_auto.backward()
    grad_x_auto = x.grad.clone()
    grad_w_auto = weight.grad.clone()

    # Manual version
    x.grad = None
    weight.grad = None
    var = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    y_manual = x * inv_rms * weight

    grad_out = torch.ones_like(y_manual)
    a = grad_out * weight
    s = (a * x).sum(dim=-1, keepdim=True)
    grad_x_manual = a * inv_rms - (x * inv_rms.pow(3) * s) / H
    grad_w_manual = (grad_out * x * inv_rms).sum(dim=(0, 1))

    print(f"RMSNorm grad_x diff: {(grad_x_auto - grad_x_manual).abs().max().item():.2e}")
    print(f"RMSNorm grad_w diff: {(grad_w_auto - grad_w_manual).abs().max().item():.2e}")
    assert torch.allclose(grad_x_auto, grad_x_manual, atol=1e-5)
    assert torch.allclose(grad_w_auto, grad_w_manual, atol=1e-5)
    print("✓ RMSNorm backward correct")

def test_adamw_bias_correction():
    """Test if AdamW has proper bias correction."""
    print("\n⚠ Testing AdamW bias correction...")
    torch.manual_seed(42)

    param = torch.randn(10, 10)
    grad = torch.randn(10, 10)

    # Without bias correction (current implementation)
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)
    beta1, beta2, lr, eps = 0.9, 0.95, 1e-3, 1e-8

    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    update_no_correction = m / (v.sqrt() + eps)

    # With bias correction (correct implementation)
    m_corrected = m / (1 - beta1)
    v_corrected = v / (1 - beta2)
    update_with_correction = m_corrected / (v_corrected.sqrt() + eps)

    ratio = update_with_correction.abs().mean() / update_no_correction.abs().mean()
    print(f"Update magnitude ratio (corrected/uncorrected): {ratio:.2f}x")
    print(f"Without correction: {update_no_correction.abs().mean():.2e}")
    print(f"With correction: {update_with_correction.abs().mean():.2e}")
    print("⚠ Current implementation is MISSING bias correction!")
    print("  This causes very small updates in early training steps.")

if __name__ == "__main__":
    test_linear_backward()
    test_rmsnorm_backward()
    test_adamw_bias_correction()
    print("\n" + "="*60)
    print("DIAGNOSIS: AdamW missing bias correction is likely the issue!")
    print("="*60)
