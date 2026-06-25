"""
Detailed diagnosis of why loss is not decreasing fast enough.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rmsnorm_numerical():
    """Check if RMSNorm implementation matches PyTorch LayerNorm."""
    print("="*60)
    print("Testing RMSNorm Numerical Correctness")
    print("="*60)

    from infinity.ops.layers import rmsnorm

    torch.manual_seed(42)
    B, T, H = 2, 4, 8
    x = torch.randn(B, T, H, requires_grad=True)
    weight = torch.ones(H, requires_grad=True)

    # Our RMSNorm
    y_ours = rmsnorm(x, weight, eps=1e-6)
    loss_ours = y_ours.sum()
    loss_ours.backward()
    grad_x_ours = x.grad.clone()
    grad_w_ours = weight.grad.clone()

    # Reference implementation
    x.grad = None
    weight.grad = None
    var = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + 1e-6)
    y_ref = x * inv_rms * weight
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    grad_w_ref = weight.grad.clone()

    print(f"Forward diff: {(y_ours - y_ref).abs().max():.2e}")
    print(f"Grad x diff: {(grad_x_ours - grad_x_ref).abs().max():.2e}")
    print(f"Grad w diff: {(grad_w_ours - grad_w_ref).abs().max():.2e}")

    if (y_ours - y_ref).abs().max() < 1e-5:
        print("✓ RMSNorm forward is correct")
    else:
        print("❌ RMSNorm forward has issues")

    if (grad_x_ours - grad_x_ref).abs().max() < 1e-5:
        print("✓ RMSNorm backward is correct")
    else:
        print("❌ RMSNorm backward has issues")


def test_attention_output():
    """Check if attention produces reasonable outputs."""
    print("\n" + "="*60)
    print("Testing Attention Output")
    print("="*60)

    from infinity.ops.layers import MultiHeadAttention

    torch.manual_seed(42)
    B, T, H = 2, 4, 8

    attn = MultiHeadAttention(hidden_size=H, num_heads=2)
    x = torch.randn(B, T, H)

    # Check output
    y = attn(x)
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")
    print(f"Output has NaN: {y.isnan().any()}")
    print(f"Output has Inf: {y.isinf().any()}")

    if not y.isnan().any() and not y.isinf().any():
        print("✓ Attention output is valid")
    else:
        print("❌ Attention output has NaN/Inf")


def test_simple_linear_model():
    """Test with simplest possible model - just linear layers."""
    print("\n" + "="*60)
    print("Testing Simple Linear Model (Baseline)")
    print("="*60)

    # Simplest model - just linear layers
    model = nn.Sequential(
        nn.Embedding(100, 8),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 100),
    )

    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (4, 10))
    labels = torch.randint(0, 100, (4, 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for step in range(100):
        optimizer.zero_grad()

        hidden = model[0](input_ids)
        hidden = model[1](hidden)
        hidden = model[2](hidden)
        logits = model[3](hidden)

        loss = nn.functional.cross_entropy(
            logits.view(-1, 100),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == 99:
            print(f"Step {step+1}: loss={loss.item():.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")

    if reduction > 50:
        print("✓ Simple model can overfit easily")
    else:
        print("⚠ Even simple model struggles - may be data issue")


def test_transformer_with_standard_pytorch():
    """Test with standard PyTorch TransformerEncoderLayer."""
    print("\n" + "="*60)
    print("Testing Standard PyTorch Transformer")
    print("="*60)

    # Use standard PyTorch transformer
    model = nn.Sequential(
        nn.Embedding(100, 8),
        nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True),
        nn.Linear(8, 100),
    )

    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (4, 10))
    labels = torch.randint(0, 100, (4, 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for step in range(100):
        optimizer.zero_grad()

        hidden = model[0](input_ids)
        hidden = model[1](hidden)
        logits = model[2](hidden)

        loss = nn.functional.cross_entropy(
            logits.view(-1, 100),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == 99:
            print(f"Step {step+1}: loss={loss.item():.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")

    if reduction > 50:
        print("✓ Standard PyTorch transformer can overfit")
    else:
        print("⚠ Standard transformer also struggles")


def main():
    print("\n" + "="*60)
    print("DETAILED DIAGNOSIS")
    print("="*60 + "\n")

    test_rmsnorm_numerical()
    test_attention_output()
    test_simple_linear_model()
    test_transformer_with_standard_pytorch()

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
