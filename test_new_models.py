#!/usr/bin/env python
"""Quick test script for LSTMMultiCell, LSTMMultiCellWrapper, and RIMModelWrapper"""

import sys
sys.path.insert(0, '/home/aalok/code/wm-computational-limits/src')

import torch
from workingmem.model.model import LSTMMultiCell, LSTMMultiCellWrapper
from workingmem.model.interface import ModelConfig

print("=" * 60)
print("Testing LSTMMultiCell")
print("=" * 60)

# Test 1: LSTMMultiCell instantiation
print("\n1. Testing LSTMMultiCell instantiation...")
try:
    cell = LSTMMultiCell(input_size=64, hidden_size=128, num_cells=3, merge_strategy="gated")
    print(f"   ✓ LSTMMultiCell created with gated merge strategy")

    # Test all merge strategies
    for strategy in ["average", "concatenate", "gated"]:
        cell = LSTMMultiCell(input_size=64, hidden_size=128, num_cells=3, merge_strategy=strategy)
        print(f"   ✓ LSTMMultiCell works with '{strategy}' merge strategy")
except Exception as e:
    print(f"   ✗ Failed to create LSTMMultiCell: {e}")
    sys.exit(1)

# Test 2: LSTMMultiCell forward pass
print("\n2. Testing LSTMMultiCell forward pass...")
try:
    cell = LSTMMultiCell(input_size=64, hidden_size=128, num_cells=3, merge_strategy="gated")
    x = torch.randn(2, 10, 64)  # (batch, seq_len, input_size)
    output, (h, c) = cell(x)

    assert output.shape == (2, 10, 128), f"Expected output shape (2, 10, 128), got {output.shape}"
    assert h.shape == (2, 128), f"Expected h shape (2, 128), got {h.shape}"
    assert c.shape == (2, 128), f"Expected c shape (2, 128), got {c.shape}"
    print(f"   ✓ Gated merge: output shape {output.shape}, h shape {h.shape}, c shape {c.shape}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 3: Concatenate merge strategy output size
print("\n3. Testing concatenate merge strategy output size...")
try:
    cell = LSTMMultiCell(input_size=64, hidden_size=128, num_cells=3, merge_strategy="concatenate")
    x = torch.randn(2, 10, 64)
    output, (h, c) = cell(x)

    expected_hidden_size = 128 * 3  # concatenate multiplies by num_cells
    assert output.shape[-1] == expected_hidden_size, f"Expected output hidden size {expected_hidden_size}, got {output.shape[-1]}"
    print(f"   ✓ Concatenate merge: output shape {output.shape}, h shape {h.shape}")
except Exception as e:
    print(f"   ✗ Concatenate merge failed: {e}")
    sys.exit(1)

# Test 4: Gradient flow
print("\n4. Testing gradient flow...")
try:
    cell = LSTMMultiCell(input_size=64, hidden_size=128, num_cells=3, merge_strategy="gated")
    x = torch.randn(2, 10, 64, requires_grad=True)
    output, (h, c) = cell(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    assert not torch.all(x.grad == 0), "Input gradient is all zeros"
    print(f"   ✓ Gradients flow successfully")
except Exception as e:
    print(f"   ✗ Gradient flow failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Testing LSTMMultiCellWrapper")
print("=" * 60)

# Test 5: LSTMMultiCellWrapper instantiation
print("\n5. Testing LSTMMultiCellWrapper instantiation...")
try:
    config = ModelConfig(
        model_class="lstm_multi_cell",
        d_vocab=1000,
        d_model=256,
        d_hidden=256,
        num_lstm_cells=3,
        lstm_merge_strategy="gated"
    )
    wrapper = LSTMMultiCellWrapper(config)
    print(f"   ✓ LSTMMultiCellWrapper created successfully")
except Exception as e:
    print(f"   ✗ Failed to create LSTMMultiCellWrapper: {e}")
    sys.exit(1)

# Test 6: LSTMMultiCellWrapper forward pass
print("\n6. Testing LSTMMultiCellWrapper forward pass...")
try:
    config = ModelConfig(
        model_class="lstm_multi_cell",
        d_vocab=1000,
        d_model=256,
        d_hidden=256,
        num_lstm_cells=3,
        lstm_merge_strategy="gated"
    )
    wrapper = LSTMMultiCellWrapper(config)
    token_ids = torch.randint(0, 1000, (2, 10))
    logits = wrapper.model(token_ids)

    assert logits.shape == (2, 10, 1000), f"Expected logits shape (2, 10, 1000), got {logits.shape}"
    print(f"   ✓ Wrapper forward pass works: logits shape {logits.shape}")
except Exception as e:
    print(f"   ✗ Wrapper forward pass failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Testing RIMModelWrapper instantiation (basic)")
print("=" * 60)

# Test 7: RIMModelWrapper creation attempt (may fail if RIM not installed)
print("\n7. Testing RIMModelWrapper instantiation...")
try:
    config = ModelConfig(
        model_class="rim",
        d_vocab=1000,
        d_model=256,
        d_hidden=256,
        num_mechanisms=4
    )
    try:
        from workingmem.model.model import RIMModelWrapper
        wrapper = RIMModelWrapper(config)
        print(f"   ✓ RIMModelWrapper created (RIM library available)")
    except ImportError as ie:
        if "RIM not installed" in str(ie):
            print(f"   ⚠ RIMModelWrapper requires RIM library: {ie}")
        else:
            raise
except Exception as e:
    print(f"   ✗ RIMModelWrapper creation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
