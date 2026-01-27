# Using nnsight with wm-computational-limits Models

This document describes how to use nnsight for interpretability with the models in this library.

## Overview

All models in this library (Transformer, RNN, LSTM) are now compatible with [nnsight](https://github.com/ndif-team/nnsight), a modern interpretability library that allows you to:
- Access activations at any layer during forward passes
- Modify activations to study causal effects  
- Compute gradients with respect to intermediate values
- Batch interventions efficiently

## Model Architecture

### Transformer
The Transformer model uses PyTorch's `nn.Transformer` as the core architecture, similar to how RNN and LSTM are implemented. This provides a bare-bones, general-purpose implementation that is:
- Not tied to any specific architecture (e.g., GPT-2, BERT)
- Easily modifiable for research purposes
- Fully compatible with nnsight for interpretability

The model supports multiple positional embedding types:
- **standard**: Learned positional embeddings (default)
- **rotary**: Rotary Position Embeddings (RoPE)
- **None**: No positional embeddings

### RNN & LSTM
RNN and LSTM models expose their internal components (embedding, rnn_core/lstm_core, output_layer) for easier access with nnsight.

## Basic Usage

### Standard Training (no interventions)

Models work exactly as before - no changes to existing code are needed:

```python
from workingmem.model import TransformerModelWrapper, ModelConfig

config = ModelConfig(
    model_class="transformer",
    n_layers=2,
    d_model=256,
    d_vocab=100,
    positional_embedding_type="rotary",  # or "standard", or None
    # ... other config options
)

model = TransformerModelWrapper(config)
# Train and evaluate as usual
```

### Using nnsight for Interpretability

When you want to inspect or modify model internals, wrap your forward pass with nnsight's trace:

```python
from nnsight import LanguageModel

# Wrap the model with nnsight's LanguageModel for tracing
traced_model = LanguageModel(model.model)

# Example: Accessing activations
with traced_model.trace(input_ids) as tracer:
    # Access embeddings
    embeddings = traced_model.embedding.output.save()
    
    # Access transformer encoder output
    encoder_output = traced_model.transformer_encoder.output.save()

print(f"Embeddings shape: {embeddings.shape}")
```

### Intervening on Activations

```python
# Zero out activations at the embedding layer
with traced_model.trace(input_ids) as tracer:
    # Modify embeddings
    traced_model.embedding.output[:] = 0
    
    # Get the output after intervention
    output = traced_model.output_layer.output.save()
```

### RNN/LSTM with nnsight

RNN and LSTM models expose their components for easy access:

```python
from workingmem.model import RNNModelWrapper, ModelConfig
from nnsight import LanguageModel

config = ModelConfig(model_class="rnn", d_model=64, d_hidden=128, ...)
rnn_model = RNNModelWrapper(config)

# Wrap for tracing
traced_rnn = LanguageModel(rnn_model.model)

with traced_rnn.trace(input_ids) as tracer:
    # Access embedding outputs
    embeddings = traced_rnn.embedding.output.save()
    
    # Access RNN core outputs  
    rnn_output = traced_rnn.rnn_core.output.save()
    
    # Modify activations
    traced_rnn.rnn_core.output[:] = 0  # Zero out RNN outputs
```

## Architecture Details

### Transformer (PyTorch nn.Transformer-based)
- **Embedding**: `model.embedding` (token embeddings)
- **Positional Encoding**: `model.pos_encoder` (positional embeddings, if enabled)
- **Transformer Encoder**: `model.transformer_encoder` (transformer layers)
- **Output**: `model.output_layer` (projection to vocabulary)

### RNN
- **Embedding**: `model.embedding`
- **RNN Core**: `model.rnn_core` (contains the RNN layers)
- **Output**: `model.output_layer`

### LSTM  
- **Embedding**: `model.embedding`
- **LSTM Core**: `model.lstm_core` (contains the LSTM layers)
- **Output**: `model.output_layer`

## Positional Embedding Types

### Standard (Learned)
```python
config = ModelConfig(
    positional_embedding_type="standard",
    # ... other options
)
```
Uses learned positional embeddings similar to the original Transformer paper.

### Rotary Position Embeddings (RoPE)
```python
config = ModelConfig(
    positional_embedding_type="rotary",
    # ... other options
)
```
Implements Rotary Position Embeddings as described in [RoFormer](https://arxiv.org/abs/2104.09864).

### No Positional Embeddings
```python
config = ModelConfig(
    positional_embedding_type=None,
    # ... other options
)
```
Disables positional embeddings entirely. Useful for studying position-independent representations.

## Important Notes

1. **Positional Embeddings**: All three types (standard, rotary, None) are fully supported.

2. **Backward Compatibility**: Existing checkpoints from previous implementations may not be directly compatible. Retrain models if needed.

3. **Performance**: The bare-bones PyTorch implementation is efficient and suitable for research purposes.

4. **Interpretability**: All components are easily accessible through nnsight's tracing functionality.

## Example: Causal Intervention Study

```python
# Study the causal effect of positional embeddings on the output
with traced_model.trace(input_ids) as tracer:
    # Save unmodified output
    with tracer.invoke():
        normal_output = traced_model.output_layer.output.save()
    
    # Zero out positional embeddings
    with tracer.invoke():
        if traced_model.pos_encoder is not None:
            traced_model.pos_encoder.output[:] = 0
        intervened_output = traced_model.output_layer.output.save()

# Compare outputs
difference = normal_output - intervened_output
print(f"Effect of positional embeddings: {difference.abs().mean()}")
```

## Further Reading

- [nnsight Documentation](https://nnsight.net)
- [nnsight GitHub](https://github.com/ndif-team/nnsight)
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Rotary Position Embeddings Paper](https://arxiv.org/abs/2104.09864)
