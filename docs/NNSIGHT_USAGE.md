# Using nnsight with wm-computational-limits Models

This document describes how to use nnsight for interpretability with the models in this library.

## Overview

All models in this library (Transformer, RNN, LSTM) are now compatible with [nnsight](https://github.com/ndif-team/nnsight), a modern interpretability library that allows you to:
- Access activations at any layer during forward passes
- Modify activations to study causal effects  
- Compute gradients with respect to intermediate values
- Batch interventions efficiently

## Changes from HookedTransformer

The Transformer model has been migrated from `HookedTransformer` (transformer-lens) to HuggingFace's `GPT2LMHeadModel`, which is:
- Faster and more widely supported
- Fully compatible with nnsight for interpretability
- Uses standard PyTorch interfaces

RNN and LSTM models have been restructured to expose their internal components (embedding, rnn_core/lstm_core, output_layer) for easier access with nnsight.

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
traced_model = LanguageModel(model.model.base_model)  # Access the underlying GPT2 model

# Example: Accessing activations
with traced_model.trace(input_ids) as tracer:
    # Access hidden states from a specific layer
    hidden_states = traced_model.transformer.h[0].output[0].save()
    
    # Access attention output
    attn_output = traced_model.transformer.h[0].attn.output[0].save()

print(f"Hidden states shape: {hidden_states.shape}")
```

### Intervening on Activations

```python
# Zero out activations at a specific layer
with traced_model.trace(input_ids) as tracer:
    # Modify layer 0 activations
    traced_model.transformer.h[0].output[0][:] = 0
    
    # Get the output after intervention
    output = traced_model.lm_head.output.save()
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

### Transformer (GPT2-based)
- **Embedding**: `model.base_model.transformer.wte` (word token embeddings)
- **Positional**: `model.base_model.transformer.wpe` (word position embeddings)
- **Layers**: `model.base_model.transformer.h[i]` (transformer blocks)
- **Output**: `model.base_model.lm_head` (language model head)

### RNN
- **Embedding**: `model.embedding`
- **RNN Core**: `model.rnn_core` (contains the RNN layers)
- **Output**: `model.output_layer`

### LSTM  
- **Embedding**: `model.embedding`
- **LSTM Core**: `model.lstm_core` (contains the LSTM layers)
- **Output**: `model.output_layer`

## Important Notes

1. **Positional Embeddings**: GPT2 uses learned absolute positional embeddings. If your config specifies `positional_embedding_type="rotary"`, a warning will be logged and standard embeddings will be used instead.

2. **No Positional Embeddings**: Setting `positional_embedding_type=None` works as before - the positional embedding weights are zeroed and frozen.

3. **Backward Compatibility**: Existing checkpoints from HookedTransformer models may not be directly compatible. You'll need to either:
   - Retrain models with the new architecture
   - Implement state dict conversion (not currently provided)

4. **Performance**: The new GPT2-based implementation should be faster than HookedTransformer for standard forward passes while maintaining full interpretability capabilities through nnsight.

## Example: Causal Intervention Study

```python
# Study the causal effect of layer 1 on the output
with traced_model.trace(input_ids) as tracer:
    # Save unmodified output
    with tracer.invoke():
        normal_output = traced_model.lm_head.output.save()
    
    # Intervene on layer 1
    with tracer.invoke():
        traced_model.transformer.h[1].output[0][:] = 0
        intervened_output = traced_model.lm_head.output.save()

# Compare outputs
difference = normal_output - intervened_output
print(f"Effect of layer 1: {difference.abs().mean()}")
```

## Further Reading

- [nnsight Documentation](https://nnsight.net)
- [nnsight GitHub](https://github.com/ndif-team/nnsight)
- [HuggingFace GPT2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
