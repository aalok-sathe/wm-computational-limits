# stdlib
from abc import ABC
import typing
import dataclasses
from pathlib import Path

# installed packages
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import Trainer
from torch.nn import Transformer

# local
from workingmem.task.interface import GeneratedCachedDataset


@dataclasses.dataclass
class ModelConfig:
    attn_only: bool = True
    n_layers: int = 2
    n_heads: int = 2
    n_ctx: int = 500  # this should be set in accordance with the task sequence length
    d_model: int = 128
    d_head: int = 64
    d_mlp: int = 0
    act_fn: str = "relu"
    d_vocab: int = None  # vocab is determined by the tokenizer
    init_weights: bool = True
    # checkpoint_dir: typing.Union[Path, None] = None
    # from_checkpoint: typing.Union[Path, None] = None


# class Embeddings: ...


# class RandomEmbeddings:
#     methods = ("standard", "one-hot", "interpolated")
#     frozen: bool

#     def __init__(self, method="standard", frozen=False):
#         assert method in RandomEmbeddings.methods
#         self.frozen = frozen


class ModelWrapper(ABC):
    """
    this model wrapper treats the model as a first-class entity unlike in many training recipes in the field.
    the model(wrapper) is now responsible to train itself, and to evaluate itself on a supplied dataset.
    """

    model: typing.Union[HookedTransformer, Transformer]
    trainer: Trainer  # default trainer to use, but can be overridden in a call to `train` with a different trainer
    history: ...  #

    def __init__(self, model: HookedTransformer, trainer: Trainer = None):
        self.trainer = trainer
        self.model = model
