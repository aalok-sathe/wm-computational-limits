# stdlib
from abc import ABC
import typing
import dataclasses
from pathlib import Path

# installed packages
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
from jaxtyping import Array, Float

# local
from workingmem.task.interface import GeneratedCachedDataset


@dataclasses.dataclass
class ModelConfig:
    """
    holds configuration parameters for the model
    aim: have some correspondence with the `HookedTransformerConfig` class

    """

    attn_only: bool = True
    n_layers: int = 2
    n_heads: int = 2
    n_ctx: int = 500  # this should be set so that it is definitely longer than the trial sequence length
    d_model: int = 128
    d_head: int = 64
    d_mlp: int = 0
    act_fn: str = "relu"
    d_vocab: int = None  # vocab is determined by the tokenizer
    init_weights: bool = True
    hooked: bool = True
    # checkpoint_dir: typing.Union[Path, None] = None
    # from_checkpoint: typing.Union[Path, None] = None


class ModelWrapper(ABC):
    """
    this model wrapper treats the model as a first-class entity unlike in many training recipes in the field.
    the model(wrapper) is now responsible to train itself, and to evaluate itself on a supplied dataset.
    """

    model: typing.Union[HookedTransformer, torch.nn.Transformer]
    # we want to document the unique identification of the dataset a model has been trained on,
    history: typing.List[typing.Dict]

    def __init__(self, config: ModelConfig):
        self.config = config

        if self.config.hooked:
            self.model = HookedTransformer(
                HookedTransformerConfig(**dataclasses.asdict(config))
            )
        else:
            # regular transformer? idk... this depends on whether HookedTfm has any performance degradation, which I don't know about
            raise NotImplementedError

    def load_checkpoint(self, checkpoint_dir: Path):
        """
        `checkpoint_dir` points to a directory containing:
        - `config.yaml` which contains the `ModelConfig`
        - `model.pth` which contains the model state_dict
        - `history.yaml` which details the training history (this is inherited and appended
            to the existing history, so a model that has been trained first on dataset X and then Y
            will say so in its history)
        """

    def save_checkpoint(self, checkpoint_dir: Path): ...

    def set_embeddings(self, embeddings: typing.Union[np.ndarray, torch.Tensor]):
        """
        explicitly set the embeddings of the model to a supplied matrix. the dimensionality of the matrix should be
        `vocab_size x d_model` as initialized (check `self.config`)
        """

    def train(self, dataset: GeneratedCachedDataset, training_config: "TrainingConfig"):
        return NotImplemented

    def _train_step(self, inputs: dict) -> torch.Tensor:
        """
        this method is responsible for computing the loss on a batch of inputs
        """
        return NotImplemented
        model(**inputs).logits

    def evaluate(self, dataset: GeneratedCachedDataset): ...
    def _evaluate_step(self, inputs: dict): ...


@dataclasses.dataclass
class TrainingConfig:
    epochs: int = 50
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0003
    output_dir: str = "output"

    logging_strategy: str = "epoch"
    logging_steps: int = 1  # log every epoch
    save_strategy: str = "epoch"
    save_steps: int = None


def compute_masked_loss(
    self,
    logits,
    inputs: typing.Dict[str, torch.Tensor],
    label_mask=None,
    return_outputs=False,
) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
    """
    this method acts on:
        - `logits` from a model (this can be softmaxed to get a distribution over the vocabulary elsewhere)
        - `labels` corresponding to input to the model,
            usually in the form of a dict as outputted by a tokenizer.
            the inputs contain labels as `inputs["labels"]` which are used to compute the loss
        - `return_outputs` will cause the method to return the outputs of the model as softmaxed and argmaxed logits
            so they can be directly compared with input_ids
    """
    # logits should have the shape (b, seq_len, vocab_size)
    b, seq_len, vocab_size = logits.shape
    raw_loss = torch.nn.functional.cross_entropy(
        logits, inputs["labels"], reduction="none"
    )
    # after crossentropy, `raw_loss` should have the shape (b, seq_len)
    assert raw_loss.squeeze().shape == (b, seq_len)

    if label_mask is not None:
        batch_label_mask = torch.einops.repeat(
            label_mask,
            "seq_len -> (b seq_len)",  # broadcast the mask to the batch dimension
            b=logits.shape[0],
            reduction="none",
        )
        masked_loss = raw_loss * batch_label_mask
    else:
        masked_loss = raw_loss  # don't apply a label mask

    if return_outputs:
        return masked_loss, torch.nn.functional.softmax(logits, dim=-1).argmax(-1)

    return masked_loss.mean()
