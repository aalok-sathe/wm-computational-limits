# stdlib
from abc import ABC
import typing
import dataclasses
from pathlib import Path

# installed packages
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
from tqdm.auto import tqdm
import einops
import wandb

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
    d_mlp: int = 0
    act_fn: str = "relu"
    d_vocab: int = None  # vocab is determined by the tokenizer
    init_weights: bool = True
    hooked: bool = True
    # checkpoint_dir: typing.Union[Path, None] = None
    # from_checkpoint: typing.Union[Path, None] = None

    @property
    def d_head(self) -> int:
        """returns the dimensionality of the attention heads in the model as per `d_model` and `n_heads`"""
        d_head = self.d_model // self.n_heads
        if d_head * self.n_heads == self.d_model:
            return d_head
        raise ValueError(
            f"the model dimensionality {self.d_model} is not divisible by the number of heads "
            f"{self.n_heads} leading to an incompatible head dimensionality {d_head}"
        )


@dataclasses.dataclass
class TrainingConfig:
    epochs: int = 50
    # optimizer: str = "adamw" # we are going to take this for granted
    learning_rate: float = 5e-5
    weight_decay: float = 0.0003
    output_dir: str = "output"

    logging_strategy: str = "epoch"
    logging_steps: int = 1  # log every epoch
    save_strategy: str = "epoch"
    save_steps: int = None


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

        if not self.config.hooked:
            # currently we only support initializing HookedTransformer model instances
            # this doesn't matter much for training but could be useful down the line
            # in eval settings or if we want to incorporate any built-in hooks for
            # mechinterp during train/eval
            raise NotImplementedError

        self.model = HookedTransformer(
            HookedTransformerConfig(**dataclasses.asdict(config))
        )

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

    def train(self, dataset: GeneratedCachedDataset, training_config: TrainingConfig):
        @dataclasses.dataclass
        class TrainingState:
            epoch: int = 0
            epoch_step: int = 0
            best_val_loss: float = np.inf
            best_val_epoch: int = 0

            @property
            def step(self):
                return self.epoch_step + self.epoch * len(dataset)

        # set the model up for training
        # set up the optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        state = TrainingState()
        for state.epoch in tqdm(range(total := training_config.epochs), total=total):
            # set the model to training mode at the beginning of each epoch, since there is
            # no guarantee that it will still be in training mode from the previous epoch
            # if we went into the eval subroutine
            self.model.train()

            for state.step, inputs in enumerate(dataset):
                loss = self._step(inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        ######
                        "train_loss": loss.item(),
                    }
                )

            if state.epoch % training_config.logging_steps == 0:
                eval_loss, eval_acc = self.evaluate(dataset)

                wandb.log(
                    {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        ######
                        "eval_loss": eval_loss.item(),
                        "eval_acc": eval_acc,
                    }
                )

    def evaluate(
        self, dataset: GeneratedCachedDataset, train_epoch: int = None
    ) -> typing.Tuple[float, float]:
        """
        Returns the average loss and accuracy of the model on the dataset (assumed eval or test split)

        Args:
        ---
        dataset: `GeneratedCachedDataset`
            the dataset instance (and split) to evaluate the model on; typically one of val, test
        train_epoch: `int` (optional)
            the epoch number of the training run that made a call to the evaluation run
        """
        self.model.eval()

        losses = []
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for eval_step, inputs in enumerate(dataset):
                loss, answers = self._step(inputs)
                # we have a single loss value per batch (this is a fine approximation)
                losses += [loss.item()]
                predictions += answers.tolist()
                actual_labels += inputs["labels"].tolist()

                # log the first 5 eval examples and predictions to `wandb`
                if eval_step < 5 and train_epoch is not None:
                    wandb.log(
                        {
                            "epoch": train_epoch,
                            "eval_example": inputs["encoding"].input_ids[0].tolist(),
                            "eval_prediction": answers[0].tolist(),
                        }
                    )

        return np.mean(losses), np.mean(
            np.array(predictions) == np.array(actual_labels)
        )

    def _step(
        self, inputs: typing.Dict[str, torch.Tensor], return_outputs=False
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        this method is responsible for computing the loss and optionally the labels
        batch of inputs
        """

        # NOTE: currently the batch dimension is presumed, but there's no mechanism in place
        # to actually batch inputs; in the parent methods (train or evaluate), we are simply
        # iterating over the dataset. the dataset doesn't automatically batch inputs.
        # do we need to use a data collator here?

        logits = (outputs := self.model(**inputs["encoding"])).logits
        if return_outputs:
            # answers is of the shape (b, seq_len), and we are only going to use
            # what's produced at `answer_locations` to compute accuracy
            # (but it may be interesting to look at what the model is producing at each
            # timestep anyway---there is absolutely no pressure for the model to produce
            # anything in particular at the locations that are not answer locations, so
            # I wonder)
            loss, answers = compute_masked_loss(
                logits, inputs, return_outputs=return_outputs
            )
            # convert the boolean mask inputs['answer_locations'] to indices
            # and gather the answers at those locations
            # first, the conversion to indices:
            answer_indices = inputs["answer_locations"].nonzero(as_tuple=False)
            # then, the gathering answers (post argmax-softmax logits)
            gathered_answers = torch.gather(answers, -1, answer_indices)
            gathered_true_labels = torch.gather(
                inputs["encoding"].input_ids, -1, answer_indices
            )

            return loss, gathered_answers, gathered_true_labels
        else:
            loss = compute_masked_loss(logits, inputs, return_outputs=return_outputs)
            return loss


def compute_masked_loss(
    self,
    logits: torch.Tensor,
    inputs: typing.Dict[str, torch.Tensor],
    # label_mask: torch.Tensor = None,
    return_outputs=False,
) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
    """
    this method acts on:
        - `logits` from a model (this can be softmaxed to get a distribution over the vocabulary elsewhere)
        - `inputs` is the dictionary of tensors supplied to the model, which includes the key `input_ids`
            and `answer_locations` which provides a 0-1 mask over which tokens in the sequence should be
            considered 'predictions' (only these are used for loss computation)
        - `label_mask` is a tensor of the same shape as `inputs["labels"]` which is used to mask out
            the locations that don't correspond to a label and that we don't want to use to compute the loss
            (e.g., in the SIR task, the instruction, register, and symbol components are fully stochastic, and
            would only make the learning problem harder if the model tried to predict them)
            if None, no mask is applied, and loss is computed at every location same as a language modeling head
        - `return_outputs` will cause the method to return the outputs of the model as softmaxed and argmaxed logits
            so they can be directly compared with input_ids
    """
    # logits should have the shape (b, seq_len, |V|)
    b, seq_len, vocab_size = logits.shape
    raw_loss = torch.nn.functional.cross_entropy(
        logits, inputs["labels"], reduction="none"
    )
    # after crossentropy, `raw_loss` should have the shape (b, seq_len)
    assert raw_loss.squeeze().shape == (b, seq_len)

    try:
        batch_label_mask = einops.repeat(
            # we are offsetting the answer locations mask by 1 to account for language modeling
            # (the first token is used to predict the next token, which corresponds to the "answer", if any, at that token position)
            inputs["answer_locations"][1:] + [0],
            "seq_len -> (batch_size seq_len)",  # broadcast the mask to the batch dimension
            batch_size=logits.shape[0],
            reduction="none",
        )
        masked_loss = raw_loss * batch_label_mask
        if return_outputs:
            # softmax + argmax leads to (b, seq_len, |V|) -> (b, seq_len)
            return masked_loss, torch.nn.functional.softmax(logits, dim=-1).argmax(-1)

        return masked_loss.mean()

    except KeyError:
        raise ValueError("inputs must contain a key `answer_locations`")
        masked_loss = raw_loss  # don't apply a label mask
