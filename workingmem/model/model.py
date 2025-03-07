# stdlib
import dataclasses
import typing
from abc import ABC
from pathlib import Path
import logging

# installed packages
import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

import wandb

# local
from workingmem.task.interface import GeneratedCachedDataset

logger = logging.getLogger(__name__)


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
    batch_size: int = 16

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

        logger.info(f"initializing model with config: {config}")
        self.model = HookedTransformer(
            HookedTransformerConfig(
                # config.n_layers,
                # config.d_model,
                # config.n_ctx,
                d_head=config.d_head,
                **dataclasses.asdict(config),
            )
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

    def train(
        self,
        dataset: GeneratedCachedDataset,
        training_config: TrainingConfig,
        eval_dataset: GeneratedCachedDataset = None,
    ):
        """
        given an `eval_dataset`, periodically evaluates model on eval_dataset and logs the results
        """

        train_dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

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

            for state.epoch_step, inputs in enumerate(train_dataloader):
                loss = self._step(inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        ######
                        "train_loss": loss.item(),
                    }
                )

                # logger.debug(f"{wandb_logged = }")

            if state.epoch % training_config.logging_steps == 0:
                eval_loss, eval_acc = self.evaluate(
                    eval_dataset, train_epoch=state.epoch
                )

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        ######
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                    }
                )
                logger.debug(f"{wandb_logged = }")

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
        logger.info("evaluating model")
        self.model.eval()

        eval_dataloader = DataLoader(
            dataset,
            batch_size=16,  # TODO, should we parameterize this?
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        losses = []
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for eval_step, inputs in enumerate(eval_dataloader):
                loss, answers, labels = self._step(inputs, return_outputs=True)
                # we have a single loss value per batch (this is a fine approximation)
                losses += loss.detach().tolist()
                predictions += answers.detach().tolist()
                actual_labels += labels.detach().tolist()

                # log the first batch of eval examples and predictions to `wandb`
                if eval_step < 5 and train_epoch is not None:
                    # TODO: should this be an artifact instead?
                    wandb.log(
                        {
                            "epoch": train_epoch,
                            "eval_step": eval_step,
                            "eval_example": inputs["tokens"],
                            "eval_prediction": dataset.tokenizer.decode(
                                answers
                            ),  # answers.tolist(),
                        }
                    )

        return (
            np.mean(losses),
            np.mean(np.array(predictions) == np.array(actual_labels)),
        )

    def _step(
        self, inputs: typing.Dict[str, torch.Tensor], return_outputs=False
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        this method is responsible for computing the loss and optionally the labels
        batch of a batch of inputs
        """

        # NOTE: currently the batch dimension is presumed, but there's no mechanism in place
        # to actually batch inputs; in the parent methods (train or evaluate), we are simply
        # iterating over the dataset. the dataset doesn't automatically batch inputs.
        # do we need to use a data collator here?

        tokens = inputs["token_ids"]
        tokens.to(self.device)
        logits = self.model(tokens)

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

            logger.debug(f"{loss.shape = }, {inputs['token_ids'].shape = }")
            logger.debug(f"{answers.shape = }, {inputs['answer_locations'].shape = }")

            # convert the boolean mask inputs['answer_locations'] to indices
            # and gather the answers at those locations
            # first, the conversion to indices:
            # NOTE this will not work if each example has a different answer_locations shape!!!
            # but it is OK to make that assumption
            answer_indices_1d = (
                inputs["answer_locations"]
                .reshape(-1)
                .nonzero(as_tuple=False)
                .reshape(-1)
            )

            logger.debug(f"{answer_indices_1d.shape = }")

            # then, the gathering answers (post argmax-softmax logits)
            gathered_answers_1d = torch.gather(
                answers.reshape(-1), 0, answer_indices_1d
            )
            gathered_true_labels_1d = torch.gather(
                inputs["token_ids"].reshape(-1), 0, answer_indices_1d
            )

            gathered_answers = gathered_answers_1d.reshape(
                *inputs["answer_locations"].shape[:-1], -1
            )
            gathered_true_labels = gathered_true_labels_1d.reshape(
                *inputs["answer_locations"].shape[:-1], -1
            )

            logger.debug(
                f"{gathered_answers.shape = }, {gathered_true_labels.shape = }"
            )

            return loss, gathered_answers, gathered_true_labels
        else:
            loss = compute_masked_loss(logits, inputs, return_outputs=return_outputs)
            return loss


def compute_masked_loss(
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
    # logits have the shape (b, seq_len, |V|)
    b, seq_len, vocab_size = logits.shape
    # logger.debug(f"{logits.shape = }, {inputs['token_ids'].shape = }")
    raw_loss = torch.nn.functional.cross_entropy(
        einops.rearrange(logits, "b seq_len vocab -> b vocab seq_len"),
        inputs["token_ids"],
        reduction="none",
    )

    # logger.debug(f"{raw_loss.shape = }, {raw_loss[0] = }")

    # after crossentropy, `raw_loss` should have the shape (b, seq_len)
    assert raw_loss.shape == (b, seq_len)

    try:
        batch_label_mask = torch.concat(
            (inputs["answer_locations"][:, 1:], inputs["answer_locations"][:, :1]),
            dim=1,
        )

        masked_loss = raw_loss * batch_label_mask
        if return_outputs:
            # softmax + argmax leads to (b, seq_len, |V|) -> (b, seq_len)
            return masked_loss, torch.nn.functional.softmax(logits, dim=-1).argmax(-1)

        return masked_loss.mean()

    except KeyError:
        raise ValueError("inputs must contain a key `answer_locations`")
        masked_loss = raw_loss  # don't apply a label mask
