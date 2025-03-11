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
from matplotlib import pyplot as plt

import wandb

# local
from workingmem.task.interface import GeneratedCachedDataset

logger = logging.getLogger("workingmem")
logger.setLevel(logging.DEBUG)


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
    epochs: int = 10
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
        raise NotImplementedError

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
                return self.epoch_step + np.ceil(
                    self.epoch * len(dataset) / training_config.batch_size
                )

        predictions_table = wandb.Table(
            columns=[
                "epoch",  # so that we can observe the evolution of the model's predictions over time
                "example_ix",
                "eval_example",
                "eval_prediction",
                "eval_labels",
                # "logits",
            ]
        )

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
                    eval_dataset,
                    train_epoch=state.epoch,
                    # passing the predictions table to the eval loop for
                    # logging predictions and heatmaps over logits
                    predictions_table=predictions_table,
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

        if predictions_table is not None:
            wandb.log({"predictions": predictions_table})

    def evaluate(
        self,
        dataset: GeneratedCachedDataset,
        train_epoch: int = None,
        predictions_table: wandb.Table = None,
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
            batch_size=1,  # TODO, should we parameterize this?
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        losses = []
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for eval_step, inputs in enumerate(eval_dataloader):
                loss, answer_logits, answers, labels = self._step(
                    inputs, return_outputs=True
                )
                # we have a single loss value per batch (this is a fine approximation)
                losses += [loss.item()]
                predictions += answers.detach().tolist()
                actual_labels += labels.detach().tolist()

                # log the first batch of eval examples and predictions to `wandb`
                if train_epoch is not None and predictions_table is not None:
                    # TODO: should this be an artifact instead?
                    # columns = [
                    #     "epoch",  # so that we can observe the evolution of the model's predictions over time
                    #     "eval_step",  # this is the step within the training epoch
                    #     "eval_example",
                    #     "eval_prediction",
                    #     "eval_labels",
                    #     "logits",  # logits will be a seq_len x |V| heatmap for each example
                    # ]

                    # b, num_answers, |V|
                    vocab_dict = dataset.tokenizer.get_vocab()
                    id2token = {v: k for k, v in vocab_dict.items()}

                    def logit_to_dict(logit: torch.Tensor):
                        """
                        opearates on a 1-D tensor of logits
                        """
                        if logit.shape != (max(vocab_dict.values()) + 1,):
                            raise ValueError(
                                f"logit must be a 1-D tensor of logits equal to {max(vocab_dict.values())+1 = }. received: {logit.shape = }"
                            )
                        d = {id2token[i]: float(logit[i]) for i in id2token}
                        # sort d by values and keep the top 10
                        return str(
                            dict(
                                sorted(
                                    d.items(),
                                    key=lambda x: x[1] if x[0] not in "samediff" else 1,
                                    reverse=True,
                                )[:12]
                            )
                        )

                    # # b, num_answers, |V|
                    # prob_matrix = (
                    #     torch.nn.functional.softmax(answer_logits, dim=-1)
                    #     .cpu()
                    #     .detach()
                    #     .numpy()
                    # )
                    # # b x num_answers dictionaries
                    # prob_dicts = []
                    # for b in range(prob_matrix.shape[0]):
                    #     this_prob_dicts = []
                    #     for ans_ix in range(prob_matrix.shape[1]):
                    #         this_prob_dicts += [logit_to_dict(prob_matrix[b, ans_ix])]
                    #     prob_dicts.append(this_prob_dicts)

                    for example_ix in range(len(inputs["tokens"])):
                        predictions_table.add_data(
                            train_epoch,
                            example_ix,  # corresponds to batch
                            inputs["tokens"][example_ix],
                            dataset.tokenizer.decode(
                                answers[example_ix].detach().tolist()
                            ),
                            dataset.tokenizer.decode(
                                labels[example_ix].detach().tolist()
                            ),
                            # prob_dicts[
                            #     example_ix
                            # ],  # explicit probability distribution over vocabulary
                        )

        return (
            np.mean(losses),
            np.mean(np.array(predictions) == np.array(actual_labels)),
        )

    def _step(
        self, inputs: typing.Dict[str, torch.Tensor], return_outputs=False
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
        # shape of logits: (b, seq_len, |V|)
        logits = self.model(tokens)

        if return_outputs:
            # answers is of the shape (b, seq_len), and we are only going to use
            # what's produced at `answer_locations` to compute accuracy
            # (but it may be interesting to look at what the model is producing at each
            # timestep anyway---there is absolutely no pressure for the model to produce
            # anything in particular at the locations that are not answer locations, so
            # I wonder)
            outputs = compute_masked_loss(logits, inputs, return_outputs=return_outputs)
            loss, gathered_logits, gathered_answers, gathered_labels = (
                outputs["loss"],
                outputs["gathered_logits"],
                outputs["gathered_answers"],
                outputs["gathered_labels"],
            )

            logger.debug(f"{loss.shape = }, {inputs['token_ids'].shape = }")
            logger.debug(
                f"{gathered_answers.shape = }, {inputs['answer_locations'].shape = }"
            )
            logger.debug(
                f"{gathered_logits.shape = }, {gathered_answers.shape = }, {gathered_labels.shape = }"
            )

            # convert the boolean mask inputs['answer_locations'] to indices
            # and gather the answers at those locations
            # first, the conversion to indices:
            # ~ ~N~O~T~E~ ~t~h~i~s~ ~w~i~l~l~ ~n~o~t~ ~w~o~r~k~ ~i~f~ ~e~a~c~h~ ~e~x~a~m~p~l~e~ ~h~a~s~ ~a~ ~d~i~f~f~e~r~e~n~t~ ~a~n~s~w~e~r~_~l~o~c~a~t~i~o~n~s~ ~s~h~a~p~e~!~!~!~
            # ~ ~b~u~t~ ~i~t~ ~i~s~ ~O~K~ ~t~o~ ~m~a~k~e~ ~t~h~a~t~ ~a~s~s~u~m~p~t~i~o~n
            # length of `answer_indices_1d`: b x O(seq_len)
            # answer_indices_1d = (
            #     inputs["answer_locations"]
            #     .reshape(-1)
            #     .nonzero(as_tuple=False)
            #     .reshape(-1)
            # )

            # logger.debug(f"{answer_indices_1d.shape = }")

            # # then, the gathering answers (post argmax-softmax logits)
            # gathered_answers_1d = torch.gather(
            #     answers.reshape(-1), dim=0, index=answer_indices_1d
            # )
            # gathered_true_labels_1d = torch.gather(
            #     inputs["token_ids"].reshape(-1), dim=0, index=answer_indices_1d
            # )
            # # we preserve the vocab dimension on logits
            # gathered_logits_2d = einops.rearrange(
            #     logits, "b seq_len vocab -> (b seq_len) vocab"
            # )[answer_indices_1d, :]

            # gathered_answers = gathered_answers_1d.reshape(
            #     # b, O(seq_len)
            #     *inputs["answer_locations"].shape[:-1],
            #     -1,
            # )
            # gathered_true_labels = gathered_true_labels_1d.reshape(
            #     # b, O(seq_len)
            #     *inputs["answer_locations"].shape[:-1],
            #     -1,
            # )

            # # NOTE: this reshape won't be possible if the number of answers per example is not the same across examples
            # # b, num_answers_per_eg, |V|
            # gathered_logits = gathered_logits_2d.reshape(
            #     inputs["answer_locations"].shape[0],
            #     -1,  # whatever the number of answers per example is
            #     logits.shape[-1],
            # )

            # batch_size = inputs["answer_locations"].shape[0]
            # answer_seq_len = len(answer_indices_1d) / batch_size
            # vocab_size = logits.shape[-1]

            # assert gathered_answers.shape == (batch_size, answer_seq_len)
            # assert gathered_logits.shape == (batch_size, answer_seq_len, vocab_size)

            # logger.debug(
            #     f"{gathered_answers.shape = }, {gathered_true_labels.shape = }, {gathered_logits.shape = }"
            # )

            return loss, gathered_logits, gathered_answers, gathered_labels
        else:
            loss = compute_masked_loss(logits, inputs, return_outputs=return_outputs)
            return loss


def compute_masked_loss(
    logits: torch.Tensor,
    inputs: typing.Dict[str, torch.Tensor],
    # label_mask: torch.Tensor = None,
    return_outputs=False,
    # return_logits=False,
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
    gathered_logits = logits[
        :, inputs["answer_locations"][0].nonzero(as_tuple=True)[0] - 1, :
    ]
    gathered_labels = inputs["token_ids"][
        :, inputs["answer_locations"][0].nonzero(as_tuple=True)[0]
    ]

    logger.debug(
        f"in COMPUTE_MASKED_LOSS: {gathered_logits.shape = }, {gathered_labels.shape = }"
    )
    # logger.debug(gathered_answers)

    loss = torch.nn.functional.cross_entropy(
        # NOTE: a simple rearrange is not appropriate here! we need to permute the dimensions.
        # see here for why: https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
        # DONT: einops.rearrange(logits, "b seq_len vocab -> b vocab seq_len"),
        # DO:
        torch.permute(gathered_logits, (0, 2, 1)),
        gathered_labels,
        reduction="mean",
    )

    if return_outputs:
        return dict(
            loss=loss,
            gathered_logits=gathered_logits,
            gathered_answers=torch.nn.functional.softmax(
                gathered_logits, dim=-1
            ).argmax(-1),
            gathered_labels=gathered_labels,
        )
    return loss
