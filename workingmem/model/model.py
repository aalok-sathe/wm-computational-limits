# stdlib
import dataclasses
import typing
from abc import ABC
from pathlib import Path
import logging
import yaml

# installed packages
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

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

    from_pretrained: typing.Union[str, None] = None
    """`from_pretrained` is a path to a directory containing the model checkpoints and config.yaml.
        typically:
        +-- config.yaml
        +-- history.yaml
        +-- checkpoints/{epoch}.pth, ...
        +-- best_model.pth 
    if supplied, any ptions in the existing `ModelConfig` are ignored.  model is initialized using the config in the config.yaml file, and the state_dict is loaded from the *.pth file.  
    """
    attn_only: bool = True
    n_layers: int = 2
    n_heads: int = 2
    n_ctx: int = 500  # this should be set so that it is definitely longer than the longest trial sequence length
    d_model: int = 128  # dimensionality of the residual stream and embeddings
    d_mlp: int = 0
    act_fn: str = "relu"  # from HookedTransformerConfig: "Must be set unless using an attn-only model."
    d_vocab: int = None  # vocab is determined by the tokenizer
    init_weights: bool = True
    seed: typing.Union[int, None] = None

    @property
    def d_head(self) -> int:
        """calculate the dimensionality of the attention heads in the model as per `d_model` and `n_heads`"""
        d_head = self.d_model // self.n_heads
        if d_head * self.n_heads == self.d_model:
            return d_head
        raise ValueError(
            f"the model dimensionality {self.d_model} is not divisible by the number of heads "
            f"{self.n_heads} leading to an incompatible head dimensionality {d_head}"
        )


@dataclasses.dataclass
class TrainingConfig:
    freeze_embeddings: bool = None
    epochs: int = 50
    optimizer: str = "adamw"  # do not change this!
    learning_rate: float = 5e-5
    weight_decay: float = 0.0003
    # this is where checkpoints are saved, if supplied.
    # if available, a wandb.run.sweep_id AND a model random seed will be appended
    # to the checkpoint directory name.
    # e.g. `model_checkpoints/{sweep_id}/{run_name}/`
    checkpoint_dir: typing.Union[str, None] = "model_checkpoints/"
    batch_size: int = 128

    logging_strategy: str = "epoch"  # log every X epochs or X steps?
    logging_steps: int = 1  # log every X epochs/steps
    log_predictions: typing.Union[bool, None] = None

    # log X many times per epoch: the # of steps to log after is determined
    # by the dataset length and batch size
    logging_steps_per_epoch: int = 5
    # 'best' saves a checkpoint each time we see a drop in validation loss, named 'best_model.pth'
    # 'epoch' saves a checkpoint at the end of each epoch named 'epoch_{epoch}.pth' in a subdirectory called 'checkpoints/'
    save_strategy: typing.Literal["best", "epoch"] = "best"
    # if strategy is 'epoch', then we save every X epochs determined by `save_steps`
    save_steps: int = None

    do_test: bool = True


@dataclasses.dataclass
class TrainingHistoryEntry:
    """
    represents one entry in model training history. provides appropriate fields that
    should be populated when recording history
    """

    # basic info
    dataset_name: str
    dataset_path: str

    # training args
    epoch: int  # remember to update this (this is the epochs trained so far)
    batch_size: int
    learning_rate: float
    weight_decay: float
    sweep_id: str
    run_name: str
    run_url: str
    checkpoint_dir: str  # this is the directory where the model checkpoint is saved
    freeze_embeddings: bool

    # outcomes
    eval_acc: float


class ModelWrapper(ABC):
    """
    this model wrapper treats the model as a first-class entity unlike in many training recipes in the field.
    the model(wrapper) is now responsible to train itself, and to evaluate itself on a supplied dataset.
    """

    model: typing.Union[HookedTransformer, torch.nn.Transformer]
    # we want to document the unique identification of the dataset a model has been trained on,
    history: typing.List[typing.Union[TrainingHistoryEntry, typing.Dict]] = None

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.from_pretrained is None:
            # if no pretrained path is supplied, we initialize the model from scratch
            logger.info(f"initializing model from scratch with config: {config}")
            self.model = HookedTransformer(
                HookedTransformerConfig(
                    d_head=config.d_head,
                    **{
                        k: v
                        for k, v in dataclasses.asdict(config).items()
                        if k != "from_pretrained"
                    },
                )
            )
        else:
            # if we're asked to load from a pretrained checkpoint, we load the model
            # using the stored config rather than the supplied config
            # note that any passed options about model parameters will be ignored!
            # we should make sure the user is aware of this.
            logger.warning(f"loading model from checkpoint: {config.from_pretrained}")
            logger.warning(
                f"any additional options passed via `ModelConfig` will be ingored!\n\t{config}"
            )
            self.load_checkpoint(config.from_pretrained)

        if self.history is None:
            self.history = []
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_dir: typing.Union[str, Path]):
        """
        `checkpoint_dir` points to a directory containing:
        - `config.yaml` which contains the `ModelConfig`
        - `*.pth`: a single .pth file that contains the model state_dict
        - `history.yaml` which details the training history (this is inherited and appended
            to the existing history, so a model that has been trained first on dataset X and then Y
            will say so in its history)
        """
        # 0. convert to Path
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # 1. load config
        with open(checkpoint_dir / "config.yaml", "r") as f:
            _config = ModelConfig(**yaml.load(f, Loader=yaml.FullLoader))
        logger.info(f"loaded config for pretrained model:\n\t{_config}")

        # 2. load history
        with open(checkpoint_dir / "history.yaml", "r") as f:
            self.history = yaml.load(f, Loader=yaml.FullLoader)

        # 3. load model
        # 3.1 load the state dict

        # NOTE: may be worth supporting state dicts other than `best_model.pth`,
        # e.g. `epoch_{epoch}.pth` for taking a model trained for X epochs
        _state_dict_path = list(checkpoint_dir.glob("*.pth"))
        if len(_state_dict_path) != 1:
            raise ValueError(
                f"expected exactly one .pth file in {checkpoint_dir}, found {_state_dict_path}"
            )
        [_state_dict_path] = _state_dict_path
        # vocab_path = os.path.join(root_dir, d, "vocab.json")

        # 3.2 initialize a model instance just based on the config (this will have
        # random weights, but we are about to overwrite them)
        self.model = HookedTransformer(
            HookedTransformerConfig(
                d_head=_config.d_head,
                **{
                    k: v
                    for k, v in dataclasses.asdict(_config).items()
                    if k != "from_pretrained"
                },
            )
        )

        # 3.3 load the state dict into the model: this should overwrite the weights
        _state_dict = torch.load(_state_dict_path, map_location=self.device)
        self.model.load_and_process_state_dict(
            _state_dict,
            center_unembed=True,  # this shifts the unembedding matrix weights to be centered around 0
            center_writing_weights=True,  # this shifts the weights written to residual stream to be centered around 0
            fold_ln=False,
            refactor_factored_attn_matrices=True,
        )
        logger.info(f"finished loading model state dict from {_state_dict_path}")

    def save_checkpoint(
        self, checkpoint_dir: typing.Union[str, Path], epoch_num: int = None
    ):
        """
        saves model.state_dict(), config, and training history to checkpoint_dir.
        by default saves under 'best_model.pth' (overwriting if needed) unless an explicit
        epoch number is supplied, in which case, it is used as 'epoch_{epoch}.pth'.
        """
        # 0. convert to Path
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # 0.1 if wandb.run.sweep_id is available, use it
        if self.history[-1].sweep_id is not None:
            checkpoint_dir /= self.history[-1].sweep_id

        # 0.2 if a run name is available, use it
        if self.history[-1].run_name is not None:
            checkpoint_dir /= self.history[-1].run_name
        else:
            # else, use a random prefix to avoid collisions
            import uuid

            # generate a random UUID
            random_string = str(uuid.uuid4())
            checkpoint_dir /= random_string[:6]

        self.history[-1].checkpoint_dir = str(checkpoint_dir)

        # 1. save model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        checkpoint_path = (
            checkpoint_dir / "best_model.pth"
            if epoch_num is None
            else checkpoint_dir / "checkpoints" / f"epoch_{epoch_num}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)

        # 2. save config
        config_path = checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(self.config), f)

        def convert_dataclass_if_needed(obj):
            """
            convert dataclass to dict if needed
            """
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return obj

        # 3. save training history
        history_path = checkpoint_dir / "history.yaml"
        with open(history_path, "w") as f:
            yaml.dump([*map(convert_dataclass_if_needed, self.history)], f)

        logger.info(f"saved model checkpoint to {checkpoint_path}")

    def set_embeddings(self, embeddings: typing.Union[np.ndarray, torch.Tensor]):
        """
        explicitly set the embeddings of the model to a supplied weight matrix W_E.
        the dimensionality of the matrix should be `vocab_size x d_model` as initialized (check
        `self.config`)
        """
        raise NotImplementedError

    def train(
        self,
        dataset: GeneratedCachedDataset,
        training_config: TrainingConfig,
        eval_dataset: GeneratedCachedDataset = None,
        test_dataset: GeneratedCachedDataset = None,
    ):
        """
        given an `eval_dataset`, periodically evaluates model on eval_dataset and logs the results
        """

        # create an entry for history logging, which will be updated as we go
        self.history += [
            TrainingHistoryEntry(
                dataset_name=repr(dataset),
                dataset_path=str(dataset.config.basedir),
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                freeze_embeddings=training_config.freeze_embeddings,
                sweep_id=wandb.run.sweep_id,
                run_name=wandb.run.name,
                run_url=wandb.run.get_url(),
                checkpoint_dir=None,  # to be filled in later
                epoch=0,  # to be filled in later
                eval_acc=None,  # to be filled in later
            )
        ]

        train_dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

        @dataclasses.dataclass
        class TrainingState:
            """
            this class is responsible for keeping track of the training state;
            it has a `step` property that is a function of the epoch, the epoch step,
            the dataset length, and batch size, and is computed on the fly and is
            therefore a function decorated with `@property`.
            when serializing this class, the `step` property will not be serialized
            automatically, so you should explicitly log it if you want to keep track
            """

            epoch: int = 0
            epoch_step: int = 0
            best_val_loss: float = np.inf
            best_val_epoch: int = -1

            @property
            def step(self):
                return self.epoch_step + np.ceil(
                    self.epoch * len(dataset) / training_config.batch_size
                )

        if training_config.log_predictions:
            predictions_table = wandb.Table(
                columns=[
                    "epoch",  # so that we can observe the evolution of the model's predictions over time
                    "example_ix",
                    "eval_example",
                    "eval_prediction",
                    "eval_labels",
                ]
            )
        else:
            predictions_table = None

        # set the model up for training
        # set up the optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        state = TrainingState()
        for state.epoch in tqdm(range(total := training_config.epochs), total=total):
            ################################
            #### begin epoch            ####
            ################################
            # set the model to training mode at the beginning of each epoch, since there is
            # no guarantee that it will still be in training mode from the previous epoch
            # if we went into the eval subroutine
            # NOTE this might not matter, since we don't use standard language modeling
            # design decisions like dropout
            self.model.train()

            # freeze model embeddings (and unembeddings) if requested
            if training_config.freeze_embeddings:
                for param in self.model.embed.parameters():
                    param.requires_grad = False
                for param in self.model.unembed.parameters():
                    param.requires_grad = False

            self.history[-1].epoch = state.epoch

            for state.epoch_step, inputs in enumerate(train_dataloader):
                loss = self._step(inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        "train_loss": loss.item(),
                    }
                )

                # evaluate the model when you reach the logging step within the epoch
                log_every_steps = (
                    len(dataset)
                    // training_config.batch_size
                    // training_config.logging_steps_per_epoch
                )
                if (
                    training_config.logging_steps_per_epoch
                    and state.epoch_step % log_every_steps == 0
                ):
                    ################################
                    # eval loop mid-epoch at however-many logging steps
                    eval_loss, eval_acc = self.evaluate(
                        eval_dataset,
                        # we will not be passing the predictions table
                        # to the eval loop; predictions will be logged at
                        # the end of each epoch
                        train_epoch=None,
                        predictions_table=None,
                    )
                    # update latest known eval_acc
                    self.history[-1].eval_acc = float(eval_acc)
                    wandb.log(
                        wandb_logged := {
                            **dataclasses.asdict(state),
                            "step": state.step,
                            "eval_loss": eval_loss,
                            "eval_acc": eval_acc,
                        }
                    )
                    logger.info(
                        f"------------- {state.epoch_step = } {eval_loss = }, {eval_acc = }"
                    )
                    # end eval loop mid-epoch at however-many logging steps
                    ################################

            if (
                training_config.logging_steps
                and state.epoch % training_config.logging_steps == 0
            ):
                ################################
                # eval once at the end of every epoch
                eval_loss, eval_acc = self.evaluate(
                    eval_dataset,
                    train_epoch=state.epoch,
                    # passing the predictions table to the eval loop for
                    # logging predictions and heatmaps over logits
                    predictions_table=predictions_table,
                )
                # update latest known eval_acc
                self.history[-1].eval_acc = float(eval_acc)

                logger.info(f"EVAL: {state.epoch = } {eval_loss = }, {eval_acc = }")

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                    }
                )
                logger.debug(f"{wandb_logged = }")

                # check if we had an improvement in validation loss
                if eval_loss < state.best_val_loss:
                    logger.info(
                        f"found new best validation loss: {eval_loss} < {state.best_val_loss}"
                    )
                    state.best_val_loss = eval_loss
                    state.best_val_epoch = state.epoch
                    # update latest known eval_acc
                    self.history[-1].eval_acc = float(eval_acc)
                    self.save_checkpoint(training_config.checkpoint_dir)
                # end eval at the end of epoch
                ################################

            # if saving strategy is epoch, then make a call to save anyway
            if training_config.save_strategy == "epoch":
                if (
                    training_config.save_steps
                    and state.epoch % training_config.save_steps == 0
                ):
                    self.save_checkpoint(
                        training_config.checkpoint_dir,
                        epoch_num=state.epoch,
                    )

            ################################
            #### end epoch              ####
            ################################

        if predictions_table is not None:
            wandb.log({"predictions": predictions_table})

        if training_config.do_test and test_dataset is not None:
            test_table = wandb.Table(
                columns=[
                    "epoch",  # so that we can observe the evolution of the model's predictions over time
                    "test_step",  # this is the step within the training epoch
                    "test_example",
                    "test_prediction",
                    "test_labels",
                ]
            )
            test_loss, test_acc = self.test(test_dataset, test_table)
            logger.info(f"TEST: {test_loss = }, {test_acc = }")
            wandb.log(
                {
                    "epoch": state.epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_predictions": test_table,
                }
            )

    def test(
        self, dataset: GeneratedCachedDataset, test_predictions_table: wandb.Table
    ):
        """
        evaluates the model on the test set
        """
        loss, acc = self.evaluate(dataset, predictions_table=test_predictions_table)
        return loss, acc

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

        inputs["answer_locations"] = inputs["answer_locations"].to(self.device)
        inputs["token_ids"] = inputs["token_ids"].to(self.device)

        # shape of logits: (b, seq_len, |V|)
        logits = self.model(inputs["token_ids"])

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

            return loss, gathered_logits, gathered_answers, gathered_labels
        else:
            loss = compute_masked_loss(logits, inputs, return_outputs=return_outputs)
            return loss


def compute_masked_loss(
    logits: torch.Tensor,
    inputs: typing.Dict[str, torch.Tensor],
    return_outputs=False,
) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
    """
    this method acts on:
        - `logits` from a model (this can be softmaxed to get a distribution over the vocabulary elsewhere)
        - `inputs` is the dictionary of tensors supplied to the model, which includes the key `input_ids`
            and `answer_locations` which provides a 0-1 mask over which tokens in the sequence should be
            considered 'predictions' (only these are used for loss computation)
        - `return_outputs` will cause the method to return the gathered logits, argmax-softmaxed logit answers, and true labels
            all as a dictionary with the keys loss, gathered_logits, gathered_answers, gathered_labels

    Example:
    --------
    ```python
    outputs = compute_masked_loss(logits, inputs, return_outputs=True)
        loss, gathered_logits, gathered_answers, gathered_labels = (
            outputs["loss"],
            outputs["gathered_logits"],
            outputs["gathered_answers"],
            outputs["gathered_labels"],
        )
    ```
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
