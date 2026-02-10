# stdlib
import dataclasses
import typing
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import yaml
import math

# installed packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

# local
from workingmem.task.interface import GeneratedCachedDataset
from workingmem.model.interface import (
    AbstractPytorchModel,
    TrainingConfig,
    TrainingHistoryEntry,
    ModelConfig,
    # TransformerConfig,
    # RNNConfig,
    compute_masked_loss,
)


logger = logging.getLogger("workingmem")
logger.setLevel(logging.DEBUG)

# Constants for positional embedding types
POSITIONAL_EMBEDDING_ROTARY = "rotary"
POSITIONAL_EMBEDDING_STANDARD = "standard"
POSITIONAL_EMBEDDING_NONE = None

# Constant for MLP dimension check
ATTENTION_ONLY_MLP_DIM = 0


class ModelWrapper(ABC):
    """
    this model wrapper treats the model as a first-class entity unlike in many training recipes in the field.
    the model(wrapper) is now responsible to train itself, and to evaluate itself on a supplied dataset.
    """

    model: AbstractPytorchModel
    # we want to document the unique identification of the dataset a model has been trained on
    history: typing.List[typing.Union[TrainingHistoryEntry, typing.Dict]] = None

    @abstractmethod
    def _init_model(self, config: ModelConfig):
        pass

    def load_state_dict(
        self, state_dict: typing.Dict[str, torch.Tensor], config: ModelConfig = None
    ):
        """
        simply make a call to the underlying model's `load_state_dict` method
        as provided by any standard pytorch model except in the case of a
        HookedTransformer, where we have to call the
        `load_and_process_state_dict` method
        """
        self.model.load_state_dict(state_dict)

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.from_pretrained is None:
            # if no pretrained path is supplied, we initialize the model from scratch
            logger.info(f"initializing model from scratch with config: {config}")
            # set the seed for initializing the model weights
            if config.seed is not None:
                logger.info(f"setting MODEL random seed to {config.seed}")
                torch.manual_seed(int(config.seed))
                np.random.seed(int(config.seed))

            # we call the abstract method _init_model which should be implemented in subclasses
            self._init_model(config)

        else:
            # if we're asked to load from a pretrained checkpoint, we load the model
            # using the stored config rather than the supplied config
            # note that any passed options about model parameters will be ignored!
            # we should make sure the user is aware of this.
            logger.warning(f"loading model from checkpoint: {config.from_pretrained}")
            logger.warning(
                f"any additional options passed to `ModelConfig` will be ignored!\n\t{config}"
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
            self.config = _config  # NOTE: added 1/15/2026; seems that we were not updating self.config here before?
            # update the config with the checkpoint dir as the new `from_pretrained` path
            # NOTE: this is unnecessary if this method was called from __init__ since the config
            # would have been set to the checkpoint dir already---that is the preferred way.
            self.config.from_pretrained = checkpoint_dir
        logger.info(f"loaded config for pretrained model:\n\t{_config}")

        # 2. load history
        with open(checkpoint_dir / "history.yaml", "r") as f:
            self.history = yaml.load(f, Loader=yaml.FullLoader)

        # 3. load model
        # 3.1 load the state dict

        ################################################################
        # TODO: may be worth supporting state dicts other than `best_model.pth`,
        ################################################################
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
        self._init_model(_config)

        # 3.3 load the state dict into the model: this should overwrite the weights
        _state_dict = torch.load(_state_dict_path, map_location=self.device)
        self.load_state_dict(_state_dict, _config)

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

    def _deactivate_positional_embeddings(self) -> None:
        """placeholder hunk for use by the TransformerModel subclass"""
        raise NotImplementedError

    def set_embeddings(self, embeddings: typing.Union[np.ndarray, torch.Tensor]):
        """
        explicitly set the embeddings of the model to a supplied weight matrix W_E.
        the dimensionality of the matrix must be `vocab_size x d_model` (check `self.config`)
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
        given an `eval_dataset` and `test_dataset`, periodically evaluates model and logs the results
        """

        # create an entry for history logging, which will be updated as we go
        self.history += [
            TrainingHistoryEntry(
                dataset_name=repr(dataset),
                dataset_path=str(dataset.config.basedir),
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                sparsity=training_config.sparsity,
                weight_decay=training_config.weight_decay,
                freeze_embeddings=training_config.freeze_embeddings,
                sweep_id=(wandb.run.sweep_id if wandb.run else None),
                run_name=(wandb.run.name if wandb.run else None),
                run_url=(wandb.run.get_url() if wandb.run else None),
                checkpoint_dir=None,  # to be filled in later
                epoch=0,  # to be filled in later
                eval_acc=None,  # to be filled in later
                eval_macro_acc=None,  # to be filled in later
                test_acc=None,  # to be filled in later
                test_macro_acc=None,  # to be filled in later
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
            best_val_acc: float = 0.0
            best_val_epoch: int = -1
            # cumulative AUC, to be updated during training. this is simply measured as an integration of eval_acc over epochs
            # so, the max possible value is 1.0 x num_epochs. for instance, a model that achieves 1.0 accuracy starting from
            # epoch 0 will have cumAUC = num_epochs
            cumAUC: float = 0.0

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
        scaler = torch.amp.grad_scaler.GradScaler()

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

            # NOTE: as of yet NotImplemented: there is no such parameter.
            # if training_config.freeze_attention:
            #     for param in self.model.attn.parameters():
            #         param.requires_grad = False
            #     for param in self.model.attn_norm.parameters():
            #         param.requires_grad = False

            self.history[-1].epoch = state.epoch

            for state.epoch_step, inputs in enumerate(train_dataloader):
                if state.best_val_acc >= 0.999:
                    logger.warning(
                        f"best validation accuracy {state.best_val_acc:.3f} reached, skipping training loop to directly evaluate the model"
                    )
                else:
                    torch.cuda.empty_cache()
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        loss = self._step(
                            inputs,
                            sparsity=training_config.sparsity,
                            mask_answer_tokens=training_config.mask_answer_tokens,
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()
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
                    eval_result = self.evaluate(
                        eval_dataset,
                        # we will not be passing the predictions table
                        # to the eval loop; predictions will be logged at
                        # the end of each epoch
                        train_epoch=None,
                        predictions_table=None,
                        mask_answer_tokens=training_config.mask_answer_tokens,
                    )
                    eval_loss, eval_acc, eval_macro_acc = (
                        eval_result["loss"],
                        eval_result["acc"],
                        eval_result["macro_acc"],
                    )
                    test_result = self.test(test_dataset, test_predictions_table=None)
                    test_loss, test_acc, test_macro_acc = (
                        test_result["loss"],
                        test_result["acc"],
                        test_result["macro_acc"],
                    )
                    # update latest known eval_acc
                    self.history[-1].eval_acc = float(eval_acc)
                    self.history[-1].test_acc = float(test_acc)
                    self.history[-1].eval_macro_acc = float(eval_macro_acc)
                    self.history[-1].test_macro_acc = float(eval_macro_acc)
                    wandb.log(
                        wandb_logged := {
                            **dataclasses.asdict(state),
                            "step": state.step,
                            "eval_loss": eval_loss,
                            "eval_acc": eval_acc,
                            "eval_macro_acc": eval_macro_acc,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "test_macro_acc": test_macro_acc,
                        }
                    )
                    logger.info(
                        f"------------- {state.epoch_step = } {eval_loss = :.3f}, {test_loss = :.3f},  {eval_acc = :.3f}, {test_acc = :.3f}"
                    )
                    # end eval loop mid-epoch at however-many logging steps
                    ################################
                    self.model.train()

            if (
                training_config.logging_steps
                and state.epoch % training_config.logging_steps == 0
            ):
                ################################
                # eval once at the end of every epoch
                eval_result = self.evaluate(
                    eval_dataset,
                    train_epoch=state.epoch,
                    # passing the predictions table to the eval loop for
                    # logging predictions and heatmaps over logits
                    predictions_table=predictions_table,
                    mask_answer_tokens=training_config.mask_answer_tokens,
                )
                eval_loss, eval_acc, eval_macro_acc = (
                    eval_result["loss"],
                    eval_result["acc"],
                    eval_result["macro_acc"],
                )
                test_result = self.test(test_dataset, test_predictions_table=None)
                test_loss, test_acc, test_macro_acc = (
                    test_result["loss"],
                    test_result["acc"],
                    test_result["macro_acc"],
                )
                # update latest known eval_acc
                self.history[-1].eval_acc = float(eval_acc)
                self.history[-1].eval_macro_acc = float(eval_macro_acc)
                state.cumAUC += eval_acc * 1

                logger.info(
                    f"EVAL: {state.epoch = } {eval_loss = }, {eval_acc = }, {test_loss = }, {test_acc = }"
                )

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "eval_macro_acc": eval_macro_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_macro_acc": test_macro_acc,
                        "cumAUC": state.cumAUC,
                        "cumAUC_normalized": state.cumAUC / state.epoch,
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
                    self.history[-1].eval_macro_acc = float(eval_macro_acc)
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
            test_result = self.test(
                test_dataset,
                test_table,
                mask_answer_tokens=training_config.mask_answer_tokens,
            )
            test_loss, test_acc, test_macro_acc = (
                test_result["loss"],
                test_result["acc"],
                test_result["macro_acc"],
            )
            logger.info(f"TEST: {test_loss = }, {test_acc = }, {test_macro_acc = }")
            wandb.log(
                {
                    "epoch": state.epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_macro_acc": test_macro_acc,
                    "test_predictions": test_table,
                }
            )

    def test(
        self,
        dataset: GeneratedCachedDataset,
        test_predictions_table: wandb.Table = None,
        mask_answer_tokens: bool = True,
    ):
        """
        evaluates the model on the test set
        """
        return self.evaluate(
            dataset,
            predictions_table=test_predictions_table,
            mask_answer_tokens=mask_answer_tokens,
        )

    def evaluate(
        self,
        dataset: GeneratedCachedDataset,
        train_epoch: int = None,
        predictions_table: wandb.Table = None,
        batch_size: int = 128,
        return_predictions: bool = False,
        mask_answer_tokens=True,
    ) -> dict:
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
            batch_size=batch_size,  # TODO, should we parameterize this?
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        losses = []
        predictions = []
        actual_labels = []
        input_sequences = []

        with torch.no_grad():
            for eval_step, inputs in enumerate(eval_dataloader):
                torch.cuda.empty_cache()
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    loss, answer_logits, answers, labels = self._step(
                        inputs,
                        sparsity=0.0,
                        return_outputs=True,
                        mask_answer_tokens=mask_answer_tokens,
                    )
                # we have a single loss value per batch (this is a fine approximation)
                losses += [loss.item()]
                # answers and labels are of the shape (b, seq_len)
                predictions += [answers.detach().cpu().numpy()]
                actual_labels += [labels.detach().cpu().numpy()]

                # log the first batch of eval examples and predictions to `wandb`
                if train_epoch is not None and predictions_table is not None:
                    for example_ix in range(len(inputs["tokens"])):
                        predictions_table.add_data(
                            train_epoch,
                            example_ix,  # corresponds to batch
                            inputs["tokens"][example_ix],
                            dataset.tokenizer.decode(
                                answers[example_ix].detach().cpu().tolist()
                            ),
                            dataset.tokenizer.decode(
                                labels[example_ix].detach().cpu().tolist()
                            ),
                        )
                if return_predictions:
                    for example_ix in range(len(inputs["tokens"])):
                        input_sequences += [inputs["tokens"][example_ix]]

        # now `predictions` is of shape (N_batches, batch_size, seq_len)
        # we want it to be of shape (N_batches * batch_size, seq_len)
        predictions = np.concat(predictions)
        actual_labels = np.concat(actual_labels)
        # predictions.shape = (N_batches * batch_size, seq_len)
        # actual_labels.shape = (N_batches * batch_size, seq_len)

        # we want to aggregate over each example in val set rather than each individual answer location
        eval_num_correct = np.sum(
            all(predictions[i] == actual_labels[i])
            for i in range(actual_labels.shape[0])
        )
        acc = np.mean(predictions == actual_labels)

        logger.info(f"percent trials correct: {acc:.5f}")
        logger.info(f"# sequences correct: {eval_num_correct} / {len(actual_labels)}")

        if return_predictions:
            return {
                "loss": np.mean(losses),
                "acc": acc,
                "macro_acc": eval_num_correct / len(actual_labels),
                "predictions": predictions,
                "actual_labels": actual_labels,
                "input_sequences": input_sequences,
            }

        return {
            "loss": np.mean(losses),
            "acc": acc,
            "macro_acc": eval_num_correct / len(actual_labels),
        }

    def _step(
        self,
        inputs: typing.Dict[str, torch.Tensor],
        sparsity: float = 0.0,
        return_outputs=False,
        mask_answer_tokens=True,
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        this method is responsible for computing the loss and optionally the labels
        batch of a batch of inputs
        """

        inputs["token_ids"] = inputs["token_ids"].to(self.device)
        inputs["answer_locations"] = inputs["answer_locations"].to(self.device)
        inputs["answer_locations"].requires_grad = False  # not backprop-able

        # a variation we can do here is to remove the actual answer tokens from the inputs
        # so this is less like a language modeling task and more like a classification task
        # (which it already is in principle due to not receiving loss on anything but the
        # answers). however, this way, it should take away the answers implicit in the input
        # text
        inputs["answers"] = inputs["token_ids"] * inputs["answer_locations"].to(
            self.device
        )  # only the relevant token_ids remain non-zeroed-out as `answers`

        if mask_answer_tokens:
            logger.debug(
                f"removing answer tokens from input: {inputs['token_ids'].gt(0).sum() = }"
            )
            inputs["token_ids"] = inputs["token_ids"] * (1 - inputs["answer_locations"])
            logger.debug(
                f"\tAFTER removing answer tokens from input: {inputs['token_ids'].gt(0).sum() = }"
            )

        # shape of logits: (b, seq_len, |V|)
        # TODO: ERROR: mismatch for model_class 'rnn' ---
        # TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
        logits = self.model(inputs["token_ids"])

        if return_outputs:
            outputs = compute_masked_loss(
                logits, inputs, sparsity=sparsity, return_outputs=return_outputs
            )
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
            loss = compute_masked_loss(
                logits, inputs, sparsity=sparsity, return_outputs=return_outputs
            )
            return loss


class RNNModelWrapper(ModelWrapper):
    """
    RNN-based language model wrapper that is compatible with nnsight for interpretability.
    The model architecture exposes individual components (embedding, rnn, output_layer) 
    to enable easier access to activations when using nnsight tracing.
    """

    class _RNNCore(torch.nn.Module):
        """
        Core RNN module that returns only the output tensor, not the hidden state,
        for compatibility with the existing ModelWrapper interface.
        """
        def __init__(self, input_size, hidden_size, num_layers, batch_first, nonlinearity):
            super().__init__()
            self.rnn = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                nonlinearity=nonlinearity,
                bidirectional=False,
            )
        
        def forward(self, input: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
            output, hidden = self.rnn(input, hx)
            return output

    class _RNNLanguageModel(torch.nn.Module):
        """
        Complete RNN language model with exposed components for nnsight compatibility.
        """
        def __init__(self, vocab_size, d_model, hidden_size, num_layers, act_fn):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.rnn_core = RNNModelWrapper._RNNCore(
                input_size=d_model,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity=act_fn,
            )
            self.output_layer = torch.nn.Linear(hidden_size, vocab_size)
            
            # Aliases for compatibility with existing freeze_embeddings code
            self.embed = self.embedding
            self.unembed = self.output_layer
        
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embedding(input_ids)
            x = self.rnn_core(x)
            logits = self.output_layer(x)
            return logits

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        """
        Initialize an RNN language model with exposed components for nnsight interpretability.
        The model converts input_ids to embeddings, processes them through an RNN,
        and projects to vocabulary space for language modeling.
        
        Components are exposed as: embedding, rnn_core, output_layer for easy access.
        """
        self.model = self._RNNLanguageModel(
            vocab_size=config.d_vocab,
            d_model=config.d_model,
            hidden_size=config.d_hidden,
            num_layers=config.n_layers,
            act_fn=config.act_fn,
        )


class LSTMModelWrapper(RNNModelWrapper):
    """
    LSTM-based language model wrapper that is compatible with nnsight for interpretability.
    The model architecture exposes individual components (embedding, lstm_core, output_layer)
    to enable easier access to activations when using nnsight tracing.
    """
    
    class _LSTMCore(torch.nn.Module):
        """
        Core LSTM module that returns only the output tensor, not the hidden states,
        for compatibility with the existing ModelWrapper interface.
        """
        def __init__(self, input_size, hidden_size, num_layers, batch_first):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=False,
            )
        
        def forward(self, input: torch.Tensor, hx: typing.Tuple = None) -> torch.Tensor:
            output, (hidden, cell) = self.lstm(input, hx)
            return output

    class _LSTMLanguageModel(torch.nn.Module):
        """
        Complete LSTM language model with exposed components for nnsight compatibility.
        """
        def __init__(self, vocab_size, d_model, hidden_size, num_layers):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.lstm_core = LSTMModelWrapper._LSTMCore(
                input_size=d_model,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output_layer = torch.nn.Linear(hidden_size, vocab_size)
            
            # Aliases for compatibility with existing freeze_embeddings code
            self.embed = self.embedding
            self.unembed = self.output_layer
        
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embedding(input_ids)
            x = self.lstm_core(x)
            logits = self.output_layer(x)
            return logits

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        """
        Initialize an LSTM language model with exposed components for nnsight interpretability.
        The model converts input_ids to embeddings, processes them through an LSTM,
        and projects to vocabulary space for language modeling.
        
        Components are exposed as: embedding, lstm_core, output_layer for easy access.
        """
        self.model = self._LSTMLanguageModel(
            vocab_size=config.d_vocab,
            d_model=config.d_model,
            hidden_size=config.d_hidden,
            num_layers=config.n_layers,
        )


class PositionalEncoding(nn.Module):
    """Standard learned or sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        
        if learnable:
            # Learned positional embeddings
            self.pos_embedding = nn.Embedding(max_len, d_model)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encodings of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if self.learnable:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            return self.pos_embedding(positions)
        else:
            return self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in https://arxiv.org/abs/2104.09864
    
    Note: This implementation applies rotary embeddings to the full sequence embeddings.
    For true RoPE behavior in multi-head attention, custom attention layers would be needed.
    This implementation approximates RoPE by rotating the embeddings before passing to the transformer.
    """
    
    def __init__(self, dim: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for rotary embeddings
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update the cached cos/sin values if sequence length changed."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Repeat frequencies for full dimension
            emb = torch.cat((freqs, freqs), dim=-1)
            # Add batch and head dimensions for broadcasting
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with rotary embeddings applied, shape (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)
        
        # Apply rotary embedding
        # Shape: (seq_len, d_model)
        cos = self._cos_cached[:seq_len, :x.shape[-1]]
        sin = self._sin_cached[:seq_len, :x.shape[-1]]
        
        # Broadcast to batch dimension
        cos = cos.unsqueeze(0)  # (1, seq_len, d_model)
        sin = sin.unsqueeze(0)  # (1, seq_len, d_model)
        
        # Apply rotation
        x_rotated = (x * cos) + (self.rotate_half(x) * sin)
        
        return x_rotated


class TransformerModelWrapper(ModelWrapper):
    """
    Transformer-based language model wrapper compatible with nnsight for interpretability.
    Uses PyTorch's nn.Transformer as the core architecture, similar to RNN/LSTM implementations.
    """
    
    class _TransformerLanguageModel(nn.Module):
        """
        Complete Transformer language model with exposed components for nnsight compatibility.
        Supports multiple positional embedding types: standard (learned/sinusoidal), rotary, or none.
        """
        
        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            d_ff: int,
            max_len: int,
            act_fn: str = "relu",
            positional_embedding_type: typing.Union[str, None] = "standard",
        ):
            super().__init__()
            self.d_model = d_model
            self.positional_embedding_type = positional_embedding_type
            
            # Token embedding
            self.embedding = nn.Embedding(vocab_size, d_model)
            
            # Positional encoding
            if positional_embedding_type == POSITIONAL_EMBEDDING_STANDARD:
                self.pos_encoder = PositionalEncoding(d_model, max_len, learnable=True)
            elif positional_embedding_type == POSITIONAL_EMBEDDING_ROTARY:
                # For rotary embeddings, use full d_model dimension
                self.pos_encoder = RotaryPositionalEmbedding(d_model, max_len)
            elif positional_embedding_type is POSITIONAL_EMBEDDING_NONE:
                self.pos_encoder = None
            else:
                raise ValueError(
                    f"Unknown positional_embedding_type: {positional_embedding_type}. "
                    f"Expected one of: {POSITIONAL_EMBEDDING_STANDARD}, {POSITIONAL_EMBEDDING_ROTARY}, or None"
                )
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                activation=act_fn,
                batch_first=True,
                norm_first=False,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )
            
            # Output projection layer
            self.output_layer = nn.Linear(d_model, vocab_size)
            
            # Aliases for compatibility with existing freeze_embeddings code
            self.embed = self.embedding
            self.unembed = self.output_layer
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights using Xavier/Glorot initialization."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the transformer.
            
            Args:
                input_ids: Tensor of shape (batch_size, seq_len)
            Returns:
                Logits of shape (batch_size, seq_len, vocab_size)
            """
            # Embed tokens
            x = self.embedding(input_ids) * math.sqrt(self.d_model)
            
            # Add positional encoding
            if self.pos_encoder is not None:
                pos_enc = self.pos_encoder(x)
                x = x + pos_enc
            
            # Create causal mask for autoregressive generation
            seq_len = input_ids.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
            
            # Pass through transformer encoder
            x = self.transformer_encoder(x, mask=mask, is_causal=True)
            
            # Project to vocabulary
            logits = self.output_layer(x)
            
            return logits

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        """
        Initialize a bare-bones Transformer model using PyTorch's nn.Transformer.
        This implementation is general-purpose and not tied to any specific architecture
        like GPT-2, similar to how RNN and LSTM models are implemented.
        
        Supports multiple positional embedding types:
        - "standard": Learned positional embeddings
        - "rotary": Rotary Position Embeddings (RoPE)
        - None: No positional embeddings
        """
        # Determine feedforward dimension
        # For attention-only models (d_mlp == 0), use 4 * d_model as default
        d_ff = config.d_mlp if config.d_mlp > ATTENTION_ONLY_MLP_DIM else 4 * config.d_model
        
        self.model = self._TransformerLanguageModel(
            vocab_size=config.d_vocab,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=d_ff,
            max_len=config.n_ctx,
            act_fn=config.act_fn,
            positional_embedding_type=config.positional_embedding_type,
        )

    def load_state_dict(
        self,
        state_dict: typing.Dict[str, torch.Tensor],
        _config: ModelConfig = None,
    ):
        """
        Load state dict into the transformer model.
        Uses standard PyTorch load_state_dict.
        """
        self.model.load_state_dict(state_dict)

    def _deactivate_positional_embeddings(self) -> None:
        """
        Deactivates the positional embedding by setting its weights to zero
        and freezing gradient updates.
        This method should only be called when positional_embedding_type is None
        at initialization, but can be used to zero out embeddings post-hoc.
        """
        if hasattr(self.model, 'pos_encoder') and self.model.pos_encoder is not None:
            if isinstance(self.model.pos_encoder, PositionalEncoding):
                if self.model.pos_encoder.learnable:
                    self.model.pos_encoder.pos_embedding.weight.data[:] = 0.0
                    self.model.pos_encoder.pos_embedding.weight.requires_grad = False
