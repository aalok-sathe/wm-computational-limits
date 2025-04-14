"""
run as: python -m workingmem
"""

#!/usr/bin/env python3
import typing
import dataclasses
import yaml
import logging
import random
from pathlib import Path
from collections import defaultdict

# 3rd party packages
import tyro
import wandb
from dacite import from_dict

# local
from workingmem.model import (
    ModelWrapper,
    ModelConfig,
    TrainingConfig,
)
from workingmem.task import SIRDataset, SIRConfig

logging.basicConfig()
logger = logging.getLogger("workingmem")
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class WandbConfig:
    create_sweep: bool = False
    run_sweep: bool = False
    sweep_id: str = None  # required if do_sweep is True
    project_name: str = "wm-comp-limit-1"
    method: str = "random"
    metric: dict = dataclasses.field(
        default_factory=lambda: {"goal": "maximize", "name": "eval_acc"}
    )
    program: str = "run_wm.py"  # the program to run with a wandb sweep agent


@dataclasses.dataclass
class MainConfig:
    """
    Run a recipe of loading a dataset, training a model, and evaluating it.
    Coming soon: load a model from a checkpoint to cross-train or evaluate it (for this, we will need to implement training history recordkeeping).
    """

    model: ModelConfig
    dataset: SIRConfig
    trainer: TrainingConfig
    wandb: WandbConfig
    seed = None
    # for array jobs, the task id is useful to, e.g., resume from the Xth pretrained model
    array_task_id: typing.Union[int, None] = None
    filter_by_accuracy: bool = True

    def __post_init__(self):
        logger.info(f"running post-init hook to set seeds to {self.seed}")
        if self.seed is not None:
            # prefer to keep using the same dataset instance across model training seeds
            # unless the dataset seet is explicitly set, so we wont be setting
            # `self.dataset.seed` here.
            self.model.seed = self.seed
            self.trainer.seed = self.seed
            # additionally set the seed globally here?

            import torch
            import numpy as np

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)


def main(config: MainConfig):
    """
    given a config, train a model on an SIR dataset, evaluate, and test, all described
    as per config. wandb is used for logging regardless of sweep or not.
    """
    logger.info(f"running main with config: {config}")
    wandb.init(project=config.wandb.project_name, config=config, dir='~/scratch/wandb')

    # set up the dataset
    logger.info("loading datasets")
    train_dataset = SIRDataset(config.dataset)
    eval_config = from_dict(SIRConfig, dataclasses.asdict(config.dataset))
    test_config = from_dict(SIRConfig, dataclasses.asdict(config.dataset))
    eval_config.split, test_config.split = "val", "test"
    eval_dataset = SIRDataset(eval_config)
    test_dataset = SIRDataset(test_config)

    logger.info("train dataset size: %s", len(train_dataset))
    logger.info("eval dataset size: %s", len(eval_dataset))
    logger.info("test dataset size: %s", len(test_dataset))

    # set up the model
    logger.info("initializing model")

    # we need to explicitly set `d_vocab` if it isn't supplied via CLI, only if we're not
    # loading a model from disk
    if config.model.d_vocab is None:
        config.model.d_vocab = eval_dataset.vocab_size

    # if we're loading a pretrained model, check if an explicit model is passed, or a directory containing many models is
    # provided, in which case, we'd use the `config.array_task_id` to load the Xth model (modulo total models in dir)
    if (
        config.model.from_pretrained
        and len(list(Path(config.model.from_pretrained).glob("*.pth"))) == 0
    ):
        # enumerate subdirectories within this dirctory
        # and load the Xth model modulo the number of models in the directory
        models_dir = Path(config.model.from_pretrained)
        models_dir = list(models_dir.glob("*"))
        assert all(len(list(m.glob("*.pth"))) == 1 for m in models_dir), (
            f"malformed model checkpoints dir passed: {models_dir}"
        )

        if config.filter_by_accuracy:
            # filter models by the accuracy recorded in their history
            def filter_by_accuracy(m: Path, threshold=0.99) -> bool:
                with open(m / "history.yaml", "r") as f:
                    history = yaml.load(f, Loader=yaml.FullLoader)
                return history[-1]["eval_acc"] >= threshold

            prev_len = len(models_dir)
            models_dir = list(filter(filter_by_accuracy, models_dir))
            logger.info(
                f"filtering models by accuracy >= .99 in {models_dir}. {prev_len = }, {len(models_dir) = }"
            )

        # set `from_pretrained` path to the Xth model NOTE randomly for now:(
        config.model.from_pretrained = str(random.choice(models_dir))
        # record the new pretrained model path corresponding to the model we're actually using
        wandb.config.update(
            {"model.from_pretrained": str(config.model.from_pretrained)}
        )

    # once the `from_pretrained` path is set, we can just use the regular way to load the model,
    # since the `ModelWrapper` class will take care of loading the model from checkpoint
    model = ModelWrapper(config.model)

    if config.dataset.split == "train":
        # train the model
        logger.info("Training the model")
        model.train(
            train_dataset,
            config.trainer,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
        )

        logger.info("Finished.")


if __name__ == "__main__":
    config = tyro.cli(MainConfig)

    # case 1 is we create a new sweep
    if config.wandb.create_sweep:
        sweep_config = dataclasses.asdict(config.wandb)
        sweep_config.update(
            {
                "parameters": {
                    # "filter_by_accuracy": {"value": "True"},
                    # model parameters
                    "model.from_pretrained": {
                        # "value": "model_checkpoints/evcxg3kc/"  # n_reg 50 exposure task
                        # "value": "model_checkpoints/vc7i09gs/"  # n_reg 2 concurrent 2
                        "value": None
                    },  # !
                    "model.n_layers": {"value": 2},
                    "model.n_heads": {"values": [2]},
                    "model.d_model": {"values": [32]},  # !
                    # "model.seed": {"values": [42, 43, 44, 45]},
                    # trainer parameters
                    "trainer.freeze_embeddings": {"value": "False"},  # !
                    "trainer.batch_size": {"value": 64},
                    "trainer.epochs": {"value": 60},
                    "trainer.learning_rate": {"value": 1e-3},
                    "trainer.weight_decay": {"value": 3e-5},
                    "trainer.checkpoint_dir": {"value": "model_checkpoints/"},
                    # dataset parameters
                    "dataset.n_train": {"value": 100_000},  # !
                    "dataset.n_val": {"value": 1_000},
                    "dataset.n_test": {"value": 1_000},
                    "dataset.seq_len": {"value": 14},
                    "dataset.concurrent_items": {"value": 3},
                    "dataset.n_items": {"value": 50},
                    "dataset.concurrent_reg": {"values": [4]},  # !
                    "dataset.n_reg": {"values": [4]},  # !
                    "dataset.heldout_reg": {"value": 0},
                    "dataset.heldout_items": {"value": 0},
                    "dataset.ignore_prob": {"value": 0.5},
                },
            },
        )

        sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
        logger.info(f"created sweep with id: {sweep_id} !")

    elif config.wandb.run_sweep:
        # if we're doing a sweep, we need to update the config with the sweep values
        logger.info(
            f"running an agent part of sweep {config.wandb.sweep_id} with: {wandb.config}"
        )
        # this uses the wandb sweep_id to initialize a single wandb agent and runs
        # the designated script as specified in the `WandbConfig` argument that was
        # used when creating the sweep (see the first clause of this if-statement)
        wandb.agent(config.wandb.sweep_id, count=1)

    else:  # run as normal in a single-run fashion using wandb only for logging
        # wandb.init(project=config.wandb.project_name, config=config)
        main(config)
