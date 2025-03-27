"""
run as: python -m workingmem
"""

#!/usr/bin/env python3
import typing
import dataclasses
import logging
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
    project_name: str = "wm-comp-limit-0"
    method: str = "random"
    metric: dict = dataclasses.field(
        default_factory=lambda: {"goal": "maximize", "name": "eval_acc"}
    )
    program: str = "run_wm.py"


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
    wandb.init(project=config.wandb.project_name, config=config)

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

    if config.wandb.create_sweep:
        sweep_config = dataclasses.asdict(config.wandb)
        sweep_config.update(
            {
                "parameters": {
                    # model parameters
                    "model.n_layers": {"value": 2},
                    "model.n_heads": {"values": [2]},
                    "model.d_model": {"values": [128]},
                    # "model.seed": {"values": [42, 43, 44, 45]},
                    # trainer parameters
                    "trainer.batch_size": {"value": 128},
                    "trainer.epochs": {"value": 60},
                    "trainer.learning_rate": {"value": 1e-3},
                    "trainer.weight_decay": {"value": 3e-5},
                    # dataset parameters
                    "dataset.n_train": {"value": 100_000},
                    "dataset.n_val": {"value": 1_000},
                    "dataset.n_test": {"value": 1_000},
                    "dataset.seq_len": {"value": 20},
                    "dataset.concurrent_items": {"value": 3},
                    "dataset.n_items": {"value": 50},
                    "dataset.concurrent_reg": {"values": [2]},  # !
                    "dataset.n_reg": {"values": [50]},  # !
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
        # set up wandb
        wandb.agent(config.wandb.sweep_id, count=1)

        # # we need to update some configs with the sweep values
        # hybrid_config = dataclasses.asdict(config)
        # hybrid_config.update(cfg_dict)

        # main(from_dict(MainConfig, hybrid_config))

        # import submitit
        # executor = submitit.AutoExecutor(folder="logs/slurm_logs")
        # executor.update_parameters(timeout_min=120, slurm_partition="cs-3090-gcondo")
        # config_array = []
        # jobs = executor.map_array(main, config_array)  # just a list of jobs

    else:  # run as normal in a single-run fashion using wandb only for logging
        wandb.init(project=config.wandb.project_name, config=config)
        main(config)
