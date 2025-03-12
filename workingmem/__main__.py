#!/usr/bin/env python3
import typing
import dataclasses
import logging
from collections import defaultdict

# 3rd party packages
import tyro
import wandb

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
    run_as_agent: bool = True
    sweep_id: str = None  # required if do_sweep is True
    project_name: str = "wm-comp-limit-0"
    method: str = "random"
    metric = {"goal": "minimize", "name": "eval_loss"}


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

    # set up the dataset
    logger.info("loading datasets")
    train_dataset = SIRDataset(config.dataset)
    eval_config = SIRConfig(**dataclasses.asdict(config.dataset))
    test_config = SIRConfig(**dataclasses.asdict(config.dataset))
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
    logger.info(f"{config}")

    if config.wandb.create_sweep:
        sweep_config = dataclasses.asdict(config.wandb)
        sweep_config.update(
            {
                "parameters": {
                    # model parameters
                    "model.n_layers": {"value": 2},
                    "model.n_heads": {"value": 2},
                    "model.d_model": {"value": 128},
                    # trainer parameters
                    "trainer.batch_size": {"value": 16},
                    "trainer.epochs": {"value": 60},
                    "trainer.learning_rate": {
                        "distribution": "uniform",
                        "max": 1e-3,
                        "min": 1e-5,
                    },
                    # dataset parameters
                    "dataset.n_train": {"value": 10_000},
                    "dataset.n_dev": {"value": 1_000},
                    "dataset.n_test": {"value": 1_000},
                    "dataset.concurrent_reg": {"value": 2},
                    "dataset.concurrent_items": {"value": 5},
                },
            },
        )

        sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
        logger.info(f"created sweep with id: {sweep_id} !")

    elif config.wandb.run_as_agent:
        # if we're doing a sweep, we need to update the config with the sweep values

        # set up wandb
        if config.wandb.run_as_agent:
            wandb.agent(config.wandb.sweep_id, count=1)
        wandb.init(project=config.wandb.project_name, config=config)

        wandb_config = wandb.config
        logger.info(
            f"running an agent part of sweep {config.wandb.sweep_id} with: {wandb_config}"
        )

        def deep_dict():
            return defaultdict(deep_dict)

        cfg_dict = deep_dict()

        def deep_insert(key, value):
            d = hybrid_config
            keys = key.split(".")
            for subkey in keys[:-1]:
                d = d[subkey]
            d[keys[-1]] = value

        for key, value in wandb_config.items():
            deep_insert(key, value)

        # we need to update some configs with the sweep values
        hybrid_config = dataclasses.asdict(config)
        hybrid_config.update(cfg_dict)

        main(MainConfig(**cfg_dict))

        # import submitit
        # executor = submitit.AutoExecutor(folder="logs/slurm_logs")
        # executor.update_parameters(timeout_min=120, slurm_partition="cs-3090-gcondo")
        # config_array = []
        # jobs = executor.map_array(main, config_array)  # just a list of jobs

    else:
        wandb.init(project=config.wandb.project_name, config=config)
        main(config)
