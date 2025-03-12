#!/usr/bin/env python3
import typing
from types import SimpleNamespace
import dataclasses
import logging

# 3rd party packages
import tyro
import wandb
from transformers import TrainingArguments

# local
from workingmem.model import (
    ModelWrapper,
    HookedTransformer,
    HookedTransformerConfig,
    ModelConfig,
    TrainingConfig,
)
from workingmem.task import SIRDataset, SIRConfig

logging.basicConfig()
logger = logging.getLogger("workingmem")
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class WandbConfig:
    use_wandb: bool = True
    project_name: str = "wm-comp-limit-0"
    method: str = "random"
    metric_name: str = "eval_loss"
    metric_goal: str = "minimize"


@dataclasses.dataclass
class MainConfig:
    model: ModelConfig
    dataset: SIRConfig
    trainer: TrainingConfig
    wandb: WandbConfig
    project_name: str = "wm-comp-limit-0"
    seed = None

    def __post_init__(self):
        if self.seed is not None:
            self.dataset.seed = self.seed
            self.model.seed = self.seed
            self.trainer.seed = self.seed
            # additionally set the seed here?


def main(config: MainConfig): ...


if __name__ == "__main__":
    config = tyro.cli(MainConfig)
    logger.info(f"{config}")

    if config.wandb.use_wandb:
        wandb.init(project=config.project_name, config=config)

    # set up the dataset
    logger.info("loading the dataset")
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
