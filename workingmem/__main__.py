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
    MaskedLossTrainer,
    TrainingConfig,
)
from workingmem.task import SIRDataset, SIRConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WandbConfig:
    use_wandb: bool = True
    project_name: str = "wm-comp-limit-0"
    method: str = "grid"
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
        # set up the sweep
        sweep_config = {}
        wandb.init(project=config.project_name, config=config)

    # set up the dataset
    dataset = SIRDataset(config.dataset)
    eval_config = SIRConfig(**dataclasses.asdict(config.dataset))
    eval_config.split = "val"
    eval_dataset = SIRDataset(eval_config)

    if config.model.d_vocab is None:
        config.model.d_vocab = dataset.tokenizer.get_vocab_size()

    logger.info("initializing model")
    # set up the model
    model = HookedTransformer(
        HookedTransformerConfig(**dataclasses.asdict(config.model))
    )
    # set up the trainer
    trainer = MaskedLossTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=config.trainer,
    )

    if config.dataset.split == "train":
        # train the model
        logger.info("Training the model")
        trainer.train()

        logger.info("Finished.")
