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


def main(config: MainConfig): ...


if __name__ == "__main__":
    config = tyro.cli(MainConfig)
    logger.info(f"{config}")

    if config.wandb.use_wandb:
        # set up the sweep
        sweep_config = {}
        wandb.init(project=config.project_name, config=config)

    # set up the model
    model = HookedTransformer(HookedTransformerConfig(**config.model))
    # set up the dataset
    dataset = SIRDataset(config.dataset)
    # set up the trainer
    trainer = MaskedLossTrainer(
        model=model.model,
        dataset=dataset,
        args=TrainingArguments(
            output_dir=".",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="steps",
            eval_steps=1,
            save_total_limit=0,
        ),
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
    )
    # train the model
    trainer.train()
