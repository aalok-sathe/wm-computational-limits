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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
logger.setLevel(logging.INFO)


def print_gpu_mem(obj: typing.Any = None):
    """
    Print the GPU memory usage.
    """
    import torch

    if torch.cuda.is_available():
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )
        if obj is not None:
            logger.info(
                f"GPU memory allocated for {obj.__class__.__name__}: "
                f"{torch.cuda.memory_allocated(obj) / 1024**3:.2f} GB"
            )
    else:
        logger.info("No GPU available; no memory report.")


@dataclasses.dataclass
class WandbConfig:
    create_sweep: bool = False
    run_sweep: bool = False
    sweep_id: str = None  # required if do_sweep is True
    project_name: str = "wm-comp-limit-4"
    # method: str = "grid"
    method: str = "bayes"  # grid, random, bayes
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
    filter_by_accuracy: bool = None

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
    supplied_batch_size = config.trainer.batch_size
    config.trainer.batch_size = 256
    logger.warning(
        f"OVERRIDE {supplied_batch_size=}: starting with {config.trainer.batch_size} to search over the memory limit"
    )
    logger.info(f"running main with config: {config}")
    wandb.init(
        project=config.wandb.project_name,
        config=config,
        dir=str(Path("~/scratch/wandb").expanduser().resolve()),
    )

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

    print_gpu_mem(train_dataset)
    print_gpu_mem(eval_dataset)

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

    # once the `from_pretrained` path is set to a not-None value, we can just use the regular way to
    # load the model, since the `ModelWrapper` class will take care of loading the model from checkpoint
    model = ModelWrapper(config.model)

    logger.info(
        f"model initialized with {config.model.n_layers} layers, {config.model.n_heads} heads, "
        f"{config.model.d_model} d_model, {config.model.d_vocab} d_vocab, "
        f"from pretrained: {config.model.from_pretrained}"
    )
    print_gpu_mem(model)

    logger.info(f"about to start training on: {repr(train_dataset)}")
    if config.dataset.split == "train":
        # train the model
        logger.info("Training the model")

        while config.trainer.batch_size >= 16:
            # if the batch size is too large, we won't be able to fit the model in memory
            # so we will reduce it until it fits
            try:
                model.train(
                    train_dataset,
                    config.trainer,
                    eval_dataset=eval_dataset,
                    test_dataset=test_dataset,
                )
                break  # if training succeeded, we can break out of the loop
            except RuntimeError as e:
                if "CUDA out of memory. Tried to allocate" in str(e):
                    logger.info(str(e))
                    logger.warning(
                        f"⚠ batch size {config.trainer.batch_size} is too large, reducing it by half to {config.trainer.batch_size // 2} and retrying"
                    )
                    config.trainer.batch_size //= 2
                    # remember to update the wandb config for logging
                    wandb.config.update(
                        {"trainer.batch_size": config.trainer.batch_size}
                    )
                else:
                    raise e
        else:
            logger.error(
                f"could not train the model with batch size {config.trainer.batch_size} even after reducing it to 16, exiting"
            )
        # model.train(
        #     train_dataset,
        #     config.trainer,
        #     eval_dataset=eval_dataset,
        #     test_dataset=test_dataset,
        # )

        logger.info("Finished.")


if __name__ == "__main__":
    config = tyro.cli(MainConfig)

    # case 1 is we create a new sweep
    if config.wandb.create_sweep:
        sweep_config = dataclasses.asdict(config.wandb)
        sweep_config.update(
            {
                "parameters": {
                    ################################
                    # global experiment parameters
                    ################################
                    # "filter_by_accuracy": {"value": "True"},
                    ################################
                    # model parameters
                    ################################
                    "model.from_pretrained": {
                        # "value": "model_checkpoints/evcxg3kc/"  # n_reg 50 exposure task
                        # "value": "model_checkpoints/b931g4g8"  # split set false
                        # "value": "model_checkpoints/nxgusfzl"  # split set true
                        "value": None
                    },  # !
                    "model.n_layers": {"value": 2},
                    "model.n_heads": {"value": 4},
                    "model.d_model": {"value": 128},  # !
                    "model.d_head": {"value": 128},  # !
                    # "model.seed": {"values": [*range(42, 62)]},
                    "model.seed": {"values": [*map(str, range(42, 62))]},
                    ################################
                    # trainer parameters
                    ################################
                    "trainer.freeze_embeddings": {"value": "False"},  # !
                    # "trainer.freeze_attention": {"value": "False"},  # NOTE: NOT IMPLEMENTED
                    # "trainer.batch_size": {"value": 16}, # this is now automatically determined
                    "trainer.epochs": {"value": 60},  # !
                    # "trainer.learning_rate": { "min": 1e-6, "max": 1e-1, "distribution": "log_uniform_values", },
                    "trainer.learning_rate": {"value": 1e-4},
                    "trainer.weight_decay": {"value": 0.0},
                    # "trainer.weight_decay": {"value": 3e-5},
                    # "trainer.weight_decay": {
                    #     "min": 1e-8,
                    #     "max": 1e-3,
                    #     "distribution": "log_uniform_values",
                    # },
                    "trainer.checkpoint_dir": {"value": "model_checkpoints/"},
                    ################################
                    # dataset parameters
                    ################################
                    "dataset.seq_len": {"value": 300},  # !
                    "dataset.concurrent_reg": {"value": 2},  # !
                    "dataset.n_reg": {"value": 100},  # !
                    "dataset.n_train": {"value": 100_000},  # !
                    # this was changed from 4 to 128 to accommodate split set control for
                    # 64 concurrent registers (doing split set control requires at least 2 items
                    # within each trial sequence that are fixed to a single register)
                    "dataset.concurrent_items": {"values": [128]},
                    "dataset.n_val": {"value": 1_000},
                    "dataset.n_test": {"value": 1_000},
                    "dataset.n_items": {"value": 256},
                    "dataset.global_split_set_control": {"value": "False"},  #!!!
                    # local split set is supposed to be a version of split set control where the split set
                    # is determined on a per-trial-sequence basis rather than at the dataset level.
                    # it's unclear if that should have any effect on the model performance. but incorporating
                    # this condition will help us understand if the 'concurrent management' we are thinking
                    # of actually works the way we think it works (at the trial sequence level).
                    # if local split set works closer to the normal (critical) condition, it doesn't,
                    # "dataset.local_split_set_control": {"value": "False"},  # NOTE: NOT IMPLEMENTED
                    "dataset.heldout_reg": {"value": 0},
                    "dataset.heldout_items": {"value": 0},
                    "dataset.ignore_prob": {"value": 0.5},
                },
            },
        )

        sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
        # dump all the parameters of this sweep to stdout
        logger.info(f"parameters of {sweep_id}:\n{yaml.dump(sweep_config)}")
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
