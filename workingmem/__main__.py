#!/usr/bin/env python3
"""
run as: `python -m workingmem [-h]`
"""

import typing
import dataclasses
import yaml
import logging
import random
from pathlib import Path
from collections import defaultdict
import os

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
from workingmem.utils import print_gpu_mem

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(LOGLEVEL)


@dataclasses.dataclass
class WandbConfig:
    create_sweep: bool = False
    run_sweep: bool = False
    sweep_id: str = None  # required if do_sweep is True
    project_name: str = "wm-comp-limit-7.3.1"
    # method: str = "bayes"  # use this for a hparam sweep
    method: str = "grid"  # use this once hparams are fixed
    metric: dict = dataclasses.field(
        default_factory=lambda: {"goal": "maximize", "name": "eval_acc"}
    )
    program: str = "run_wm.py"  # the program to run with a wandb sweep agent
    from_config: typing.Union[str, None] = None
    """
    `from_config`: only applicable with `create_sweep=True`. reads in a config
    file (YAML) if supplied that enumerates variations over individual variables
    the product of each variable's possible values is used to create a product
    of that many new sweeps, also printed out as a table at the end of running
    this module with this option enabled (both `create_sweep` and `from_config`).
    expects a simple enumaration of values (e.g., `dataset.concurrent_reg: [2,4,8]`)
    rather than `wandb`-specific format (i.e., `dataset.concurrent_reg: {values: [2,4,8]}`)  
    """


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
    array_task_id: typing.Union[int, None] = None
    filter_by_accuracy: bool = None

    def __post_init__(self):
        logger.info(f"running post-init hook to set seeds to {self.seed}")
        if self.seed is not None:
            # prefer to keep using the same dataset instance across model training seeds
            # unless the dataset seed is explicitly set, so we wont be setting
            # `self.dataset.seed` here.
            self.model.seed = self.seed
            self.trainer.seed = self.seed
            # additionally set the seed globally here?
            # NOTE do not set seed for dataset here---we don't want datasets to vary for each instance of
            # a model, because that would introduce too much variability in the model training outcomes

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
    print_gpu_mem(test_dataset)

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

        # set `from_pretrained` path to one of the pretrained models after filtering for its end accuracy.
        # if a seed is provided, we actually use the seed as a modulo rotary operator to pick the Xth index.
        # if no seed is provided, we randomly pick from the list of models.
        if config.model.seed is not None:
            logger.info(
                f"{config.model.seed = }. picking {config.model.seed % len(models_dir)}th model from {len(models_dir)} models"
            )
            config.model.from_pretrained = str(
                models_dir[config.model.seed % len(models_dir)]
            )
        else:
            config.model.from_pretrained = str(random.choice(models_dir))
        # record the new pretrained model path corresponding to the model we're actually using
        wandb.config.update(
            {"model.from_pretrained": str(config.model.from_pretrained)},
            allow_val_change=True,
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

    # NOTE!
    new_epochs = int(config.trainer.epochs / (1 - config.trainer.sparsity))
    # adjust epochs for sparsity
    logger.info(
        f"adjusting epochs for sparsity: {config.trainer.epochs} -> {new_epochs}"
    )
    config.trainer.epochs = new_epochs
    wandb.config.update(
        {"trainer.epochs": config.trainer.epochs}, allow_val_change=True
    )

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
                        {"trainer.batch_size": config.trainer.batch_size},
                        allow_val_change=True,
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

        ############
        # parameters to use when we want to optimize hyperparameters before fixing them for experimentation
        ############
        hparam_optimization_params = {
            "model.n_heads": {"values": [2, 4, 6]},
            "model.d_model": {"values": [64, 128, 256, 512]},
            "model.d_head": {"values": [64, 128, 256, 512]},
            # we use a smaller range of seeds just to make sure out hparams aren't overly seed-specific.
            # TODO: this should actually be set to `None` at optimization-time so the sweep doesn't overfit
            # to a particular subset of seeds (there is unfortunately no way to fully exclude the random seed
            # from sweep parameters)
            "model.seed": {"values": [*map(str, range(62, 67))]},
            "trainer.learning_rate": {
                "min": 1e-6,
                "max": 1e-2,
                "distribution": "log_uniform_values",
            },
        }
        ############
        # parameters to use when we want to run a grid search over a fixed set of hyperparameters
        # NOTE: change these based on the outcomes of the hparam optimization sweep above!
        ############
        fixed_experimental_params = {
            "model.n_heads": {"value": 4},
            "model.d_head": {"value": 256},
            "model.d_model": {"value": 256},
            "model.seed": {
                "values": [*map(str, range(42, 42 + 15))]
            },  # 15 random seeds; non-overlapping range with the seeds used for hparam sweep above
            "trainer.learning_rate": {"value": 5e-4},
        }
        ############

        which_params_to_use = (
            hparam_optimization_params
            if config.wandb.method == "bayes"
            else fixed_experimental_params
        )

        sweep_config.update(
            {
                "parameters": {
                    **which_params_to_use,  # use either hparam optimization or fixed params
                    ################################
                    # generally often-changing params
                    # NOTE don't forget to change 'bayes' to 'grid' following initial hparam sweep
                    ################################
                    # sparsity of feedback (loss) over training
                    "trainer.sparsity": {"value": 0.0},  # !!!!! change!
                    "dataset.concurrent_reg": {"value": 2},
                    "dataset.global_split_set_control": {
                        "value": "False",
                        # "value": "True",
                    },  #!!!
                    "dataset.heldout_items_per_reg": {"value": 15},
                    "dataset.td_prob": {"values": [0, 0.5, 1]},
                    "dataset.n_back": {"value": 5},
                    ################################
                    #                              #
                    #                              #
                    #                              #
                    ################################################################
                    # global experiment parameters that are generally fixed
                    ################################################################
                    #
                    ################################
                    # model parameters
                    ################################
                    "model.from_pretrained": {
                        # "value": "model_checkpoints/evcxg3kc/"  # n_reg 50 exposure task
                        # "value": "model_checkpoints/b931g4g8"  # split set false
                        # "value": "model_checkpoints/nxgusfzl"  # split set true
                        # "value": "model_checkpoints/qc820c8e"  # 100_2 task, 256 item, 128 concurrent
                        "value": None
                    },  # !
                    # "filter_by_accuracy": {"value": "True"}, # only relevant when `from_pretrained` is provided
                    "model.n_layers": {"value": 2},
                    "model.positional_embedding_type": {"value": "rotary"},
                    ################################
                    # trainer parameters
                    ################################
                    "trainer.mask_answer_tokens": {"value": "True"},
                    "trainer.freeze_embeddings": {"value": "False"},
                    # "trainer.freeze_attention": {"value": "False"},  # NOTE: NOT IMPLEMENTED
                    # "trainer.batch_size": {"value": 16}, # now this is automatically determined so we don't parameterize it
                    "trainer.epochs": {"value": 60},
                    "trainer.weight_decay": {"value": 0.0},
                    "trainer.checkpoint_dir": {"value": "model_checkpoints/"},
                    ################################
                    # dataset parameters
                    ################################
                    "dataset.seq_len": {"value": 300},
                    "dataset.n_reg": {"value": 50},
                    "dataset.n_train": {"value": 100_000},
                    # this was changed from 4 to 128 to accommodate split set control for
                    # 64 concurrent registers (doing split set control requires at least 2 items
                    # within each trial sequence that are fixed to a single register). but then, we
                    # observed an increased propensity of networks to engage in heuristic solutions
                    #
                    "dataset.concurrent_items": {"value": 4},
                    "dataset.n_val": {"value": 1_000},
                    "dataset.n_test": {"value": 1_000},
                    "dataset.n_items": {"value": 50},
                    # local split set is supposed to be a version of split set control where the split set
                    # is determined on a per-trial-sequence basis rather than at the dataset level.
                    # it's unclear if that should have any effect on the model performance. but incorporating
                    # this condition will help us understand if the 'concurrent management' we are thinking
                    # of actually works the way we think it works (at the trial sequence level).
                    # if local split set works closer to the normal (critical) condition, it doesn't,
                    # "dataset.local_split_set_control": {"value": "False"},  # NOTE: NOT IMPLEMENTED
                    "dataset.heldout_reg": {"value": 0},
                    "dataset.heldout_items": {"value": 0},
                    "dataset.ignore_prob": {"value": 0.0},
                },
            },
        )

        if config.wandb.from_config is not None:
            # read the YAML file
            with open(config.wandb.from_config, "r") as f:
                from_config_params = yaml.load(f, Loader=yaml.FullLoader)
            # for each of the variables (keys) in this config, we want to do
            # a product of all possible values each variable takes
            sweep_configs = []
            from itertools import product

            keys, values = zip(*from_config_params.items())
            for vals in product(*values):
                this_sweep_config = sweep_config.copy()
                this_sweep_config["parameters"] = this_sweep_config["parameters"].copy()
                print("# ---- -------- new sweep ----")
                for key, val in zip(keys, vals):
                    this_sweep_config["parameters"][key] = {"value": val}

                sweep_id = wandb.sweep(
                    this_sweep_config, project=config.wandb.project_name
                )
                bash_template = f"python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/{config.wandb.project_name}/{sweep_id}"
                # what makes this sweep special?
                sweep_configs.append(
                    "# "
                    + " ".join(
                        f"{k}={v}"
                        for k, v in zip(keys, vals)
                        if k in this_sweep_config["parameters"]
                    )
                    + "\n# "
                    + bash_template
                    + "\n"
                )

            print(*sweep_configs, sep="")

        else:
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
