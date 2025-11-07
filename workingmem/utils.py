import typing
from functools import lru_cache
import logging

import pandas as pd
from tqdm.auto import tqdm
import wandb

api = wandb.Api()


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


@lru_cache(maxsize=None)
def get_wandb_runs(
    project_name="wm-comp-limit-7.3.1", sweep_id="4rrckcd1", user="aloxatel"
):
    runs = api.sweep(f"{user}/{project_name}/{sweep_id}").runs
    dfs = []
    for run in tqdm([*runs]):
        metrics = run.history(pandas=True, samples=10_000)
        metrics["run_id"] = run.name
        metrics["sweep_id"] = sweep_id
        dfs += [metrics.groupby("epoch").first().reset_index()]

    df = pd.concat(dfs).reset_index(drop=True)
    return df
