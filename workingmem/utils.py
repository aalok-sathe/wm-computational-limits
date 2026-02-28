from itertools import product
import typing
from functools import lru_cache
import logging

import pandas as pd
from tqdm.auto import tqdm
import wandb

wandbapi = wandb.Api()


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
    runs = wandbapi.sweep(f"{user}/{project_name}/{sweep_id}").runs
    dfs = []
    for run in tqdm([*runs]):
        metrics = run.history(pandas=True, samples=10_000)
        metrics["run_id"] = run.name
        metrics["sweep_id"] = sweep_id
        try:
            dfs += [metrics.groupby("epoch").first().reset_index()]
        except KeyError:
            # this run doesn't have enough data to have 'epoch' as a key; skip for now
            print(f"\tskipping run: {sweep_id}/{run.name}")
            pass

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def _flatten_collection_of_tuples(
    keys_tuples_collection: typing.Collection[tuple],
    vals_tuples_collection: typing.Collection[tuple],
):
    keys_flat, vals_flat = [], []
    for keys_tuple, vals_tuple in zip(keys_tuples_collection, vals_tuples_collection):
        # deflate the tuples
        keys_flat += [*keys_tuple]
        vals_flat += [*vals_tuple]
    return keys_flat, vals_flat


def parse_config(config) -> typing.Generator[dict, None, None]:
    if "independent_variables" in config:
        independent_variables: typing.List[typing.Dict] = config[
            "independent_variables"
        ]

        conditional_variables: typing.List[dict] = config.get(
            "conditional_variables",
            [{"index": {}, "kwargs": {}}],  # default is no values to look up
        )

        def _lookup_kwargs(parameters):
            # we iterate through conditoinal variable entries in order
            # and check if
            kwargs = {}
            for cond_variable_set in conditional_variables:
                index = cond_variable_set["index"]
                if all(parameters[k] == v for k, v in index.items()):
                    this_kwargs = cond_variable_set["kwargs"]
                    kwargs.update(this_kwargs)
                    break
                continue
            return kwargs

        # we maintain tuples of keys and tuples of values
        # to enable grouping them together. after taking their product
        # we will uncouple them
        ind_keys_tuples, ind_vals_tuples = [], []

        for d in independent_variables:
            ind_keys_tuples += [d.keys()]
            ind_vals_tuples += [tuple(zip(*d.values()))]

        print(ind_keys_tuples, ind_vals_tuples)

        assert len(ind_keys_tuples) == len(ind_vals_tuples)

        values_product = [*product(*ind_vals_tuples)]

        # we want to create a sweep corresponding to each 'value set' at the end of the
        # cross product between all possible values of covarying independent variable sets
        for this_values_set in values_product:
            keys, vals = _flatten_collection_of_tuples(ind_keys_tuples, this_values_set)
            parameters = dict(zip(keys, vals))
            print(parameters)
            parameters.update(_lookup_kwargs(parameters))
            yield parameters

    else:  # this means the config file just contains (key: values) entries, old-style format
        logger.warning(
            "config supplied is old-style formatted; parsing assuming flat key-value structure."
        )
        keys, values = zip(*config.items())
        for this_values_set in product(*values):
            yield dict(zip(keys, this_values_set))
