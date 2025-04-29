# in this script, we will fetch wandb runs based on sweep_id
# we will use the wandb API to fetch the runs
# %%
import wandb
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path


def collection2str(collection, sep: str = ","):
    """return deterministic string representation of a collection"""
    return sep.join(sorted(collection))


def singleton(iterable):
    """
    Returns the single element of an iterable, or raises an error if the iterable is empty or has more than one element.
    """
    it = iter(iterable)
    first = next(it)
    try:
        next(it)
        raise ValueError("iterable has more than one item")
    except StopIteration:
        if isinstance(first, list):
            return first[0]
        return first


# %%
def get_run_history(
    sweep_id: str,
    project_name: str,
    keys=["eval_acc", "epoch", "step"],
    basedir="wandb_api_data/",
    filename_params=[
        "dataset.n_reg",
        "dataset.concurrent_reg",
        "dataset.seq_len",
        "model.d_model",
    ],
):
    """
    Get the eval accuracy curve for a given sweep_id and project_name
    """

    # Initialize wandb API & fetch the sweep
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")

    # get sweep params of interest from sweep config
    sweep_config = sweep.config["parameters"]
    del sweep_config["trainer.checkpoint_dir"]
    print(sweep_config)
    filename_params = [
        f"{param.split('.')[-1]}={singleton(sweep_config[param].values())}"
        for param in filename_params
    ]
    print(f"{filename_params = }")

    # check to see if we've already retrieved the data for this sweep_id
    filename = Path(
        f"{basedir}/{project_name}/{sweep_id},{'eval_acc' if 'eval_acc' in keys else 'train_loss'},{collection2str(filename_params)}.csv"
    )
    if filename.exists():
        print(f"file {filename} already exists. Loading from file.")
        df = pd.read_csv(filename)
        return df

    records = []
    for run in tqdm(sweep.runs):
        print(run.name)
        # eval_acc = run.summary.get("eval_accuracy")
        # get entire eval_acc curve by training step
        history = run.scan_history(
            # samples=1_000,
            keys=keys,
            # x_axis="step",
            # pandas=True,
        )

        while True:
            try:
                records += [next(history)]
            except StopIteration:
                break
        break

    records = pd.DataFrame(records)
    # add sweep params to the records
    for param, value in sweep_config.items():
        records[param] = pd.Series(
            [singleton(value.values())] * len(records), index=records.index
        )
    # save the records to a csv file
    filename.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(filename, index=False)
    print(f"Saved records to {filename}")
    return records


if __name__ == "__main__":
    # %%
    import tyro

    tyro.cli(get_run_history)
    # ("y81xhmif", "wm-comp-limit-1")

    # %%
    # print(df.columns)
    # df
