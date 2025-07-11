import logging
import typing
from abc import ABC, abstractmethod
from hashlib import sha1
from pathlib import Path
import dataclasses

import numpy as np
import tokenizers
import torch
import yaml
import json

logger = logging.getLogger("workingmem")


@dataclasses.dataclass
class GeneratedCachedDatasetConfig:
    """
    configuration for a `GeneratedCachedDataset` instance
    """

    split: typing.Literal["train", "val", "test"] = "train"
    """the split of the dataset to use. if no data already exists on disk,
    data is generated for all splits (we need to make sure all examples
    are unique and non-repeating across splits). if data already exists
    (or once data has been generated), simply supplies examples from
    appropriate split. defaults to "train".
    """
    rootdir: typing.Union[str, Path] = "datasets"
    """where the dataset should be stored and/or read from"""
    seed: typing.Union[int, None] = None
    """by default, no seed is set to enable random generation. upon setting a seed, the dataset instance should be reproducible"""
    generate: bool = True
    """whether to generate the dataset if it doesn't already exist on disk,
       or simply to initialize it to enable calling `generate_trial_sequence`"""
    load: bool = True


class SupportsGetitem(typing.Protocol):
    def __getitem__(self, index): ...


class GeneratedCachedDataset(ABC, torch.utils.data.Dataset):
    """
    abstract class for a dataset that is generated and cached on disk. the dataset
    has three splits: train, test, valid. the dataset is generated by the `generate`
    and cached by the `cache` method.
    a generated dataset is loaded by the `from_path` classmethod which returns an instance of the
    dataset using the config available at the path.
    the dataset object has a `config` which specifies the parameters used to generate the dataset.

    """

    _hash_length = 6
    config: GeneratedCachedDatasetConfig
    label_mask: torch.Tensor = None
    tokenizer: tokenizers.Tokenizer
    config_class: type = GeneratedCachedDatasetConfig

    def __init__(
        self,
        config: GeneratedCachedDatasetConfig,
    ):
        self.config = config
        rootdir = Path(self.config.rootdir).expanduser().resolve()
        self.config.basedir = rootdir / str(self)

        train_path = self.config.basedir / "train.json"
        eval_path = self.config.basedir / "val.json"
        test_path = self.config.basedir / "test.json"
        if train_path.exists() and eval_path.exists() and test_path.exists():
            # list contents of the directory
            logger.info(
                f"data already exists at {self.config.basedir}: \n"
                + "\n\t".join(map(str, self.config.basedir.iterdir()))
            )
            self._load_split()
        else:
            if self.config.basedir.exists():
                # since the directory exists, another process is likely generating the dataset, so
                # we will block execution for 5 minutes for it to finish
                import time

                logger.info(
                    f"waiting for {train_path} to be generated by another process"
                )
                seconds_waiting = 0
                while not train_path.exists():
                    if seconds_waiting % 60 == 0:
                        logger.warning(
                            f"waiting {seconds_waiting / 60} min for {train_path} to be generated"
                        )
                    seconds_waiting += 1
                    if seconds_waiting > 15 * 60:
                        # timeout
                        logger.error(
                            f"timed out waiting {seconds_waiting / 60} min for {train_path} to be generated"
                        )
                    time.sleep(1)
                logger.info(
                    f"found {train_path} after waiting for {seconds_waiting} seconds"
                )
                self._load_split()

            else:
                self.config.basedir.mkdir(parents=True, exist_ok=True)
                if self.config.generate:
                    data = self.generate()
                    self._to_disk(data)

                    if self.config.load:
                        self.data = data
                        self._load_split()  # change this with self.data = data to trivially save some time
                else:
                    logger.info(
                        f"no data found at {self.config.basedir}, and `generate` is set to False. "
                        "please set `generate=True` to generate and save the dataset or make"
                        "calls to `generate_trial_sequence()` for individual trial sequences"
                    )

    def _load_split(self):
        """
        loads the split from disk
        """
        # if self.data is already loaded, we don't need to load it again
        if hasattr(self, "data") and len(self.data) == getattr(
            self.config, f"n_{self.config.split}"
        ):
            return

        split_path = self.config.basedir / f"{self.config.split}.json"

        with split_path.open("r") as f:
            self.data = json.load(f)

        assert len(self.data) == getattr(self.config, f"n_{self.config.split}"), (
            f"Mismatch in # of examples in {self.config.split} on disk at {split_path} ({len(self.data)}) and config value {getattr(self.config, 'n_' + self.config.split)}"
        )

    def _to_disk(self, data: typing.Collection):
        """
        writes the generated data to disk in three splits: train, val, test.
        uses n_{split} to determine the number of examples to write to disk.
        validates that we have enough examples of each split to write to disk.
        """

        attr_str, H = self._attr_str_hash()
        with (
            open(self.config.basedir / "config.yaml", "w") as f,
            open(
                self.config.basedir
                / (
                    H + f"({self.config.n_reg=},{self.config.concurrent_reg=})" + ".txt"
                ),
                "w",
            ) as k,
        ):
            yaml.dump(self._metadata(), f, Dumper=yaml.SafeDumper)
            # yaml.dump(self._metadata(), k, Dumper=yaml.SafeDumper)
            k.write("\n")

        total_examples = sum(
            getattr(self.config, f"n_{split}") for split in ["train", "val", "test"]
        )
        if len(data) != total_examples:
            raise ValueError(
                f"expected {total_examples} examples, got {len(data)} instead"
            )

        cutoffs = [
            getattr(self.config, f"n_{split}") for split in ["train", "val", "test"]
        ]
        cutoffs = np.cumsum(cutoffs)[:-1]
        # logger.info(f"splitting data into {cutoffs} splits")
        splits = np.split(data, cutoffs)

        for split, data in zip(["train", "val", "test"], splits):
            logger.info(
                f"writing {len(data)} examples to {split} at {self.config.basedir}"
            )
            split_path = self.config.basedir / f"{split}.json"
            with split_path.open("w") as f:
                # with gzip.open(split_path, "wt", encoding="utf-8") as f:
                # yaml.dump(data.tolist(), f, Dumper=yaml.SafeDumper, width=float("inf"))
                json.dump(data.tolist(), f)

    def __len__(self):
        assert getattr(self.config, f"n_{self.config.split}") == len(self.data), (
            "Mismatch in dataset length at initialization and number of examples in the dataset on disk"
        )
        return getattr(self.config, f"n_{self.config.split}")

    def _attr_str_hash(self) -> typing.Tuple[str, str]:
        def bool_to_none(v):
            if isinstance(v, bool):
                return v if v else None
            return v

        attr_str = ",".join(
            [
                f"{k}={bool_to_none(v)}"
                for k, v in sorted(self._metadata().items())
                if k
                not in (
                    "basedir",
                    "rootdir",
                    "split",
                    "generate",
                    # "local_split_set_control",
                )  # NOTE! local split set control is here for now.
            ]
        )
        H = sha1(attr_str.encode()).hexdigest()[: self._hash_length].upper()
        return attr_str, H

    def __str__(self) -> str:
        """
        creates a stringified identity of the dataset for storing on disk
        """
        _, H = self._attr_str_hash()
        return f"{self.__class__.__name__}_{H}"

    def __repr__(self):
        """
        creates a stringified identity of the dataset for printing during execution
        """
        attr_str, H = self._attr_str_hash()
        return f"{self.__class__.__name__}_{H}_({attr_str})"

    def _metadata(self) -> typing.Mapping:
        """
        returns a dictionary of metadata for the dataset, which will (ideally)
        be dumped as a YAML file alongside the dataset in the dataset root
        """
        attrs = dataclasses.asdict(self.config)
        # return attrs without the basedir
        return {
            k: v for k, v in attrs.items() if k not in {"basedir", "split", "generate"}
        }

    def __eq__(self, other) -> bool:
        """
        we consider two datasets equivalent (not equal) if they share many of the
        attributes that characterizes the task demand
        """
        return str(self) == str(other)

    def __iter__(self) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
        """
        makes the dataset iterable over the examples in the appropriate split
        """
        for idx in range(getattr(self.config, f"n_{self.config.split}")):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, index):
        # this behavior is both, dataset, and split, dependent
        return NotImplemented

    def generate_trial_sequence(self):
        """
        generates a single trial sequence for the task. this is a single example.
        not a necessary method to implement for a subclass---a subclass may choose
        to do something else in `generate` instead of calling this method.
        """
        NotImplemented

    @abstractmethod
    def generate(self) -> typing.Collection[typing.Sequence[str]]:
        """
        generate a total of n_train + n_val + n_test examples of the dataset such that they are all unique.
        this function makes a guarantee of uniqueness in its output, so the calling stub doesn't have to
        worry about it.
        to simplify things, the caller will treat it as a collection implementing __len__.

        uses random seed (self.seed) if supplied to ensure reproducibility.

        Returns:
        ---
        typing.Collection[typing.Sequence[str]]:
            an iterable over sequences of strings, where each inner sequence is a sequence of tokens (str)
        """
        # every inheriting base class should implement
        return NotImplemented

    @classmethod
    def from_path(
        cls,
        path: typing.Union[str, Path],
        split="train",
        generate=False,
    ) -> "GeneratedCachedDataset":
        """
        creates a dataset instance from a path to a stored dataset

        Args:
        ---
        path (typing.Union[str, Path]):
            1. path to the stored dataset
            2. string path to the stored dataset
            3. hash string identifier of the dataset (e.g. XY7ZY)

        basedir (str):
            the base directory to store the dataset in (default: 'datasets/')

        split (str):
            the split of the dataset to use. if no data already exists on disk,
            data is generated for all splits (we need to make sure all examples
            are unique and non-repeating across splits). if data already exists
            (or once data has been generated), simply supplies examples from
            appropriate split. defaults to "train".
        """
        if isinstance(path, str):
            dest = Path(path).expanduser().resolve()
            if not dest.exists():
                raise FileNotFoundError(f"{dest} does not exist")
        with (dest / "config.yaml").open("r") as f:
            config = cls.config_class(**yaml.load(f, Loader=yaml.SafeLoader))
            # config.rootdir = dest.parent # TODO---rootdir should not have been included in the hash!
            config.split = split
            config.generate = generate
        instance = cls(config)

        return instance
