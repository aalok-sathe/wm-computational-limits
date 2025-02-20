"""
This file houses classes and functions for the Store-Ignore-Recall (SIR) task
"""

# stdlib
import typing
from pathlib import Path
import yaml
import logging
import random

# packages
import tokenizers
import torch
from transformers import PreTrainedTokenizerFast
import numpy as np

# local
from workingmem.task.interface import GeneratedCachedDataset, SupportsGetitem


logger = logging.getLogger(__name__)


# Create a custom tokenizer class by extending PreTrainedTokenizerFast


class SIRTokenizer:
    """
    taken from: https://discuss.huggingface.co/t/creating-a-custom-token-vocabulary-for-gpt-2/134522
    spaces act as a delimiter! spaces are just for human readability, and do not matter for the actual
    tokens.
    """

    template = "{instr} {reg} {item} {samediff} "
    query = "{instr} {reg} {item} "
    instructions = ("store", "ignore")
    labels = ("same", "diff")

    @classmethod
    def from_params(
        cls, n_reg: int, n_items: int, *args, **kwargs
    ) -> tokenizers.Tokenizer:
        # token_to_id
        vocab = {
            "UNK": 0,
            "St": 1,
            "Ig": 2,
            "Re": 3,
            "same": 7,
            "diff": 8,
            **{f"r_{i}": 100 + i for i in range(n_reg)},
            **{f"i_{i}": 100 + i + n_reg for i in range(n_items)},
        }
        # id_to_token = {i: token for token, i in vocab.items()}

        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab, unk_token="UNK"), *args, **kwargs
        )
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        return tokenizer


class SIRDataset(GeneratedCachedDataset):
    """
    dataset instance for the SIR task that inherits methods for caching and loading from
    disk
    """

    data: SupportsGetitem

    def __init__(
        self,
        n_reg: int = 100,
        n_items: int = 100,
        seq_len: int = 100,
        concurrent_reg: int = 2,
        heldout_reg: int = 20,
        heldout_items: int = 20,
        locality: typing.Union[int, None] = 10,
        ignore_prob: float = 0.3,
        #
        n_train: int = 100_000,
        n_val: int = 2_000,
        n_test: int = 2_000,
        #
        split="train",
        basedir="datasets",
    ):
        """
        Class representing an instance of a dataset for the SIR task.
        It encodes many relevant values that manipulate task demands, including,
        the vocabulary size, the number of things held out at the time of training,
        the length of a trial sequence, the number of registers to use concurrently,
        and whether concurrently-used registers have a local structure to them (locality).

        Args:
        ---
            n_reg (int):
                total number of registers in vocab to draw from.  defaults to 100.
            n_items (int):
                total number of items in vocab to draw from.  defaults to 100.
            seq_len (int):
                length of a trial sequence. defaults to 100.
            concurrent_reg (int):
                number of registers to use concurrently within a trial.  default: 2. other instances
                of the experiment will change this parameter upwards to 3, 4, etc.
            heldout_reg (int):
                number (absolute) of registers to hold out. these registers will never make an
                appearance in the train set. defaults to 20.
            heldout_items (int):
                number (absolute) of items to hold out. these items will never appear in the train
                set. defaults to 20.
            locality (typing.Union[int, None]):
                the locality value, when supplied, is used to sample concurrent registers locally
                (numerically close to one another). i.e., register_i can only ever occur in the same
                trial sequence as register_{i \pm locality}.  this allows us to break the locality
                constraint at test time to see out-of-locality-distribution generalization.
                defaults to 10.
            ignore_prob (float):
                probability of an ignore instruction. defaults to 0.3.

            n_train (int):
                number of training trials to generate. defaults to 10,000.
            n_val (int):
                number of validation trials to generate. defaults to 2,000.
            n_test (int):
                number of test trials to generate. defaults to 2,000.

            split (str):
                the split of the dataset to use. if no data already exists on disk,
                data is generated for all splits (we need to make sure all examples
                are unique and non-repeating across splits). if data already exists
                (or once data has been generated), simply supplies examples from
                appropriate split. defaults to "train".
            basedir (str):
                the base directory to store the dataset in.
        """
        # calling super() with kwargs stores these things in self.attrs
        super().__init__(
            # the first set of kwargs makes it into the `self.attrs` dict
            n_reg=n_reg,
            n_items=n_items,
            seq_len=seq_len,
            concurrent_reg=concurrent_reg,
            heldout_reg=heldout_reg,
            heldout_items=heldout_items,
            locality=locality,
            ignore_prob=ignore_prob,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            # the second two are used in the init
            basedir=basedir,
            split=split,
        )

    def __getitem__(self, idx) -> typing.Sequence[str]:
        logger.debug(f"__getitem__ called for index {idx}")
        # since our data supports __getitem__ (for now) we can index into it
        return self.data[idx]

    def generate(self) -> typing.Collection[typing.Sequence[str]]:
        logger.info("generating data for SIR task")

        # seed the random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        return ["St r_0 i_0 diff Ig r_1 i_1 diff St r_1 i_3 diff Ig r_0 i_0 same"] * (
            self.attrs["n_train"] + self.attrs["n_val"] + self.attrs["n_test"]
        )
