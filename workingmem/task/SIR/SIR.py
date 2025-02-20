"""
This file houses classes and functions for the Store-Ignore-Recall (SIR) task
"""

# stdlib
from dataclasses import dataclass
import typing
import logging
import random

# packages
import tokenizers
import numpy as np
from tqdm.auto import tqdm

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

    @dataclass
    class instructions:
        store = "St"
        ignore = "Ig"
        recall = "Re"

    @dataclass
    class labels:
        same = "same"
        diff = "diff"

    @dataclass
    class symbols:
        register: typing.Callable = lambda i: f"reg_{i}"
        item: typing.Callable = lambda i: f"item_{i}"

    @classmethod
    def from_params(
        cls,
        n_reg: int,
        n_items: int,
        *args,
        **kwargs,
    ) -> tokenizers.Tokenizer:
        # token_to_id

        # dataclasses that have default instantiations represent text-form of symbols such
        # as instructions, labels, register identifiers and item identifiers

        # we define the vocabulary here with maximally identifiable token_ids:
        # 0-10 represents special symbols
        # 100-n_reg represents registers
        # 300(ish)-n_items represents items. typically, n_reg<=300 so n_items can start at 300
        #   (see highlighted line below)
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        vocab = {
            "UNK": 0,  # it is problematic if this is ever used
            cls.instructions.store: 1,
            cls.instructions.ignore: 2,
            cls.instructions.recall: 3,
            cls.labels.same: 7,
            cls.labels.diff: 8,
            **{cls.symbols.register(i): 100 + i for i in range(n_reg)},
            **{
                cls.symbols.item(i): max(300, 100 + n_reg) + i
                #                    ^^^^^^^^^^^^^^^^^^^^^
                for i in range(n_items)
            },
        }
        # id_to_token = {i: token for token, i in vocab.items()}

        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab, unk_token="UNK"), *args, **kwargs
        )
        # we want to tokenize on whitespace so that whitespace is ignored fully
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
        concurrent_items: int = 4,
        heldout_reg: int = 20,
        heldout_items: int = 20,
        locality: typing.Union[int, None] = 10,
        ignore_prob: float = 0.3,
        same_diff_prob: float = 0.5,
        #
        seed=42,
        #
        n_train: int = 100_000,
        n_val: int = 2_000,
        n_test: int = 2_000,
        #
        split="train",
        basedir="datasets",
        generate=True,
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
            concurrent_items (int):
                number of items to use concurrently within a trial. defaults to 4.
                if this number is too high, we risk a simple heuristic solution: simply check if
                an item has appeared in the prior history, when number of total items n_items is high.
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
            same_diff_prob (float):
                probability of a 'same' outcome on a particular register. varies independently of
                store/ignore instruction. defaults to 0.4.

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
        # seed the random number generator
        np.random.seed(seed)
        random.seed(seed)

        self.tokenizer = SIRTokenizer.from_params(n_reg, n_items)

        # calling super() with kwargs stores these things in self.attrs
        super().__init__(
            # the first set of kwargs makes it into the `self.attrs` dict
            # and largely determines the nature of the dataset
            n_reg=n_reg,
            n_items=n_items,
            seq_len=seq_len,
            concurrent_reg=concurrent_reg,
            concurrent_items=concurrent_items,
            heldout_reg=heldout_reg,
            heldout_items=heldout_items,
            locality=locality,
            ignore_prob=ignore_prob,
            same_diff_prob=same_diff_prob,
            # random seed
            seed=seed,
            # specify sizes of splits
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            # the next set of kwargs are used in the init but not recorded in attrs
            basedir=basedir,
            split=split,
            generate=generate,
        )

    def __getitem__(self, idx) -> typing.Sequence[str]:
        logger.debug(f"__getitem__ called for index {idx}")
        # since our data supports __getitem__ (for now) we can index into it
        return self.data[idx]

    def generate_trial_sequence(
        self,
    ) -> str:
        """
        Generates a sequence of `seq_len` trials for the SIR task

        key things to remember:
        1. we have n_reg registers and n_items items in total, but not all of them are
           used in every trial. only concurrent_reg are used. and same_diff_prob determines
           how items are drawn: if a 'same' outcome is picked, the same item is used for that
           register in the next trial. if a 'diff' outcome is picked, a new item is uniformly
           drawn from the item pool without replacement.
        2. ignore_prob determines the likelihood of an ignore instruction. if an ignore instruction
           is given, the register is not updated.
        3. locality determines the locality of the registers. if locality is None, registers are
           drawn uniformly from the register pool. if locality is an integer, registers are drawn
           from a local pool of registers that are within locality distance of the current register.
           in practice, this means, picking a start_idx uniformly from the range [0, n_reg) with
           wraparound, and picking registers uniformly without replacement from this range

        algorithm
          1. pick a start_idx uniformly from the range [0, n_reg)
          2. choose concurrent_reg registers from the range [start_idx, start_idx + (locality or n_reg)) % n_reg
             and likewise choose concurrent_items items from the range [0, n_items)
        +-------- (repeat for seq_len steps) ----------------------------+
        | 3. pick one register to operate on from the chosen registers   |
        | 4. pick an instruction using ignore_prob                       |
        | 5. pick an item using same_diff_prob, unless there was no      |
        |     previous item (note that 4 & 5 are independent)            |
        | 6. update the register with the item if the instruction is     |
        |     not ignore                                                 |
        +----------------------------------------------------------------+
        """
        store = SIRTokenizer.instructions.store
        ignore = SIRTokenizer.instructions.ignore
        # recall = self.tokenizer.instructions.recall
        same = SIRTokenizer.labels.same
        diff = SIRTokenizer.labels.diff
        reg = SIRTokenizer.symbols.register
        item = SIRTokenizer.symbols.item

        # step 1
        # pick a start_idx uniformly from the range [0, n_reg)
        start_idx = np.random.randint(0, self.attrs["n_reg"])

        # step 2
        # pick registers from the range [start_idx, start_idx + (locality or n_reg)) % n_reg
        assert self.attrs["locality"] >= self.attrs["concurrent_reg"], (
            f"locality must be at least the number of concurrent registers to use. you supplied: {self.attrs['locality']} < {self.attrs['concurrent_reg']}"
        )
        reg_range = (
            np.arange(
                start_idx,
                start_idx + (self.attrs["locality"] or self.attrs["n_reg"]),
                dtype=int,
            )
            % self.attrs["n_reg"]
        )

        # sample w/o replacement
        # regs_chosen now contains the indexes of the registers to use
        regs_chosen: typing.Collection[int] = np.random.choice(
            reg_range, self.attrs["concurrent_reg"], replace=False
        )
        items_chosen: typing.Collection[int] = np.random.choice(
            np.arange(self.attrs["n_items"]),
            self.attrs["concurrent_items"],
            replace=False,
        )

        # here is where we start maintaining the state of the registers (as in, the item they currently hold)
        reg_state = {i: -1 for i in regs_chosen}

        this_trial_seq = []
        # repeat seq_len times:
        for i in range(self.attrs["seq_len"]):
            # step 3
            # pick one register to operate on from the chosen registers
            # NOTE: in the future, to manipulate delayed recall from a certain register,
            # we can use the `p=...` argument to np.random.choice to bias the selection
            # away from a certain register
            this_reg_idx = np.random.choice(regs_chosen, p=None)

            # step 4
            # pick an instruction using ignore_prob
            this_instr = (
                ignore if np.random.rand() < self.attrs["ignore_prob"] else store
            )

            # step 5
            # pick an item using same_diff_prob, unless there was no previous item, in which case,
            # we must pick a new item and make the instruction be 'diff' by default
            if (
                reg_state[this_reg_idx] == -1
                or np.random.rand() > self.attrs["same_diff_prob"]
            ):
                this_item = np.random.choice(items_chosen, p=None)
                this_label = diff
            else:
                this_item = reg_state[this_reg_idx]
                this_label = same

            # right here is where we assemble the current trial
            this_trial = [
                this_instr,
                reg(this_reg_idx),
                item(this_item),
                this_label,
            ]
            this_trial_seq.extend(this_trial)

            # step 6
            # update the register with the new item if the instruction is not ignore
            if this_instr != ignore:
                # doesn't matter if it's the same or a new item
                reg_state[this_reg_idx] = this_item

        return " ".join(this_trial_seq)

    def generate(self) -> typing.Collection[typing.Sequence[str]]:
        """
        makes repeated calls to `_generate_trial_sequence` to generate a total of
        n_train + n_val + n_test examples.
        """
        logger.info("generating data for SIR task")

        # seed the random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        examples = set()

        for _ in tqdm(
            range(
                (
                    total := self.attrs["n_train"]
                    + self.attrs["n_val"]
                    + self.attrs["n_test"]
                )
            ),
            desc="generating SIR trials",
            total=total,
        ):
            # check for duplicate trials
            while True:
                trial = self.generate_trial_sequence()
                if trial not in examples:
                    examples.add(trial)
                    break
            # yield _generate_trial()
        return list(examples)
