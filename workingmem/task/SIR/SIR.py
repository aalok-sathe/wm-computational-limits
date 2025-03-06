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
import torch
from tqdm.auto import tqdm

# local
from workingmem.task.interface import (
    GeneratedCachedDataset,
    GeneratedCachedDatasetConfig,
    SupportsGetitem,
)


logger = logging.getLogger(__name__)


@dataclass
class SIRConfig(GeneratedCachedDatasetConfig):
    n_reg: int = 100
    """total number of registers in vocab to draw from"""
    n_items: int = 100
    """total number of items in vocab to draw from"""
    seq_len: int = 100
    """ length of a trial sequence"""
    concurrent_reg: int = 2
    """number of registers to use concurrently within a trial. if this
    number is too high, we risk a simple heuristic solution such as: 
    simply check if an item has appeared in the prior history, when 
    number of total items n_items is high"""
    concurrent_items: int = 5
    """number of items to use concurrently within a trial"""
    heldout_reg: int = 20
    """number (absolute) of registers to hold out. these registers will never make an
    appearance in the train set"""
    heldout_items: int = 20
    """number (absolute) of items to hold out. these items will never appear in the train"""
    locality: typing.Union[int, None] = None
    """the locality value, when supplied, is used to sample concurrent registers locally
        (numerically close to one another). i.e., register_i can only ever occur in the same
        trial sequence as register_{i \pm locality}.  this allows us to break the locality
        constraint at test time to see out-of-locality-distribution generalization.
        TODO: option to manipulate locality of train/test split. alternatively, we could
        do this evaluation using a separate dataset with the locality parameter relaxed
        (which should make the test data OOD)"""
    ignore_prob: float = 0.3
    """probability of an ignore instruction"""
    same_diff_prob: float = 0.5
    """probability of a 'same' outcome on a particular register. varies independently of
        store/ignore instruction"""

    # seed: int = None
    n_train: int = 1_000
    n_val: int = 500
    n_test: int = 500


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
        """
        this class could be used to create a loss mask, if the structure of each
        individual trial were variable. else, we could just use the positions to
        create the mask in the case of the SIR task.
        """

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
            "PAD": 1,
            cls.instructions.store: 3,
            cls.instructions.ignore: 4,
            cls.instructions.recall: 5,
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
        tokenizer.enable_padding(
            direction="left",
            length=None,
            pad_id=1,
            pad_token="PAD",
        )

        return tokenizer


class SIRDataset(GeneratedCachedDataset):
    """
    dataset instance for the SIR task that inherits methods for caching and loading from
    disk
    """

    data: SupportsGetitem
    trial_label_mask: tuple = (0, 0, 0, 1)
    # use_bos_token: bool = True # alternatively, we just suck it up and handle the offset on the model side

    def __init__(
        self,
        config: SIRConfig,
        tokenizer: tokenizers.Tokenizer = None,
    ):
        """
        Class representing an instance of a dataset for the SIR task.
        It encodes many relevant values that manipulate task demands, including,
        the vocabulary size, the number of things held out at the time of training,
        the length of a trial sequence, the number of registers to use concurrently,
        and whether concurrently-used registers have a local structure to them (locality).
        """
        super().__init__(config)
        # seed the random number generator
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        self.tokenizer = tokenizer or SIRTokenizer.from_params(
            self.config.n_reg, self.config.n_items
        )

    def __getitem__(
        self, idx: int
    ) -> typing.Dict[str, typing.Union[torch.Tensor, tokenizers.Encoding]]:
        logger.debug(f"__getitem__ called for index {idx}")
        # since our data supports __getitem__ (for now) we can index into it
        sequence = torch.LongTensor(self.data[idx]["sequence"])
        answer_locations = torch.LongTensor(self.data[idx]["answer_locations"])
        encoding = self.tokenizer.encode(sequence)
        return {
            "tokens": sequence,
            "answer_locations": answer_locations,
            "encoding": encoding,
        }

    def generate_trial_sequence(
        self,
    ) -> dict:
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
          1. pick a start_idx uniformly from the range [0, n_reg) [oops we forgot about the heldout regs]
          2. choose concurrent_reg registers from the range [start_idx, start_idx + (locality or n_reg)) % n_reg
             and likewise choose concurrent_items items from the range [0, n_items)
        +-------- (repeat for seq_len steps) ----------------------------+
        | 3. pick one register to operate on from the chosen registers   |
        | 4. pick an instruction using ignore_prob                       |
        | 5. pick an item using same_diff_prob, unless there was no      |
        |     previous item (note that 4 & 5 are independent)            |
        |     [oops, we forgot about the heldout items]                  |
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
        start_idx = np.random.randint(0, self.config.n_reg)

        # step 2
        # pick registers from the range [start_idx, start_idx + (locality or n_reg)) % n_reg
        if self.config.locality is not None:
            assert self.config.locality >= self.config.concurrent_reg, (
                f"locality must be at least the number of concurrent registers to use. you supplied: {self.config.locality} < {self.config.concurrent_reg}"
            )
        reg_range = np.arange(
            start_idx,
            start_idx
            + (
                self.config.locality
                # WLG, heldout registers are numerically at the end of the reg pool
                or (self.config.n_reg - self.config.heldout_reg)
            ),
            dtype=int,
            # this way, we wraparound at the end of the available (non-held-out) register pool
        ) % (self.config.n_reg - self.config.heldout_reg)

        # sample w/o replacement
        # regs_chosen now contains the indexes of the registers to use
        regs_chosen: np.ndarray[int] = np.random.choice(
            reg_range, self.config.concurrent_reg, replace=False
        )
        items_chosen: np.ndarray[int] = np.random.choice(
            # WLG, heldout items are numerically at the end of the item pool, and
            # so far we aren't doing anything like locality or wraparound with them
            # so this hunk is simpler
            np.arange(self.config.n_items - self.config.heldout_items),
            self.config.concurrent_items,
            replace=False,
        )

        # here is where we start maintaining the state of the registers (as in, the item they currently hold)
        reg_state = {i: -1 for i in regs_chosen}

        this_trial_seq = []
        # repeat seq_len times:
        for i in range(self.config.seq_len):
            # step 3
            # pick one register to operate on from the chosen registers
            # NOTE: in the future, to manipulate delayed recall from a certain register,
            # we can use the `p=...` argument to np.random.choice to bias the selection
            # away from a certain register
            this_reg_idx = np.random.choice(regs_chosen, p=None)

            # step 4
            # pick an instruction using ignore_prob
            this_instr = ignore if np.random.rand() < self.config.ignore_prob else store

            # step 5
            # pick an item using same_diff_prob, unless there was no previous item, in which case,
            # we must pick a new item and make the instruction be 'diff' by default
            if (
                reg_state[this_reg_idx] == -1
                or np.random.rand() > self.config.same_diff_prob
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

        return {
            "sequence": " ".join(this_trial_seq),
            "regs_used": tuple(regs_chosen.tolist()),
            "items_used": tuple(items_chosen.tolist()),
            "locality": self.config.locality,
        }

    @property
    def answer_locations(self) -> typing.List[int]:
        """
        returns a mask for the locations of the "answers" in the sequence where loss should be computed (the only deterministic/structured)
        part of the SIR task.
        """
        return list(SIRDataset.trial_label_mask * self.config.seq_len)

    def generate(self) -> typing.Collection[typing.Sequence[str]]:
        """
        makes repeated calls to `_generate_trial_sequence` to generate a total of
        n_train + n_val + n_test examples.
        """
        logger.info("generating data for SIR task")

        # seed the random number generator
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        examples = set()
        examples_list = []

        total = self.config.n_train + self.config.n_val + self.config.n_test
        for _ in tqdm(
            range(total),
            desc="generating SIR trials",
            total=total,
        ):
            # check for duplicate trials
            while True:
                trial = self.generate_trial_sequence()
                fstrial = frozenset(sorted(trial.items()))
                if fstrial not in examples:
                    examples.add(fstrial)
                    examples_list.append(trial)
                    break
            # yield _generate_trial()
        return list(examples_list)
