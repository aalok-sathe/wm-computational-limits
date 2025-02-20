from workingmem.task.SIR import SIRDataset, SIRTokenizer

import logging

logger = logging.getLogger(__name__)


def main(args):
    # make a call to SIRDataset with the arguments from the command line
    dataset = SIRDataset(**vars(args))
    tokenizer = SIRTokenizer.from_params(args.n_reg, args.n_items)
    trial_seq = dataset.generate_trial_sequence()
    print(trial_seq)
    print(tokenizer.encode(trial_seq["sequence"]).ids)


if __name__ == "__main__":
    import argparse

    # here, we create an argparser with all the arguments that SIRDataset requires to generate data
    parser = argparse.ArgumentParser(
        description="Generate a dataset for the SIR task, or load one if it already exists, and output a few examples"
    )

    parser.add_argument(
        "--n_reg",
        type=int,
        default=100,
        help="Number of registers (vocab). [100]",
    )
    parser.add_argument(
        "--n_items",
        type=int,
        default=100,
        help="Number of items (vocab). [100]",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length (trials in a sequence). [100]",
    )
    parser.add_argument(
        "--concurrent_reg",
        type=int,
        default=2,
        help="Number of concurrent registers. [2]",
    )
    parser.add_argument(
        "--concurrent_items",
        type=int,
        default=5,
        help="Number of concurrent items. [4]",
    )
    parser.add_argument(
        "--heldout_reg",
        type=int,
        default=20,
        help="Held-out registers for testing. [20]",
    )
    parser.add_argument(
        "--heldout_items",
        type=int,
        default=20,
        help="Held-out items for testing. [20]",
    )
    parser.add_argument(
        "--locality",
        type=int,
        default=10,
        help="Locality of registers to draw from (concurrent regs always within +=L of each other). [10]",
    )
    parser.add_argument(
        "--ignore_prob",
        type=float,
        default=0.3,
        help="Probability of ignoring an item.",
    )
    parser.add_argument(
        "--same_diff_prob",
        type=float,
        default=0.5,
        help="Probability of 'same' outcome [.5]",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=10_000,
        help="Number of training samples. [10,000]",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=2000,
        help="Number of validation samples. [2000]",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=2000,
        help="Number of test samples. [2000]",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        default="datasets",
        help="Base directory to save or load data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for dataset generation",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate the dataset splits and store to disk? if not passed, "
        "will simply output a single example generated on-the-fly.",
    )

    args = parser.parse_args()

    main(args)
