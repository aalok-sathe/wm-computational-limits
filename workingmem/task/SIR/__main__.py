from workingmem.task.SIR import SIRDataset, SIRTokenizer

import logging

logger = logging.getLogger(__name__)


def main(args):
    # make a call to SIRDataset with the arguments from the command line
    dataset = SIRDataset(**vars(args))
    tokenizer = SIRTokenizer.from_params(args.n_reg, args.n_items)
    print(dataset[0])
    print(tokenizer(dataset[0]))


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
        help="Number of registers.",
    )
    parser.add_argument(
        "--n_items",
        type=int,
        default=100,
        help="Number of items.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length.",
    )
    parser.add_argument(
        "--concurrent_reg",
        type=int,
        default=2,
        help="Number of concurrent registers.",
    )
    parser.add_argument(
        "--heldout_reg",
        type=int,
        default=20,
        help="Held-out registers for testing.",
    )
    parser.add_argument(
        "--heldout_items",
        type=int,
        default=20,
        help="Held-out items for testing.",
    )
    parser.add_argument(
        "--locality",
        type=int,
        default=10,
        help="Locality for the tasks.",
    )
    parser.add_argument(
        "--ignore_prob",
        type=float,
        default=0.3,
        help="Probability of ignoring an item.",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=10000,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=2000,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=2000,
        help="Number of test samples.",
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

    args = parser.parse_args()

    main(args)
