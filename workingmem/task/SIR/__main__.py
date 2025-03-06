import logging
import dataclasses

import tyro

from workingmem.task.SIR import SIRDataset, SIRTokenizer, SIRConfig

logger = logging.getLogger(__name__)


def main(config: SIRConfig):
    # make a call to SIRDataset with the arguments from the command line
    dataset = SIRDataset(config)
    tokenizer = dataset.tokenizer

    trial_seq = dataset.generate_trial_sequence()
    print(trial_seq)
    print(tokenizer.encode(trial_seq["sequence"]))


if __name__ == "__main__":
    config = tyro.cli(SIRConfig)
    main(config)
