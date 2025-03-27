# wm-computational-limits

Are there computational limits on human working memory (WM) capacity aside from anatomical limits?

The directory `workingmem.task.SIR` contains a version of the Store-Ignore-Recall (SIR) task used in human
experiments to tax working memory (CITE). The task involves storing and recalling items stored in
virtual WM 'slots', here, 'registers'. In humans, the task requires active role-addressable 
maintenance of information. In computational models, the retrieval may be facilitated by a number
of strategies.

To generate a dataset, use:
```
python -m workingmem.task.SIR -h
usage: __main__.py [-h] [--n_reg N_REG] [--n_items N_ITEMS] [--seq_len SEQ_LEN] [--concurrent_reg CONCURRENT_REG] [--concurrent_items CONCURRENT_ITEMS]
                   [--heldout_reg HELDOUT_REG] [--heldout_items HELDOUT_ITEMS] [--locality LOCALITY] [--ignore_prob IGNORE_PROB]
                   [--same_diff_prob SAME_DIFF_PROB] [--n_train N_TRAIN] [--n_val N_VAL] [--n_test N_TEST] [--split SPLIT] [--basedir BASEDIR]
                   [--seed SEED] [--generate]

Generate a dataset for the SIR task, or load one if it already exists, and output a few examples

options:
  -h, --help            show this help message and exit
  --n_reg N_REG         Number of registers (vocab). [100]
  --n_items N_ITEMS     Number of items (vocab). [100]
  --seq_len SEQ_LEN     Sequence length (trials in a sequence). [100]
  --concurrent_reg CONCURRENT_REG
                        Number of concurrent registers. [2]
  --concurrent_items CONCURRENT_ITEMS
                        Number of concurrent items. [4]
  --heldout_reg HELDOUT_REG
                        Held-out registers for testing.
  --heldout_items HELDOUT_ITEMS
                        Held-out items for testing.
  --locality LOCALITY   Locality for the tasks.
  --ignore_prob IGNORE_PROB
                        Probability of ignoring an item.
  --same_diff_prob SAME_DIFF_PROB
                        Probability of 'same' outcome [.5]
  --n_train N_TRAIN     Number of training samples. [10,000]
  --n_val N_VAL         Number of validation samples. [2000]
  --n_test N_TEST       Number of test samples. [2000]
  --split SPLIT         Dataset split to use.
  --basedir BASEDIR     Base directory to save or load data.
  --seed SEED           random seed for dataset generation
  --generate            generate the dataset splits and store to disk? if not passed, will simply output a single example generated on-the-fly.
```

Examples look of this sort:
```
St reg_54 item_4 diff St reg_54 item_81 diff St reg_60 item_81 diff St reg_60 item_81 diff St reg_54 item_81 same St reg_54 item_81 same St reg_60 item_40 diff Ig reg_54 item_81 same St reg_54 item_81 same Ig reg_54 item_81 diff St reg_60 item_81 diff St reg_54 item_81 diff St reg_60 item_40 diff St reg_60 item_4 diff Ig reg_54 item_81 same
```

The main module (`-m workingmem`), implemented with an entrypoint in `workingmem/__main__.py` does the orchestrating of loading/constructing datasets, training/evaluating models.
To see the options, run `python -m workingmem -h`.

The module allows you to integrate the main script with [Weights & Biases (`wandb`)](https://wandb.ai)