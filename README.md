# `simWM` 🏊: simulation of working memory management with neural networks

- [Introduction](#introduction)
- [Installation](#getting-started--install)

## Introduction
The `simWM` computational modeling framework serves to enable computational
simulations of working memory tasks. 
A robustly-engineered framework allows researchers to define their own tasks and
generate datasets with fine control over the parameters of those tasks. The
framework allows for training neural models in an architecture-agnostic manner,
so long as they support a PyTorch backend. This includes transformers, recurrent
neural networks, and long short-term memory networks that can flexibly act as
_participants_ in the same tasks, allowing for clean, controlled comparisons
across model instantiations. Users of this framework can add their own models
to work with the rest of the framework as long as they provide a wrapper
that conforms with the 
[model wrapper interface](https://aalok-sathe.github.io/working-memory/workingmem/model.html#ModelWrapper).
Similarly, users can use the default supported Reference-Back task from
Rac-Lubashevsky & Kessler, 2016, implemented here, or provide their own implementation
that conforms to the [dataset interface](https://aalok-sathe.github.io/working-memory/workingmem/task/interface.html#GeneratedCachedDataset).

Integration with the [weights & biases (`wandb`)](https://wandb.ai)
experiment-tracking platform promotes robust, open, and replicable science by
constructing uniquely-tagged and documented config-driven experiments for each
condition a researcher may be interested in. These experimental conditions live
in separate spaces on the disk, each with meticulously documented metadata that
makes discerning results and analyzing data pleasantly organized. Furthermore,
the software supports exposure and transfer-learning experiments using these
same condition-based tags, allowing precise documentation of the entire training
history of a computational model and subsequent experiments on pretrained
models.

The main module (`python -m workingmem`), implemented as an entrypoint via
`workingmem/__main__.py`, does the orchestrating of running experiments, i.e., 
loading/constructing datasets, training/evaluating models. However, much of the
library's functionality exposed for programmatic use as well, and allows
researchers to construct datasets in a custom manner, manage their own
training-eval routines, and handle data management.

**A typical experiment workflow looks like:**
1. identify manipulations of interest (see what variations the library already supports using `python -m workingmem -h`).
2. write/modify a config defining experimental conditions ([example](./configs/sample_conditional_config))
    - configs follow an "independent variables" and "conditional variables" format---independent variables are enumerated as lists
      that yield a cross-product over all possible combinations of their variation. "conditional variables" are typically hyperparameters
      that need to be looked up dependent on the particular condition.  
3. use `python -m workingmem --wandb.create_sweep` along with the flag `--wandb.from_config [path/to/config]` to define individual experimental conditions.
  at this point, the library evaluates a cross-product over all possible conditions in your experiment and creates individual
  W&B "sweeps" for each condition. this enables separate tracking of the progress of experiments in a web browser, as well as unique-ID-based
  retrieval after the experiment finishes for clean, reproducible science.

For programmatic use, components of the library can be imported in your program: `import workingmem`, or `from workingmem import LSTMModelWrapper, SIRDataset`.

To exhaustively see the CLI options, run `python -m workingmem -h`.

## Getting started / Install
1. Use with Weights and Biases (recommended)

   `simWM` is best used alongside Weights and Biases. In order to do so, you will have to create an account on the [W&B website](https://wandb.ai).
    There are many ways in which to do so, including using your GitHub login.
   
1. Install `uv` (recommended)

    `simWM` uses [`uv`](https://astral.sh/uv) as its package- and environement-manager. `uv` makes painless the age-old task of managing dependencies
    in Python. In order to install the framework, you'll need to install `uv` on your system. This is fairly straightforward---visit the link from before.

1. Install `sWiMm` 
    - using `uv`:
      - `uv sync`: install the python virtual environment with all requisite packages (needed once)
      - `. ./.venv/bin/activate`: activate the virtual environment in the directory of the library (needed each time you log in to your compute node until you exit/log out)
    - using `pip`:
      - pip install .


## References
- Rac-Lubashevsky, R., & Kessler, Y. (2016). Dissociating working memory updating and automatic updating: The reference-back paradigm. Journal of Experimental Psychology: Learning, Memory, and Cognition, 42(6), 951–969. https://doi.org/10.1037/xlm0000219

# Tutorials
