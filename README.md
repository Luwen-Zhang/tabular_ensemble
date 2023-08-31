# tabular_ensemble
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/LuoXueling/tabular_ensemble/graph/badge.svg?token=APnN7LFtv9)](https://codecov.io/gh/LuoXueling/tabular_ensemble)
[![Test](https://github.com/LuoXueling/tabular_ensemble/actions/workflows/python-package.yml/badge.svg)](https://github.com/LuoXueling/tabular_ensemble/actions/workflows/python-package.yml)
[![](https://img.shields.io/badge/Python-3.8 | 3.9 | 3.10-blue)](https://github.com/LuoXueling/tabular_ensemble)

A framework to evaluate various models for tabular regression and classification tasks. The package integrates the 
following well-established model bases as baselines:

* [`autogluon`](https://github.com/autogluon/autogluon)

* [`pytorch_widedeep`](https://github.com/jrzaurin/pytorch-widedeep)

* [`pytorch_tabular`](https://github.com/manujosephv/pytorch_tabular)

The supported model bases currently have 40+ machine learning (including deep learning) models for tabular prediction 
tasks. You are able to implement your own models, data processing pipelines, and datasets under the flexible and 
well-tested framework for consistent comparisons with baseline models, which is even easier when your own model is 
based on `pytorch`. 

Supported features for all model bases:

* Data processing
  * Data splitting (training/validation/testing sets)
  * Data imputation
  * Data filtering
  * Data scaling
  * Data augmentation
  * Feature augmentation
  * Feature selection
  * etc.
* Multi-modal data
* Loading [UCI datasets](https://archive.ics.uci.edu/datasets)
* Data/result analyzing
  * Leaderboard
  * Box plot
  * Pair plot
  * Pearson correlation
  * Partial dependency plot (with bootstrapping)
  * Feature importance (Permutation and SHAP)
  * etc.
* Building models upon other trained models
* `pytorch_lightning`-based training for `pytorch` models
* Gaussian-process-based Bayesian hyperparameter optimization
* Cross-validation (including continuing from a cross-validation checkpoint)
* Saving, loading, and migrating models

The package stands on the shoulder of giants:

* [scikit-learn](https://scikit-learn.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/)
* etc. (See `requirements.txt`)


## Installation/Usage

See the documentation pages. 

## Contribution

Feel free to create issues if you implement new features, propose a new model, find mistakes, or have any question.

## Citation

If you use this repository, please cite us as:

```text
XXXXXXXXX
```