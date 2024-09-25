# tabular_ensemble
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/LuoXueling/tabular_ensemble/graph/badge.svg?token=APnN7LFtv9)](https://codecov.io/gh/LuoXueling/tabular_ensemble)
[![Test](https://github.com/LuoXueling/tabular_ensemble/actions/workflows/python-package.yml/badge.svg)](https://github.com/LuoXueling/tabular_ensemble/actions/workflows/python-package.yml)
[![](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://github.com/LuoXueling/tabular_ensemble)
[![Documentation Status](https://readthedocs.org/projects/tabular-ensemble/badge/?version=latest)](https://tabular-ensemble.readthedocs.io/en/latest/?badge=latest)

A framework to evaluate various models for tabular regression and classification tasks. The package integrates 25 machine learning (including deep learning) models for tabular prediction 
tasks from the following well-established model bases:

* [`autogluon`](https://github.com/autogluon/autogluon)
  * `"LightGBM"`, `"CatBoost"`, `"XGBoost"`, `"Random Forest"`, `"Extremely Randomized Trees"`, `"K-Nearest Neighbors"`, `"Linear Regression"`, `"Neural Network with MXNet"`, `"Neural Network with PyTorch"`, `"Neural Network with FastAI"`.
* [`pytorch_widedeep`](https://github.com/jrzaurin/pytorch-widedeep)
  * `"TabMlp"`, `"TabResnet"`, `"TabTransformer"`, `"TabNet"`, `"SAINT"`, `"ContextAttentionMLP"`, `"SelfAttentionMLP"`, `"FTTransformer"`, `"TabPerceiver"`, `"TabFastFormer"`.
* [`pytorch_tabular`](https://github.com/manujosephv/pytorch_tabular)
  * `"Category Embedding"`, `"NODE"`, `"TabNet"`, `"TabTransformer"`, `"AutoInt"`, `"FTTransformer"`.

You are able to implement your own models, data processing pipelines, and datasets under the flexible and 
well-tested framework for consistent comparisons with baseline models, which is even easier when your own model is 
based on `pytorch`. 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/0fe47266-ae58-4e6b-bcf6-1108ebd762bc">

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
* Data/result analysis
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

The package stands on the shoulder of the giants:

* [scikit-learn](https://scikit-learn.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/)
* etc. (See `requirements.txt`)


## Installation/Usage

A full documentation is available [here](https://tabular-ensemble.readthedocs.io/en/latest/index.html). For a quick start:

1. `tabular_ensemble` can be installed using pypi by running the following command:

```shell
pip install tabensemb[torch]
```

Please use `pip install tabensemb` instead if you already have `torch>=1.12.0` installed. Use `pip install tabensemb[test]` if you want to run unit tests. 

2. (Optional) Run unit tests after installed `tabensemb[test]`:

```shell
cd test
pytest .
```

3. Place your `.csv` or `.xlsx` file in a `data` subfolder (e.g., `data/sample.csv`), and generate a configuration file in a `configs` subfolder (e.g., `configs/sample.py`), containing the following content
```python
cfg = {
    "database": "sample",
    "continuous_feature_names": ["cont_0", "cont_1", "cont_2", "cont_3", "cont_4"],
    "categorical_feature_names": ["cat_0", "cat_1", "cat_2"],
    "label_name": ["target"],
}
```

4. Run the experiment using the configuration and the data using
```python
python main.py --base sample --epoch 10
```
where `--base` refers to the configuration file, and additional arguments (such as `--epoch` here) refer to those in `config/default.py`.

See the [documentation pages](https://tabular-ensemble.readthedocs.io/en/latest/index.html) for details.

## Citation

If you use this repository, please cite us as:

```text
(Will be updated after released on arXiv or published)
```
