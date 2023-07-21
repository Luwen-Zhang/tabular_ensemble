# tabular_ensemble
 A framework to ensemble model bases and evaluate various models for tabular predictions.

## Usage

The bash script `run_sample.sh` give a example to use the main script `main.py`.

A configuration file in `.py` or `.json` format provides all information to build a training instance and run scripts, including training epochs, bayesian hyperparameter optimization, and data processing. See examples in `configs`.

## Requirements

We recommend using a virtual environment with `python=3.8` and conda (we have verified both [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [anaconda](https://www.anaconda.com/)).

```shell
conda create -n myvenv python=3.8
conda activate myvenv
```

First, install `torch==1.12.0` with CUDA 1.16 (if a Nvidia GPU is available). 

```shell
pip install torch==1.12.0+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
```

Then install the package

```shell
pip install -e .[all]
```

If the minimal functionality is needed (without any other model bases), install without the postfix`[all]`.

To test basic functionalities like loading data and training model bases, run the following `unittest`

```shell
cd test
python test_general.py
```

## Modelbases

The respository merges following well-established modelbases as baselines:

* [`autogluon`](https://github.com/autogluon/autogluon)

* [`pytorch_widedeep`](https://github.com/jrzaurin/pytorch-widedeep)

* [`pytorch_tabular`](https://github.com/manujosephv/pytorch_tabular)


## Implementing new features

New modelbases, individual models, tabular databases, feature derivation procedures, data preprocessing procedures, etc., can be easily extended in the framework. New features follow certain structures given by parent classes, and several methods should be implemented for them.

We wrote some useful utilities specifically for fatigue prediction and composite laminates, like S-N curve calculation, lay-up sequence treatment, etc.

### New data processing procedure

`tabensemb.data.DataModule` controls data processing, which includes four parts:

* **Data splitting**: Split the dataset into training, validation, and testing sets. Inherit `tabensemb.data.AbstractSplitter` and implement the `_split` method. See examples in `tabensemb.data.datasplitter.py`.

  **Remark**: After implementation, update `tabensemb.data.datasplitter.splitter_mapping` to include the implemented feature.

* **Data imputation**: Fill missing values. Inherit `tabensemb.data.AbstractImputer` and implement `_fit_transform` and `_transform`, or `tabensemb.data.AbstractSklearnImputer` (for `sklearn` type imputers)  and implement `_new_imputer`. See examples in `tabensemb.data.dataimputer`.

  **Remark**: After implementation, update `tabensemb.data.dataimputer.imputer_mapping` to include the implemented feature.

* **Data derivation**: Derive new features based on the given dataset. This is one type of data augmentation. Inherit `tabensemb.data.AbstractDeriver` and implement 

  * `_derive`
  * `_defaults` for default arguments of `_derive`
  * `_derived_names` for a list of names of derived features
  * `_required_cols` for necessary data columns to run the derivation
  * `_required_params` for necessary arguments.

  See examples in `tabensemb.data.dataderiver.py`.

  **Remark**: After implementation, update `tabensemb.data.dataderiver.deriver_mapping` to include the implemented feature.

  **Remark**: In `_defaults`, `stacked` and `intermediate` are necessary. `stacked=True` means that the derived feature is stacked to the dataframe as a tabular feature, otherwise the feature is stored in `DataModule.derived_data` and can be accessed using `DataModule.D_train/val/test` (Note that these features can be **multi-modal**). When `stacked=True`,  `intermediate=False` means that the derived feature will be added to `DataModule.cont_feature_names`, otherwise it will not.

* **Data processing**: Functions that increase/decrease the number of data points (data filtering and augmentation), decrease the number of features (feature selection), and modify values of data (category encoding or scaling, etc.).  See examples in `tabensemb.data.dataprocessor.py`

  * *Data filtering*: Inherit `tabensemb.data.AbstractProcessor` and implement `_fit_transform` and `_transform`. Note that when inferring new data (`datamodule.training==True`), do not decrease the number of data points.
  * *Data augmentation*: Inherit `tabensemb.data.AbstractAugmenter` and implement `_get_augmented`.
  * *Feature selection*: Inherit `tabensemb.data.AbstractFeatureSelector` and implement `_get_feature_names_out`.
  * *Modify values of data*: Inherit `tabensemb.data.AbstractTransformer` and implement `_fit_transform` and `_transform`. Specifically, scalers like `StandardScaler` and `Normalizer` inherit `tabensemb.data.AbstractScaler`. 

  **Remark**: After implementation, update `tabensemb.data.dataprocessor.processor_mapping` to include the implemented feature.

  **Remark**: All modules run `_fit_transform` on the training and validation set and `_transform` on the testing set when `datamodule.training==True`, and run `_transform` on the new data when `datamodule.training==False`.

  **Remark**: There must be an `AbstractScaler` at the end of `datamodule.dataprocessors`

**Remark**: Do not change the index of the input `DataFrame`.

To set up the above features for a `DataModule`, provide a `Dict` that contains `data_splitter`, `data_imputer`, `data_processors`, and `data_derivers` to set up internally implemented modules. See examples in `configs`. If users implement new features, modify `datamodule.datasplitter/dataimputer/dataprocessors/dataderivers` directly, but make sure that the basic restrictions in remarks are satisfied.

### New model base and models

Model bases inherit from `tabensemb.model.AbstractModel`. Some methods should be implemented:

* `_get_model_names` get a list of available models
* `_get_program_name` the name of the model base
* `_new_model` initialize a new model given its name.
* `_train_data_preprocess` process data from a `DataModule` instance embeded in the `Trainer`
* `_data_preprocess` perform the same data processing as `_train_data_preprocess` when inferencing
* `_train_single_model` train a model given the model from `_new_model` and data from `_train_data_preprocess`
* `_pred_single_model` infer new data using a trained model from `_new_model`
* `_space` define the hyperparameter searching space
* `_initial_values` define default hyperparameters
* `_conditional_validity` check the validity of a model under certain circumstances using the `Trainer` 

**Remark**: If the model base is for `torch` models, inherit from `tabensemb.model.TorchModel` which has implemented useful utilities, including `_train_data_preprocess`, `_data_preprocess`, `_train_single_model`, and`_pred_single_model`, based on the powerful pytorch extension [`pytorch_lightning`](https://github.com/Lightning-AI/lightning). The only restriction is that the returned model from `_new_model` should inherit `tabensemb.model.AbstractNN`.

**Remark**: If model bases `AbstractModel` share the same `Trainer` (i.e. the configuration file), they share the same data preprocessing procedure (i.e. `DataModule`) embeded in the `Trainer`. If someone focuses on the improvement of data preprocessing (e.g. physics-informed data augmentation), create an individual `DataModule` instance inside the customized `AbstractModel` , specifically in `_train_data_preprocess` and `_data_preprocess`, where the embeded `DataModule` and data processed by it are passed as arguments for further customized processing. Note that `_data_preprocess` receives data that is not scaled (e.g. standard scaled or normalized).

### New data base

Since we aim to build a transparent benchmark platform, users are welcomed to upload datasets of their researches.

## Contribution

Feel free to create issues if you implement new features, propose a new model, find mistakes, or have any question.

