import os.path
import numpy as np
import pandas as pd
import tabensemb
from tabensemb.utils import *
from tabensemb.config import UserConfig
from copy import deepcopy as cp
from typing import *
import torch.cuda
import torch.utils.data as Data
from torch.utils.data import Subset
from sklearn.decomposition import PCA
import sklearn.pipeline
import sklearn.ensemble
from collections.abc import Iterable
from sklearn.preprocessing import OrdinalEncoder
import scipy.stats as st
from functools import reduce


class DataModule:
    """
    The dataset manager that provides loading, processing, and analyzing utilities.

    Attributes
    ----------
    args
        A :class:`tabensemb.config.UserConfig` instance.
    augmented_indices
        Indices of data points that are augmented by :class:`tabensemb.data.AbstractAugmenter`. The first index starts
        from the number of data points in the original dataset.
    cat_feature_mapping
        Original values of categorical features before ordinal encoding. The index of a value represents the encoded
        value.
    cat_feature_names
        Names of categorical features.
    cont_feature_names
        Names of continuous features.
    data_path
        The path to the data file.
    dataderivers
        A list of :class:`tabensemb.data.AbstractDeriver`.
    dataimputer
        A :class:`tabensemb.data.AbstractImputer`.
    dataprocessors
        A list of :class:`tabensemb.data.AbstractProcessor`.
    datasplitter
        A :class:`tabensemb.data.AbstractSplitter`.
    derived_data
        The derived unstacked data calculated using :attr:`dataderivers` whose argument "stacked" is set to False.
    df
        The unscaled processed dataset.
    dropped_indices
        Indices of data points that are removed from the original dataset.
    label_name
        The name(s) of target(s)
    label_ordinal_encoder
        A ``sklearn.preprocessing.OrdinalEncoder`` that encodes the classification targets.
    loss
        The type of the loss function. See :meth:`_infer_loss`.
    n_classes
        The number of unique values for each classification target.
    retained_indices
        Indices of data points that are retained in the original dataset.
    scaled_df
        The scaled processed dataset. See also :attr:`df`.
    task
        The type of the task. See :meth:`_infer_task`.
    tensors
        :meth:`feature_data`, expanded :attr:`derived_data`, and :meth:`label_data` in the torch.Tensor form.
    test_dataset
        The testing set of the entire ``torch.utils.data.Dataset``.
    test_indices
        Indices of the testing set in the entire dataset (:attr:`df`).
    train_dataset
        The training set of the entire ``torch.utils.data.Dataset``.
    train_indices
        Indices of the training set in the entire dataset (:attr:`df`).
    training
        The training status of the :class:`DataModule`. See :meth:`set_status`.
    val_dataset
        The validation set of the entire ``torch.utils.data.Dataset``.
    val_indices
        Indices of the validation set in the entire dataset (:attr:`df`).
    D_test
    D_train
    D_val
    X_test
    X_train
    X_val
    all_feature_names
    cat_imputed_mask
    cat_num_unique
    categorical_data
    cont_imputed_mask
    derived_stacked_features
    feature_data
    label_data
    unscaled_feature_data
    unscaled_label_data
    y_test
    y_train
    y_val
    """

    def __init__(
        self,
        config: Union[UserConfig, Dict],
        initialize: bool = True,
    ):
        self.args = config
        if initialize:
            self.set_data_splitter(
                self.args["data_splitter"], ratio=self.args["split_ratio"]
            )
            self.set_data_imputer(self.args["data_imputer"])
            self.set_data_processors(self.args["data_processors"])
            self.set_data_derivers(self.args["data_derivers"])
        self.training = False
        self.data_path = None

    def set_status(self, training: bool):
        """
        Set the status of the datamodule. If a datamodule is not training, some data processing modules will use learned
        characteristics to process new data. True when setting the dataset. False when processing an upcoming dataset.

        Parameters
        ----------
        training
            The training status of the datamodule.
        """
        self.training = training

    def set_data_splitter(
        self,
        config: Union[str, Tuple[str, Dict]],
        ratio: Union[List[float], np.ndarray] = None,
    ):
        """
        Set the data splitter. The specified splitter should be registered in
        ``tabensemb.data.datasplitter.splitter_mapping``.

        Parameters
        ----------
        config
            The name of a data splitter or a tuple providing the name and kwargs of the data splitter.
        ratio
            The ratio of training, validation, and testing sets. For example, [0.6, 0.2, 0.2].
        """
        from tabensemb.data.datasplitter import get_data_splitter

        if type(config) in [tuple, list]:
            self.datasplitter = get_data_splitter(config[0])(**dict(config[1]))
        else:
            if ratio is None:
                self.datasplitter = get_data_splitter(config)()
            else:
                self.datasplitter = get_data_splitter(config)(train_val_test=ratio)

    def set_data_imputer(self, config):
        """
        Set the data imputer. The specified splitter should be registered in
        ``tabensemb.data.dataimputer.imputer_mapping``.

        Parameters
        ----------
        config
            The name of a data imputer or a tuple providing the name and kwargs of the data imputer.
        """
        from tabensemb.data.dataimputer import get_data_imputer

        if type(config) in [tuple, list]:
            self.dataimputer = get_data_imputer(config[0])(**dict(config[1]))
        else:
            self.dataimputer = get_data_imputer(config)()

    def set_data_processors(self, config: List[Tuple[str, Dict]]):
        """
        Set a list of data processors containing the name and arguments for each data processor. The processor should be
        registered in ``tabensemb.data.dataprocessor.processor_mapping``.

        Parameters
        ----------
        config
            A list of tuples. Each tuple includes the name of the processor and a dict of kwargs for the processor.

        Notes
        -----
        Only one :class:`~tabensemb.data.base.AbstractScaler` can be used, and the
        :class:`~tabensemb.data.base.AbstractScaler` must be the last one.
        """
        from tabensemb.data.dataprocessor import get_data_processor, AbstractScaler

        self.dataprocessors = [
            get_data_processor(name)(**kwargs) for name, kwargs in config
        ]
        is_scaler = np.array(
            [int(isinstance(x, AbstractScaler)) for x in self.dataprocessors]
        )
        if np.sum(is_scaler) > 1:
            raise Exception(f"More than one AbstractScaler.")
        if is_scaler[-1] != 1:
            raise Exception(f"The last dataprocessor should be an AbstractScaler.")

    def set_data_derivers(self, config: List[Tuple[str, Dict]]):
        """
        Set a list of data derivers containing the name and arguments for each data deriver. The deriver should be
        registered in ``tabensemb.data.dataderiver.deriver_mapping``.

        Parameters
        ----------
        config
            A list of tuple. Each tuple includes the name of the deriver and a dict of kwargs for the deriver.
        """
        from tabensemb.data.dataderiver import get_data_deriver

        self.dataderivers = [
            get_data_deriver(name)(**kwargs) for name, kwargs in config
        ]

    def load_data(
        self,
        data_path: str = None,
        save_path: str = None,
        **kwargs,
    ) -> None:
        """
        Load tabular data. Either a .csv or .xlsx file is supported.

        Parameters
        ----------
        data_path
            Path to the tabular data. By default, the file
            ``tabensemb.setting["default_data_path"]/{database}.csv(.xlsx)`` is loaded where "database" is given in the
            configuration.
        save_path
            Path to save the loaded data.
        **kwargs
            Arguments for ``pd.read_excel`` or ``pd.read_csv``.
        """
        if data_path is None:
            if self.data_path is None:
                path = self.args["database"]
                data_path = (
                    path
                    if "/" in path or os.path.isfile(path)
                    else os.path.join(tabensemb.setting["default_data_path"], path)
                )
            else:
                print(f"Using previously used data path {self.data_path}")
                data_path = self.data_path
        file_type = os.path.splitext(data_path)[-1]
        if file_type == "":
            is_csv = os.path.isfile(data_path + ".csv")
            is_xlsx = os.path.isfile(data_path + ".xlsx")
            if is_csv and is_xlsx:
                raise Exception(
                    f"Both {data_path}.csv and {data_path}.xlsx exist. Provide the postfix in data_path."
                )
            if not is_csv and not is_xlsx:
                raise Exception(f"{data_path}.csv and .xlsx do not exist.")
            file_type = ".csv" if is_csv else ".xlsx"
            data_path = data_path + file_type
        if file_type == ".xlsx":
            self.df = pd.read_excel(data_path, engine="openpyxl", **kwargs)
        else:
            self.df = pd.read_csv(data_path, **kwargs)
        self.data_path = data_path

        cont_feature_names = self.extract_cont_feature_names(
            self.args["feature_names_type"].keys()
        )
        cat_feature_names = self.extract_cat_feature_names(
            self.args["feature_names_type"].keys()
        )
        label_name = self.args["label_name"]

        self.set_data(self.df, cont_feature_names, cat_feature_names, label_name)
        print(
            "Dataset size:",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        if save_path is not None:
            self.save_data(save_path)

    def set_data(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        derived_stacked_features: List[str] = None,
        derived_data: Dict[str, np.ndarray] = None,
        warm_start: bool = False,
        verbose: bool = True,
        all_training: bool = False,
        train_indices: np.ndarray = None,
        val_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
    ):
        """
        Set up the datamodule with a DataFrame. Data splitting, imputation, derivation, and processing will be performed.

        Parameters
        ----------
        df
            The tabular dataset.
        cont_feature_names
            A list of continuous features in the tabular dataset.
        cat_feature_names
            A list of categorical features in the tabular dataset.
        label_name
            A list of targets. Multi target tasks are experimental.
        derived_stacked_features
            A list of derived features in the tabular dataset. If not None, only these features are retained after
            derivation, and all AbstractFeatureSelectors will be skipped.
        derived_data
            The derived data calculated using data derivers whose argument "stacked" is set to False, i.e. unstacked
            data. Unstacked derivations will be skipped if it is given.
        warm_start
            Whether to use fitted data processors to process the data.
        verbose
            Verbosity.
        all_training
            Whether all samples are used for training.
        train_indices
            Manually specify the training indices.
        val_indices
            Manually specify the validation indices.
        test_indices
            Manually specify the testing indices.
        """
        if len(label_name) > 1:
            warnings.warn(
                f"Multi-target task is currently experimental. Some model base does not support multi-target"
                f"well, pytorch-widedeep for example."
            )

        detected_cat_feature_names = df.dtypes.index[df.dtypes == object]
        illegal_cont_features = list(
            np.intersect1d(detected_cat_feature_names, cont_feature_names)
        )
        for feature in illegal_cont_features:
            try:
                df[feature] = df[feature].values.astype(np.float64)
                illegal_cont_features.remove(feature)
            except:
                pass
        if len(illegal_cont_features) > 0:
            raise Exception(
                f"{illegal_cont_features} are object, but are included in continuous features. Please remove them "
                f"or add them to `categorical_feature_names` in the configuration file."
            )

        self.set_status(training=True)
        self.cont_feature_names = cont_feature_names
        self.cat_feature_names = cat_feature_names
        self.cat_feature_mapping = {}
        for feature in self.cat_feature_names:
            self.cat_feature_mapping[feature] = []
        self.label_name = label_name
        self.df = df.copy()
        if pd.isna(df[self.label_name]).any().any():
            raise Exception("Label missing in the input dataframe.")

        if all_training:
            self.train_indices = self.test_indices = self.val_indices = np.arange(
                len(self.df)
            )
        elif train_indices is None or val_indices is None or test_indices is None:
            (
                self.train_indices,
                self.val_indices,
                self.test_indices,
            ) = self.datasplitter.split(
                self.df, cont_feature_names, cat_feature_names, label_name
            )
        else:
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.test_indices = test_indices

        self._cont_imputed_mask = pd.DataFrame(
            columns=self.cont_feature_names,
            data=np.isnan(self.unscaled_feature_data.values).astype(int),
            index=np.arange(len(self.df)),
        )
        self._cat_imputed_mask = pd.DataFrame(
            columns=self.cat_feature_names,
            data=pd.isna(self.df[self.cat_feature_names]).values.astype(int),
            index=np.arange(len(self.df)),
        )

        def make_imputation():
            train_val_indices = list(self.train_indices) + list(self.val_indices)
            self.df.loc[train_val_indices, :] = self.dataimputer.fit_transform(
                self.df.loc[train_val_indices, :], datamodule=self
            )
            self.df.loc[self.test_indices, :] = self.dataimputer.transform(
                self.df.loc[self.test_indices, :], datamodule=self
            )

        make_imputation()
        (
            self.df,
            cont_feature_names,
        ) = self.derive_stacked(self.df)
        if derived_stacked_features is not None:
            current_derived_stacked_features = (
                self.extract_derived_stacked_feature_names(cont_feature_names)
            )
            removed = list(
                np.setdiff1d(current_derived_stacked_features, derived_stacked_features)
            )
            cont_feature_names = [x for x in cont_feature_names if x not in removed]
        self.cont_feature_names = cont_feature_names
        # There may exist nan in stacked features.
        self._cont_imputed_mask = pd.concat(
            [
                self._cont_imputed_mask,
                pd.DataFrame(
                    columns=self.derived_stacked_features,
                    data=np.isnan(self.df[self.derived_stacked_features].values).astype(
                        int
                    ),
                    index=np.arange(len(self.df)),
                ),
            ],
            axis=1,
        )[self.cont_feature_names]
        make_imputation()

        self._data_process(
            warm_start=warm_start,
            verbose=verbose,
            skip_selector=derived_stacked_features is not None,
        )

        self._cont_imputed_mask = (
            self._cont_imputed_mask.loc[self.retained_indices, self.cont_feature_names]
            .copy()
            .reset_index(drop=True)
        )
        self._cat_imputed_mask = (
            self._cat_imputed_mask.loc[self.retained_indices, self.cat_feature_names]
            .copy()
            .reset_index(drop=True)
        )

        def update_indices(indices):
            return np.array(
                [
                    x - np.count_nonzero(self.dropped_indices < x)
                    for x in indices
                    if x in self.retained_indices
                ]
            )

        self.train_indices = update_indices(self.train_indices)
        self.test_indices = update_indices(self.test_indices)
        self.val_indices = update_indices(self.val_indices)

        if len(self.augmented_indices) > 0:
            augmented_indices = self.augmented_indices - len(self.dropped_indices)
            np.random.shuffle(augmented_indices)
            self.train_indices = np.array(
                list(self.train_indices) + list(augmented_indices)
            )

        if (
            len(self.train_indices) == 0
            or len(self.val_indices) == 0
            or len(self.test_indices) == 0
        ):
            raise Exception(
                "No sufficient data after preprocessing. This is caused by arguments train/val/test_"
                "indices or warm_start of set_data()."
            )

        self.derived_data = (
            self.derive_unstacked(self.df)
            if derived_data is None
            else self.sort_derived_data(derived_data)
        )
        self.task = self._infer_task()
        self.loss = self._infer_loss(self.task)
        if self.task in ["binary", "multiclass"]:
            self.n_classes = [
                len(np.unique(self.label_data[col])) for col in self.label_name
            ]
            self.label_ordinal_encoder = OrdinalEncoder()
            res = self.label_ordinal_encoder.fit_transform(self.label_data).astype(int)
            self.df[self.label_name] = res
            self.scaled_df[self.label_name] = res
        else:
            self.label_ordinal_encoder = None
            self.n_classes = [None]
        self.update_dataset()
        self.set_status(training=False)

    @property
    def cat_num_unique(self) -> List[int]:
        """
        The number of unique values of each categorical feature. Only valid when a
        :class:`~tabensemb.data.dataprocessor.CategoricalOrdinalEncoder` is used.
        """
        return (
            [len(x) for x in self.cat_feature_mapping.values()]
            if hasattr(self, "cat_feature_mapping")
            else []
        )

    def _infer_task(self) -> str:
        """
        Automatically infer the task type using the target values. If the inferred type is not the same as that set
        in the configuration, the inferred type is used if the global setting
        ``tabensemb.setting["raise_inconsistent_inferred_task"]`` is False.

        Returns
        -------
        str
            "binary", "multiclass", or "regression"
        """
        selected_task = self.args["task"] if "task" in self.args.keys() else None
        # if isinstance(selected_task, dict):
        #     return selected_task
        if selected_task is not None and not isinstance(selected_task, str):
            raise Exception(f"Multiple tasks is not supported.")
        available_tasks = ["binary", "multiclass", "regression"]

        def infer_one_col(col, task):
            if col.values.dtype == object:
                is_cat = True
            else:
                try:
                    is_cat = np.all(np.mod(col, 1) == 0)
                except:
                    raise Exception(f"Unrecognized target type {col.values.dtype}.")
            if is_cat:
                n_unique = len(np.unique(col))
                if n_unique <= 2:
                    infer_task = "binary"
                else:
                    infer_task = "multiclass"
            else:
                infer_task = "regression"
            if task is None:
                return infer_task
            elif task is not None and task not in available_tasks:
                raise Exception(f"Unsupported task {task}.")
            elif infer_task != task:
                if tabensemb.setting["raise_inconsistent_inferred_task"]:
                    raise Exception(
                        f"The inferred task {infer_task} is not consistent with the selected task {task}."
                    )
                else:
                    warnings.warn(
                        f"The inferred task {infer_task} is not consistent with the selected task {task}. Using the "
                        f"selected task."
                    )
                    return task
            else:
                return task

        if len(self.label_name) > 1:
            task = [
                infer_one_col(self.label_data[name], selected_task)
                for name in self.label_name
            ]
            if any([t in ["binary", "multiclass"] for t in task]):
                raise Exception(f"Multi-target classification task is not supported.")
            else:
                task = "regression"
        else:
            task = infer_one_col(self.label_data, selected_task)
        return task

    def _infer_loss(self, task: str):
        """
        Automatically infer the loss type using the name of the task and the loss type given in the configuration.

        Parameters
        ----------
        task
            "binary", "multiclass", or "regression"

        Returns
        -------
        str
            "mse" or "mae" for regression tasks and "cross_entropy" for classification tasks.
        """
        selected_loss = self.args["loss"] if "loss" in self.args.keys() else None
        # if isinstance(selected_loss, dict):
        #     return selected_loss
        if selected_loss is not None and not isinstance(selected_loss, str):
            raise Exception(f"Multiple losses is not supported.")
        if task in ["binary", "multiclass"]:
            available_losses = ["cross_entropy"]
        else:
            available_losses = ["mse", "mae"]
        if selected_loss is None:
            loss = available_losses[0]
        else:
            if selected_loss not in available_losses:
                raise Exception(
                    f"The selected loss {selected_loss} is not supported for {task} tasks"
                )
            loss = selected_loss
        return loss

    def prepare_new_data(
        self, df: pd.DataFrame, derived_data: dict = None, ignore_absence=False
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Prepare the new tabular dataset for predictions using :meth:`~tabensemb.model.AbstractModel._predict`
        Stacked and unstacked features are derived; missing values are imputed; The ``transform`` method of
        :class:`~tabensemb.data.base.AbstractProcessor` is called. Users usually do not need to
        call this because :meth:`~tabensemb.model.AbstractModel.predict` does it.

        Parameters
        ----------
        df:
            A new tabular dataset.
        derived_data:
            Data derived from :meth:`.derive_unstacked`. If not None, unstacked data will be re-derived.
        ignore_absence:
            Whether to ignore absent keys in derived_data. Use True only when the model does not use derived_data.

        Returns
        -------
        df
            The dataset after derivation, imputation, and processing. It has the same structure as ``self.X_train``
        derived_data:
            Data derived from :meth:`.derive_unstacked`. It has the same structure as ``self.D_train``

        Notes
        -----
        The returned ``df`` is not scaled for the sake of further treatments. To scale the df,
        run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        self.set_status(training=False)
        absent_features = [
            x
            for x in np.setdiff1d(self.all_feature_names, self.derived_stacked_features)
            if x not in df.columns
        ]
        absent_derived_features = [
            x for x in self.derived_stacked_features if x not in df.columns
        ]
        if len(absent_features) > 0:
            raise Exception(f"Feature {absent_features} not in the input dataframe.")

        # Predicting on a dataset whose categorical features are already encoded. It may interrupt
        # data derivers that rely on categorical features, or get inconsistent results between
        # _predict and predict.
        df = self.categories_inverse_transform(df)

        if derived_data is None or len(absent_derived_features) > 0:
            df, _, derived_data = self.derive(df)
        else:
            absent_keys = [
                key
                for key in self.derived_data.keys()
                if key not in derived_data.keys()
            ]
            if len(absent_keys) > 0 and not ignore_absence:
                raise Exception(
                    f"Additional feature {absent_keys} not in the input derived_data."
                )
        df = self.dataimputer.transform(df.copy(), self)
        df = self.data_transform(df, skip_scaler=True)
        derived_data = self.sort_derived_data(
            derived_data, ignore_absence=ignore_absence
        )
        if getattr(self, "label_ordinal_encoder", None) is not None:
            df = self.label_categories_transform(df)
        return df, derived_data

    @property
    def cont_imputed_mask(self) -> pd.DataFrame:
        """
        A byte mask for continuous data, where 1 means the data is imputed, and 0 means the data originally exists.

        Returns
        -------
        pd.DataFrame
            A byte mask dataframe.
        """
        return self._cont_imputed_mask[self.cont_feature_names]

    @property
    def cat_imputed_mask(self) -> pd.DataFrame:
        """
        A byte mask for categorical data, where 1 means the data is imputed, and 0 means the data originally exists.

        Returns
        -------
        pd.DataFrame
            A byte mask dataframe.
        """
        return self._cat_imputed_mask[self.cat_feature_names]

    @property
    def all_feature_names(self) -> List[str]:
        """
        Get continuous feature names and categorical feature names after ``load_data()``.

        Returns
        -------
        List
            A list of continuous features and categorical features.
        """
        return self.cont_feature_names + self.cat_feature_names

    @property
    def derived_stacked_features(self) -> List[str]:
        """
        Find derived features in ``all_feature_names`` derived by data derivers whose argument "stacked" is set to True,
        i.e. the stacked data.

        Returns
        -------
        List
            A list of feature names.
        """
        return self.extract_derived_stacked_feature_names(self.all_feature_names)

    def get_feature_types(self, features: List[str]) -> List[str]:
        """
        Get the type of each feature in ``features``, which is defined by ``feature_names_type`` and ``feature_types``
        in the configuration.

        Parameters
        ----------
        features
            A list of features.

        Returns
        -------
        list
            The type of each feature
        """
        invalid_features = [
            feature
            for feature in features
            if feature not in self.args["feature_names_type"].keys()
            and feature not in self.get_all_derived_stacked_feature_names()
        ]
        if len(invalid_features) > 0:
            raise Exception(f"Unknown features: {invalid_features}")
        return [
            self.args["feature_types"][
                self.args["feature_names_type"].get(
                    i, self.args["feature_types"].index("Derived")
                )
            ]
            for i in features
        ]

    def get_feature_names_by_type(self, typ: str) -> List[str]:
        """
        Find features of the specified type defined by ``feature_names_type`` and ``feature_types`` in the configuration.

        Parameters
        ----------
        typ
            One type of features defined in ``feature_types`` in the configuration.

        Returns
        -------
        List
            A list of found features.

        See Also
        --------
        :meth:`get_feature_idx_by_type`
        """
        if typ not in self.args["feature_types"]:
            raise Exception(
                f"Feature type {typ} is invalid (among {self.args['feature_types']})"
            )
        return [
            name
            for name, type_idx in self.args["feature_names_type"].items()
            if type_idx == self.args["feature_types"].index(typ)
            and name in self.all_feature_names
        ]

    def get_feature_idx_by_type(self, typ: str) -> np.ndarray:
        """
        Find features (by their index) of the specified type defined by ``feature_names_type`` and ``feature_types``
        in the configuration.

        Parameters
        ----------
        typ
            One type of features in ``feature_types`` in the configuration.

        Returns
        -------
        np.ndarray
            A list of indices of found features.

        See Also
        --------
        :meth:`get_feature_names_by_type`
        """
        names = self.get_feature_names_by_type(typ=typ)
        if typ == "Categorical":
            return np.array([self.cat_feature_names.index(name) for name in names])
        else:
            return np.array([self.cont_feature_names.index(name) for name in names])

    def extract_cont_feature_names(self, all_feature_names: List[str]) -> List[str]:
        """
        Get original continuous features specified in the configuration.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some original features in the configuration.

        Returns
        -------
        List
            Names of continuous original features both in the configuration and the input list.
        """
        return [
            x
            for x in all_feature_names
            if x in self.args["feature_names_type"].keys()
            and x not in self.args["categorical_feature_names"]
        ]

    def extract_cat_feature_names(self, all_feature_names: List[str]) -> List[str]:
        """
        Get original categorical features specified in the configuration.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some original features in the configuration.

        Returns
        -------
        List
            Names of categorical original features that are both in the configuration and the input list.
        """
        return [
            str(x)
            for x in np.intersect1d(
                list(all_feature_names),
                self.args["categorical_feature_names"],
            )
        ]

    def extract_derived_stacked_feature_names(
        self, all_feature_names: List[str]
    ) -> List[str]:
        """
        Find derived features in the input list derived by data derivers whose argument "stacked" is set to True,
        i.e. the stacked data.

        Parameters
        ----------
        all_feature_names
            A list of features that contains some stacked features.

        Returns
        -------
        List
            Names of stacked features in the input list.
        """
        return [
            str(x)
            for x in all_feature_names
            if x in self.get_all_derived_stacked_feature_names()
        ]

    def get_all_derived_stacked_feature_names(self):
        """
        Get all derived stacked features (not intermediate) from arguments of :attr:`dataderivers`.

        Returns
        -------
        List
            Names of all derived stacked from the current data derivers.
        """
        names = []
        for deriver in self.dataderivers:
            if deriver.kwargs["stacked"] and not deriver.kwargs["intermediate"]:
                names += deriver.last_derived_col_names
        return names

    def set_feature_names(self, all_feature_names: List[str]):
        """
        Set feature names to a subset of current features (i.e. ``self.all_feature_names``) and reload the data.

        Parameters
        ----------
        all_feature_names
            A subset of current features.
        """
        self.set_status(training=True)
        cont_feature_names = self.extract_cont_feature_names(all_feature_names)
        cat_feature_names = self.extract_cat_feature_names(all_feature_names)
        derived_stacked_features = self.extract_derived_stacked_feature_names(
            all_feature_names
        )
        has_indices = hasattr(self, "train_indices")
        self.set_data(
            self.df,
            cont_feature_names,
            cat_feature_names,
            self.label_name,
            derived_stacked_features=derived_stacked_features,
            verbose=False,
            train_indices=self.train_indices if has_indices else None,
            val_indices=self.val_indices if has_indices else None,
            test_indices=self.test_indices if has_indices else None,
        )
        self.set_status(training=False)

    def sort_derived_data(
        self, derived_data: Dict[str, np.ndarray], ignore_absence: bool = False
    ) -> Union[Dict[str, np.ndarray], None]:
        """
        Sort the dict of derived unstacked data according to the order of derivation.

        Parameters
        ----------
        derived_data
            A dict of derived unstacked data calculated by :meth:`derive_unstacked`
        ignore_absence
            Whether to ignore absent keys in derived_data.

        Returns
        -------
        dict or None
            The sorted derived unstacked data.
        """
        if derived_data is None:
            return None
        else:
            tmp_derived_data = {}
            for key in self.derived_data.keys():
                if ignore_absence:
                    if key in derived_data.keys():
                        tmp_derived_data[key] = derived_data[key]
                else:
                    tmp_derived_data[key] = derived_data[key]
            return tmp_derived_data

    def _get_categorical_ordinal_encoder(self):
        """
        Find and return the :class:`~tabensemb.data.dataprocessor.CategoricalOrdinalEncoder` in data processors..

        Returns
        -------
        CategoricalOrdinalEncoder
        """
        from tabensemb.data.dataprocessor import CategoricalOrdinalEncoder

        for processor in self.dataprocessors:
            if isinstance(processor, CategoricalOrdinalEncoder):
                encoder = processor.transformer
                cat_features = processor.record_cat_features
                if len(cat_features) == 0:
                    return None
                return encoder, cat_features
        else:
            return None

    def categories_inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation of :class:`~tabensemb.data.dataprocessor.CategoricalOrdinalEncoder` for categorical
        features (If there is one in ``self.dataprocessors``).

        Parameters
        ----------
        X
            The data to be inverse-transformed.

        Returns
        -------
        pd.DataFrame
            The inverse-transformed data.
        """
        encoder_features = self._get_categorical_ordinal_encoder()
        if encoder_features is None:
            return X.copy()
        else:
            encoder, cat_features = encoder_features
        X = self._ordinal_encoder_transform(encoder, X, cat_features, transform=False)
        return X

    def categories_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformation of :class:`~tabensemb.data.dataprocessor.CategoricalOrdinalEncoder` for categorical features
        (If there is one in ``self.dataprocessors``).

        Parameters
        ----------
        X
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        encoder_features = self._get_categorical_ordinal_encoder()
        if encoder_features is None:
            return X.copy()
        else:
            encoder, cat_features = encoder_features
        X = self._ordinal_encoder_transform(encoder, X, cat_features, transform=True)
        return X

    def label_categories_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical ordinal encoding for the target.

        Parameters
        ----------
        X
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        res = self._ordinal_encoder_transform(
            self.label_ordinal_encoder, X, self.label_name, transform=True
        )
        res[self.label_name] = res[self.label_name].astype(int)
        return res

    def label_categories_inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation of categorical ordinal encoding for the target.

        Parameters
        ----------
        X
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        return self._ordinal_encoder_transform(
            self.label_ordinal_encoder, X, self.label_name, transform=False
        )

    @staticmethod
    def _ordinal_encoder_transform(
        encoder: OrdinalEncoder,
        X: pd.DataFrame,
        cat_features: List[str],
        transform: bool,
    ) -> pd.DataFrame:
        """
        For :meth:`categories_inverse_transform`, :meth:`categories_transform`,
        :meth:`label_categories_transform`, and :meth:`label_categories_inverse_transform`, make the input legal
        including getting ready for missing columns, return the input if it is already transformed, etc.

        Parameters
        ----------
        encoder
            A categorical ordinal encoder that has methods ``fit_transform`` and ``transform``.
        X
            The data to be transformed.
        cat_features
            Encoded categorical features (or targets).
        transform
            True if doing transform and False if doing inverse transform.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        X = X.copy()
        missing_cols = np.setdiff1d(cat_features, list(X.columns)).astype(str)
        if len(missing_cols) > 0:
            X[missing_cols] = -1
        X.columns = X.columns.astype(str)
        try:
            if transform:
                X[cat_features] = encoder.transform(X[cat_features].copy()).astype(int)
            else:
                X[cat_features] = encoder.inverse_transform(X[cat_features].copy())
        except:
            try:
                if transform:
                    encoder.inverse_transform(X[cat_features].copy())
                else:
                    encoder.transform(X[cat_features].copy())
            except:
                raise Exception(
                    f"Categorical features are not compatible with the fitted OrdinalEncoder."
                )
        for col in missing_cols:
            del X[col]
        return X

    def save_data(self, path: str):
        """
        Save the tabular data processed by :meth:`set_data`. Two files will be saved: ``data.csv`` contains all
        information from the input dataframe, and ``tabular_data.csv`` contains merely used features.

        Parameters
        ----------
        path
            The path to save the data.
        """
        self.categories_inverse_transform(self.df).to_csv(
            os.path.join(path, "data.csv"), encoding="utf-8", index=False
        )
        tabular_data, _, cat_feature_names, _ = self.get_tabular_dataset()
        tabular_data_inv = (
            self.categories_inverse_transform(tabular_data)
            if len(cat_feature_names) > 0
            else tabular_data
        )
        tabular_data_inv.to_csv(
            os.path.join(path, "tabular_data.csv"), encoding="utf-8", index=False
        )

        print(f"Data saved to {path} (data.csv and tabular_data.csv).")

    def derive(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, np.ndarray]]:
        """
        Derive both stacked and unstacked features using the input dataframe.

        Parameters
        ----------
        df
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The tabular dataset with derived stacked features.
        List
            Continuous feature names with derived stacked features.
        dict
            The derived unstacked data.

        See Also
        --------
        :meth:`.derive_stacked`, :meth:`.derive_unstacked`.
        """
        df_tmp, cont_feature_names = self.derive_stacked(df)
        derived_data = self.derive_unstacked(df_tmp)

        return df_tmp, cont_feature_names, derived_data

    def derive_stacked(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Derive stacked features using the input dataframe. Calculated using data derivers whose argument "stacked" is
        set to True.

        Parameters
        ----------
        df
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The tabular dataset with derived stacked features.
        List
            Continuous feature names with derived stacked features.
        """
        df_tmp = df.copy()
        cont_feature_names = cp(self.cont_feature_names)
        for deriver in self.dataderivers:
            if deriver.kwargs["stacked"]:
                value, name, col_names = deriver.derive(df_tmp, datamodule=self)
                if not deriver.kwargs["intermediate"]:
                    for col_name in col_names:
                        if col_name not in cont_feature_names:
                            cont_feature_names.append(col_name)
                            if self.training:
                                self.cont_feature_names.append(col_name)
                df_tmp[col_names] = value
        return df_tmp, cont_feature_names

    def derive_unstacked(
        self, df: pd.DataFrame, categorical_only=False
    ) -> Dict[str, np.ndarray]:
        """
        Derive unstacked features using the input dataframe. Calculated using data derivers whose argument "stacked" is
        set to False. Categorical features will be added to the returned dict with the key "categorical". Indices
        stating which data points are augmented will be added to the returned dict with the key
        "augmented" (1 for augmented ones).

        Parameters
        ----------
        df
            The tabular dataset.
        categorical_only
            Whether to only return categorical features.

        Returns
        -------
        dict
            The derived unstacked data.
        """
        derived_data = {}
        if not categorical_only:
            for deriver in self.dataderivers:
                if not deriver.kwargs["stacked"]:
                    value, name, _ = deriver.derive(df, datamodule=self)
                    derived_data[name] = value
        if len(self.cat_feature_names) > 0:
            derived_data["categorical"] = self.categories_transform(
                df[self.cat_feature_names]
            ).values
        if len(self.augmented_indices) > 0:
            augmented = np.zeros((len(df), 1))
            if self.training:
                augmented[self.augmented_indices - len(self.dropped_indices), 0] = 1
            derived_data["augmented"] = augmented
        return derived_data

    def _data_process(
        self,
        warm_start: bool = False,
        skip_selector: bool = False,
        verbose: bool = True,
    ):
        """
        The main procedure to process data after splitting and imputation. Both scaled and unscaled data will be recorded.
        Note that processors will fit on training and validation datasets and transform the testing set by calling
        :meth:`_data_preprocess` with different arguments.

        Parameters
        ----------
        warm_start
            Whether to use fitted data processors to process the data.
        skip_selector
            True to skip feature selections.
        verbose
            Verbosity.
        """
        self.df.reset_index(drop=True, inplace=True)
        self.scaled_df = self.df.copy()
        original_length = len(self.df)

        with HiddenPrints(disable_std=not verbose):
            if len(self.train_indices) == len(self.val_indices) and np.all(
                np.sort(self.train_indices) == np.sort(self.val_indices)
            ):
                df_training = self.df.loc[list(self.train_indices), :]
            else:
                df_training = self.df.loc[
                    list(self.train_indices) + list(self.val_indices), :
                ]
            unscaled_training_data = self._data_preprocess(
                df_training,
                warm_start=warm_start,
                skip_scaler=True,
                skip_selector=skip_selector,
            )
            training_data = self._data_preprocess(
                unscaled_training_data,
                warm_start=warm_start,
                scaler_only=True,
                skip_selector=skip_selector,
            )
            unscaled_testing_data = self.data_transform(
                self.df.loc[self.test_indices, :], skip_scaler=True
            )
            testing_data = self.data_transform(unscaled_testing_data, scaler_only=True)

        all_indices = np.unique(
            np.sort(np.array(list(training_data.index) + list(testing_data.index)))
        )
        self.retained_indices = np.intersect1d(all_indices, np.arange(original_length))
        self.dropped_indices = np.setdiff1d(
            np.arange(original_length), self.retained_indices
        )
        self.augmented_indices = np.setdiff1d(
            training_data.index, np.arange(original_length)
        )
        if len(np.setdiff1d(testing_data.index, np.arange(original_length))) > 0:
            raise Exception(f"Testing data should not be augmented.")

        def process_df(df, training, testing):
            inplace_training_index = np.intersect1d(
                training.index, self.retained_indices
            )
            df.loc[inplace_training_index, training.columns] = training.loc[
                inplace_training_index, :
            ].values
            df.loc[testing.index, testing.columns] = testing.values
            df = df.loc[self.retained_indices, :].copy().reset_index(drop=True)
            if len(self.augmented_indices) > 0:
                df = pd.concat(
                    [df, training.loc[self.augmented_indices, :]], axis=0
                ).reset_index(drop=True)
            df[self.cat_feature_names] = df[self.cat_feature_names].astype(int)
            return df

        self.df = process_df(self.df, unscaled_training_data, unscaled_testing_data)
        self.scaled_df = process_df(self.scaled_df, training_data, testing_data)

    @property
    def unscaled_feature_data(self) -> pd.DataFrame:
        """
        The unscaled feature data.
        """
        return self.df[self.cont_feature_names].copy()

    @property
    def unscaled_label_data(self) -> pd.DataFrame:
        """
        The unscaled label data.
        """
        return self.df[self.label_name].copy()

    @property
    def categorical_data(self) -> pd.DataFrame:
        """
        The categorical data.
        """
        return self.df[self.cat_feature_names].copy()

    @property
    def feature_data(self) -> pd.DataFrame:
        """
        The scaled feature data.
        """
        return self.scaled_df[self.cont_feature_names].copy()

    @property
    def label_data(self) -> pd.DataFrame:
        """
        The scaled label data.
        """
        return self.scaled_df[self.label_name].copy()

    def dataset_dict(self):
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "D_train": self.D_train,
            "D_val": self.D_val,
            "D_test": self.D_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
        }

    def __getitem__(self, item):
        return self.dataset_dict()[item]

    @property
    def X_train(self):
        """
        The unscaled training dataset.
        """
        return self.df.loc[self.train_indices, :].copy()

    @property
    def X_val(self):
        """
        The unscaled validation dataset.
        """
        return self.df.loc[self.val_indices, :].copy()

    @property
    def X_test(self):
        """
        The unscaled testing dataset.
        """
        return self.df.loc[self.test_indices, :].copy()

    @property
    def y_train(self):
        """
        The target of the training dataset.
        """
        return self.df.loc[self.train_indices, self.label_name].values

    @property
    def y_val(self):
        """
        The target of the validation dataset.
        """
        return self.df.loc[self.val_indices, self.label_name].values

    @property
    def y_test(self):
        """
        The target of the testing dataset.
        """
        return self.df.loc[self.test_indices, self.label_name].values

    @property
    def D_train(self):
        """
        The derived unstacked data of the training dataset.
        """
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.train_indices
        )

    @property
    def D_val(self):
        """
        The derived unstacked data of the validation dataset.
        """
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.val_indices
        )

    @property
    def D_test(self):
        """
        The derived unstacked data of the testing dataset.
        """
        return self.get_derived_data_slice(
            derived_data=self.derived_data, indices=self.test_indices
        )

    def _data_preprocess(
        self,
        input_data: pd.DataFrame,
        warm_start: bool = False,
        skip_scaler: bool = False,
        skip_selector: bool = False,
        scaler_only: bool = False,
    ) -> pd.DataFrame:
        """
        Call data processors to fit and/or transform the input tabular dataset. It is automatically called by
        :meth:`_data_process` and :meth:`data_transform` with different arguments.

        Parameters
        ----------
        input_data
            The tabular dataset.
        warm_start
            False to fit and transform data processors, and True to transform only.
        skip_scaler
            True to skip scaling (the last processor).
        skip_selector
            True to skip feature selections.
        scaler_only
            True to only perform scaling (the last processor).

        Returns
        -------
        pd.DataFrame
            The processed data.
        """
        from .base import AbstractScaler, AbstractFeatureSelector

        if skip_scaler and scaler_only:
            raise Exception(f"Both skip_scaler and scaler_only are True.")
        data = input_data.copy()
        for processor in self.dataprocessors:
            if skip_scaler and isinstance(processor, AbstractScaler):
                continue
            if skip_selector and isinstance(processor, AbstractFeatureSelector):
                continue
            if scaler_only and not isinstance(processor, AbstractScaler):
                continue
            if warm_start:
                data = processor.transform(data, self)
            else:
                data = processor.fit_transform(data, self)
        return data

    def data_transform(
        self,
        input_data: pd.DataFrame,
        skip_scaler: bool = False,
        scaler_only: bool = False,
    ):
        """
        Transform the input tabular dataset using fitted data processors.

        Parameters
        ----------
        input_data
            The tabular dataset.
        skip_scaler
            True to skip scaling (the last processor).
        scaler_only
            True to only perform scaling (the last processor).

        Returns
        -------
        pd.DataFrame
            The transformed tabular dataset.
        """
        return self._data_preprocess(
            input_data.copy(),
            warm_start=True,
            skip_scaler=skip_scaler,
            scaler_only=scaler_only,
        )

    def update_dataset(self):
        """
        Update PyTorch tensors and datasets. This is called after features change.
        """
        X, D, y = self.generate_tensors(self.scaled_df, self.derived_data)
        dataset = Data.TensorDataset(X, *D, y)
        self.train_dataset, self.val_dataset, self.test_dataset = self.generate_subset(
            dataset
        )
        self.tensors = (X, *D, y)

    def generate_tensors(self, scaled_df, derived_data):
        """
        Generate PyTorch tensors.

        Parameters
        ----------
        scaled_df
            The scaled dataset.
        derived_data
            A dict of derived unstacked data calculated by :meth:`derive_unstacked`

        Returns
        -------
        torch.Tensor
            A tensor of continuous features, a list of tensors of derived_unstacked data, and a tensor of the target.
        """
        X = torch.tensor(
            scaled_df[self.cont_feature_names].values.astype(np.float64),
            dtype=torch.float32,
        )
        D = [torch.tensor(value.astype(np.float64)) for value in derived_data.values()]
        y = torch.tensor(
            scaled_df[self.label_name].values.astype(np.float64),
            dtype=torch.float32,
        )
        return X, D, y

    def generate_subset(
        self, dataset: Data.Dataset
    ) -> Tuple[Data.Subset, Data.Subset, Data.Subset]:
        """
        Split the dataset into training, validation and testing subsets.

        Parameters
        ----------
        dataset
            A ``torch.utils.data.Dataset`` instance.

        Returns
        -------
        torch.utils.data.Subset
            Training, validation and testing subsets.
        """
        return (
            Subset(dataset, self.train_indices),
            Subset(dataset, self.val_indices),
            Subset(dataset, self.test_indices),
        )

    def get_derived_data_slice(
        self, derived_data: Dict[str, np.ndarray], indices: Iterable
    ) -> Dict[str, np.ndarray]:
        """
        Get slices of the derived unstacked data.

        Parameters
        ----------
        derived_data
            A dict of derived unstacked data calculated by :meth:`derive_unstacked``
        indices
            The indices to make slice.

        Returns
        -------
        dict
            The sliced derived unstacked data.
        """
        tmp_derived_data = {}
        for key, value in derived_data.items():
            tmp_derived_data[key] = value[indices, :]
        return tmp_derived_data

    def get_zero_slip(self, feature_name: str) -> float:
        """
        See how data processors act on a feature if its value is zero.
        It is a wrapper for :meth:`get_var_change`.

        Parameters
        ----------
        feature_name
            The investigated feature.

        Returns
        -------
        float
            The transformed value for the feature using data processors.
        """
        return self.get_var_change(feature_name=feature_name, value=0)

    def get_var_change(self, feature_name: str, value: float) -> float:
        """
        See how data processors act on a feature if its value is ``value``.

        Parameters
        ----------
        feature_name
            The investigated feature.
        value
            The investigated value.

        Returns
        -------
        value
            The transformed value for the feature using data processors.
        """
        from .dataprocessor import AbstractTransformer

        if not hasattr(self, "dataprocessors"):
            raise Exception(f"Run load_config first.")
        elif len(self.dataprocessors) == 0 and feature_name in self.cont_feature_names:
            return 0
        if feature_name not in self.dataprocessors[-1].record_cont_features:
            raise Exception(f"Feature {feature_name} not available.")

        x = value
        for processor in self.dataprocessors:
            if isinstance(processor, AbstractTransformer) and hasattr(
                processor, "transformer"
            ):
                x = processor.var_slip(feature_name, x)
        return x

    def describe(self, transformed=False) -> pd.DataFrame:
        """
        Describe the dataset using ``pd.DataFrame.describe``, skewness, gini index, mode values, etc.

        Parameters
        ----------
        transformed
            Whether to describe the scaled data.

        Returns
        -------
        pd.DataFrame
            The descriptions of the dataset.
        """
        tabular = self.get_tabular_dataset(transformed=transformed)[0]
        desc = tabular.describe()

        skew = tabular.skew()
        desc = pd.concat(
            [
                desc,
                pd.DataFrame(
                    data=skew.values.reshape(len(skew), 1).T,
                    columns=skew.index,
                    index=["Skewness"],
                ),
            ],
            axis=0,
        )

        g = self._get_gini(tabular)
        desc = pd.concat([desc, g], axis=0)

        mode, cnt_mode, mode_percent = self._get_mode(tabular)
        desc = pd.concat([desc, mode, cnt_mode, mode_percent], axis=0)

        kurtosis = self._get_kurtosis(tabular)
        kurtosis[self.cat_feature_names] = np.nan
        desc = pd.concat([desc, kurtosis], axis=0)

        return desc

    def get_derived_data_sizes(self) -> List[Tuple]:
        """
        Get dimensions of derived unstacked features.

        Returns
        -------
        sizes
            A list of np.ndarray.shape representing dimensions of each derived unstacked feature.
        """
        return [x.shape for x in self.derived_data.values()]

    @staticmethod
    def _get_gini(tabular: pd.DataFrame) -> pd.DataFrame:
        """
        Get the gini index for each feature in the tabular dataset.

        Parameters
        ----------
        tabular
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The gini index for each feature in the dataset.
        """
        return pd.DataFrame(
            data=np.array([[gini(tabular[x]) for x in tabular.columns]]),
            columns=tabular.columns,
            index=["Gini Index"],
        )

    @staticmethod
    def _get_kurtosis(tabular: pd.DataFrame) -> pd.DataFrame:
        """
        Get the kurtosis for each feature in the tabular dataset.

        Parameters
        ----------
        tabular
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The kurtosis for each feature in the dataset.
        """
        return pd.DataFrame(
            data=st.kurtosis(tabular.values, axis=0).reshape(1, -1),
            columns=tabular.columns,
            index=["Kurtosis"],
        )

    @staticmethod
    def _get_mode(
        tabular: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the mode value for each feature in the tabular dataset.

        Parameters
        ----------
        tabular
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The mode value for each feature in the dataset.
        pd.DataFrame
            The number of the mode value for each feature.
        pd.DataFrame
            The percentage of the mode value for each feature.
        """
        mode = tabular.mode().loc[0, :]
        cnt_mode = pd.DataFrame(
            data=np.array(
                [
                    [
                        tabular[mode.index[x]].value_counts()[mode.values[x]]
                        for x in range(len(mode))
                    ]
                ]
            ),
            columns=tabular.columns,
            index=["Mode counts"],
        )
        mode_percent = cnt_mode / tabular.count()
        mode_percent.index = ["Mode percentage"]

        mode = pd.DataFrame(
            data=mode.values.reshape(len(mode), 1).T, columns=mode.index, index=["Mode"]
        )
        return mode, cnt_mode, mode_percent

    def pca(
        self, feature_names: List[str] = None, feature_idx: List[int] = None, **kwargs
    ) -> PCA:
        """
        Perform ``sklearn.decomposition.PCA``

        Parameters
        ----------
        feature_names
            A list of names of continuous features.
        **kwargs
            Arguments of sklearn.decomposition.PCA.

        Returns
        -------
        sklearn.decomposition.PCA
            A ``sklearn.decomposition.PCA`` instance.
        """
        pca = PCA(**kwargs)
        if feature_names is not None:
            pca.fit(self.feature_data.loc[self.train_indices, feature_names])
        elif feature_idx is not None:
            pca.fit(
                self.feature_data.loc[
                    self.train_indices, np.array(self.cont_feature_names)[feature_idx]
                ]
            )
        else:
            pca.fit(self.feature_data.loc[self.train_indices, :])
        return pca

    def divide_from_tabular_dataset(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get continuous feature data, categorical feature data, and label data respectively from the input DataFrame.

        Parameters
        ----------
        data
            The tabular dataset.

        Returns
        -------
        pd.DataFrame
            The continuous feature data.
        pd.DataFrame
            The categorical feature data.
        pd.DataFrame
            The label data.
        """
        feature_data = data[self.cont_feature_names]
        categorical_data = data[self.cat_feature_names]
        label_data = data[self.label_name]

        return feature_data, categorical_data, label_data

    def get_tabular_dataset(
        self, transformed: bool = False
    ) -> Tuple[pd.DataFrame, List, List, List]:
        """
        Get the tabular dataset loaded in the DataModule.

        Parameters
        ----------
        transformed
            Whether to return the scaled data or not.

        Returns
        -------
        pd.DataFrame
            The tabular dataset.
        List
            The continuous feature names in the dataset.
        List
            The categorical feature names in the dataset.
        List
            The target names.
        """
        if transformed:
            feature_data = self.feature_data
        else:
            feature_data = self.unscaled_feature_data

        cont_feature_names = cp(self.cont_feature_names)
        cat_feature_names = cp(self.cat_feature_names)
        label_name = cp(self.label_name)

        tabular_dataset = pd.concat(
            [feature_data, self.categorical_data, self.label_data], axis=1
        )

        return tabular_dataset, cont_feature_names, cat_feature_names, label_name

    def cal_corr(
        self, imputed: bool = False, features_only: bool = False
    ) -> pd.DataFrame:
        """
        Calculate Pearson correlation coefficients among continuous features.

        Parameters
        ----------
        imputed
            Whether the imputed dataset should be considered. If False, some NaN values may exist for features that have
            missing values.
        features_only
            If False, the target is also considered.

        Returns
        -------
        pd.DataFrame
            The correlation dataframe.
        """
        subset = (
            self.cont_feature_names
            if features_only
            else self.cont_feature_names + self.label_name
        )
        if not imputed:
            not_imputed_df = self.get_not_imputed_df()
            return not_imputed_df[subset].corr()
        else:
            return self.df[subset].corr()

    def get_not_imputed_df(self) -> pd.DataFrame:
        """
        Get the tabular data without imputation.

        Returns
        -------
        pd.DataFrame
            The tabular dataset without imputation.
        """
        tmp_cont_df = self.df.copy()[self.cont_feature_names]
        if np.sum(np.abs(self.cont_imputed_mask.values)) != 0:
            tmp_cont_df[self.cont_imputed_mask == 1] = np.nan
        tmp_cat_df = self.categories_inverse_transform(self.df).loc[
            :, self.cat_feature_names
        ]
        if np.sum(np.abs(self.cat_imputed_mask.values)) != 0:
            tmp_cat_df[self.cat_imputed_mask == 1] = np.nan
        not_imputed_df = self.df.copy()
        not_imputed_df[self.all_feature_names] = pd.concat(
            [tmp_cont_df, tmp_cat_df], axis=1
        )
        return not_imputed_df

    def _get_indices(self, partition: str = "train") -> np.ndarray:
        """
        Get training/validation/testing indices.

        Parameters
        ----------
        partition
            "train", "val", "test", or "all"

        Returns
        -------
        np.ndarray
            The indices of the selected partition.
        """
        indices_map = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
            "all": np.array(self.feature_data.index),
        }

        if partition not in indices_map.keys():
            raise Exception(
                f"Partition {partition} not available. Select among {list(indices_map.keys())}"
            )

        return indices_map[partition]

    def get_additional_tensors_slice(self, indices) -> Union[Tuple[Any], Tuple]:
        """
        Get slices of tensors of derived unstacked data.

        Parameters
        ----------
        indices
            The indices to make the slice.

        Returns
        -------
        Tuple[torch.Tensor]
            Sliced derived unstacked tensors.
        """
        res = []
        for tensor in self.tensors[1 : len(self.tensors) - 1]:
            if tensor is not None:
                res.append(tensor[indices, :])
        return tuple(res)

    def get_first_tensor_slice(self, indices) -> torch.Tensor:
        """
        Get a slice of the tensor of continuous features.

        Parameters
        ----------
        indices
            The indices to make the slice.

        Returns
        -------
        torch.Tensor
            The sliced tensor of continuous feature.
        """
        return self.tensors[0][indices, :]

    def get_base_predictor(
        self,
        categorical: bool = True,
        **kwargs,
    ) -> Union[sklearn.pipeline.Pipeline, sklearn.ensemble.RandomForestRegressor]:
        """
        Get a sklearn ``RandomForestRegressor`` for fundamental usages like pre-processing.

        Parameters
        ----------
        categorical
            Whether to include ``OneHotEncoder`` for categorical features.
        kwargs
            Arguments for ``sklearn.ensemble.RandomForestRegressor``

        Returns
        -------
        sklearn.pipeline.Pipeline or sklearn.ensemble.RandomForestRegressor
            A Pipeline if categorical is True, or a RandomForestRegressor if categorical is False.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(**kwargs)

        if len(self.cat_feature_names) > 0 and categorical:
            categorical_encoder = OneHotEncoder()
            numerical_pipe = SimpleImputer(strategy="mean")
            preprocessing = ColumnTransformer(
                [
                    (
                        "cat",
                        categorical_encoder,
                        lambda x: [y for y in self.cat_feature_names if y in x.columns],
                    ),
                    (
                        "num",
                        numerical_pipe,
                        lambda x: [
                            y for y in self.cont_feature_names if y in x.columns
                        ],
                    ),
                ],
                verbose_feature_names_out=False,
            )

            pip = Pipeline(
                [
                    ("preprocess", preprocessing),
                    ("classifier", rf),
                ]
            )
            return pip
        else:
            return rf

    def select_by_value(
        self,
        selection: Dict[str, Union[str, int, float, List, Tuple]],
        df: pd.DataFrame = None,
        partition: str = None,
        eps: float = None,
    ) -> np.ndarray:
        """
        Select data points with the given value(s) in the given column(s).

        Parameters
        ----------
        selection
            A dictionary whose items indicate the columns to be investigated and the values (if is a list/int/float/str)
            or a range of values (closed interval, if is a tuple with two components) to be selected for each column.
        df
            A dataframe to be filtered. If not given, :attr:`df` is used.
        partition
            "train", "val", "test", or "all"
        eps
            A tolerance value if the value to be selected is a float. If None, only values "equal" to the float will be
            selected.

        Returns
        -------
        np.ndarray
            Indices of the selected data points in the dataframe.
        """
        if partition is not None and df is not None:
            raise Exception(f"Provide only one of `partition` and `df`.")
        if df is None:
            df = self.df
        if partition is not None:
            part = self._get_indices(partition)
        else:
            part = np.array(df.index)
        col_res_ls = []
        for col, val in selection.items():
            if isinstance(val, list):
                col_res = []
                for v in val:
                    col_res += list(df[df[col] == v].index)
            elif isinstance(val, tuple) and len(val) == 2:
                col_res = df[(df[col] >= val[0]) & (df[col] <= val[1])].index
            elif isinstance(val, int) or isinstance(val, float) or isinstance(val, str):
                if isinstance(val, float) and eps is not None:
                    col_res = np.array(df[((df[col] - val).__abs__() <= eps)].index)
                else:
                    col_res = df[df[col] == val].index
            else:
                raise Exception(f"Unrecognized selection of {col}: {val}.")
            col_res = np.sort(np.intersect1d(np.unique(col_res), part))
            col_res_ls.append(col_res)
        res = np.sort(reduce(np.intersect1d, col_res_ls))
        return res
