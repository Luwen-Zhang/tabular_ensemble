from tabensemb.utils import *
from copy import deepcopy as cp
from typing import *
from .datamodule import DataModule
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numbers


class AbstractDataStep:
    """
    By inheriting this class, the input kwargs will be used to update default values defined in
    :meth:`~AbstractDataStep._defaults`. The requirements defined in
    :meth:`~AbstractDataStep._cls_required_kwargs` and :meth:`~AbstractDataStep._required_kwargs` will be
    checked. The final kwargs will be stored as ``self.kwargs``.
    """

    def __init__(self, **kwargs):
        self.kwargs = self._defaults()
        self.kwargs.update(kwargs)
        for key in self._cls_required_kwargs():
            self._check_arg(key)
        for key in self._required_kwargs():
            self._check_arg(key)
        self.record_cont_features = None
        self.record_cat_features = None
        self.record_cat_dtypes = {}

    def _record_features(self, input_data: pd.DataFrame, datamodule: DataModule):
        self.record_cont_features = cp(datamodule.cont_feature_names)
        self.record_cat_features = cp(datamodule.cat_feature_names)
        self.record_cat_dtypes = {
            feature: type(input_data[feature].values[0])
            for feature in datamodule.cat_feature_names
        }

    def _restore_features(self, input_data: pd.DataFrame, datamodule: DataModule):
        if not getattr(datamodule, "_force_features", False):
            datamodule.cont_feature_names = cp(self.record_cont_features)
            datamodule.cat_feature_names = cp(self.record_cat_features)
        for feature, dtype in self.record_cat_dtypes.items():
            if feature in input_data.columns:
                input_data[feature] = input_data[feature].values.astype(dtype)
        return input_data

    def _defaults(self) -> Dict:
        """
        Defaults values for arguments defined in :meth:`~AbstractDataStep._cls_required_kwargs` and
        :meth:`~AbstractDataStep._required_kwargs`

        Returns
        -------
        dict
            A dictionary of default values.
        """
        return {}

    def _cls_required_kwargs(self) -> List:
        """
        kwargs required by the class.

        Returns
        -------
        List
            A list of names of arguments that should be defined either in :meth:`~AbstractDataStep._defaults` or
            in the configuration file.
        """
        return []

    def _required_kwargs(self) -> List:
        """
        kwargs required by the class. It is for a specific subclass rather than an abstract class.

        Returns
        -------
        List
            A list of names of arguments that should be defined either in :meth:`~AbstractDataStep._defaults` or
            in the configuration file.
        """
        return []

    def _check_arg(self, name: str):
        """
        Check whether the required parameter is specified in the configuration file or in
        :meth:`~AbstractDataStep._defaults`.

        Parameters
        ----------
        name
            The name of argument in the input arguments.
        """
        if name not in self.kwargs.keys():
            raise Exception(f"{self.__class__.__name__}: {name} should be specified.")


class AbstractDeriver(AbstractDataStep):
    def __init__(self, **kwargs):
        """
        The base class for all data-derivers, which will derive new features based on the input DataFrame and return
        the derived values, or load and return multimodal data. It is recommended to learn the derivation on the
        training set only.

        Parameters
        ----------
        kwargs
            Arguments required by the deriver. It will be stored as ``self.kwargs``.
        """
        super(AbstractDeriver, self).__init__(**kwargs)
        for arg_name in self._required_cols():
            self._check_arg(arg_name)
        self.last_derived_col_names = []
        self.derived_dtype = None

    def _cls_required_kwargs(self):
        """
        kwargs required by the class. "stacked", "intermediate", and "derived_name" are required for all data derivers.

        Returns
        -------
        List
            A list of names of arguments that should be defined either in :meth:`~AbstractDataStep._defaults` or
            in the configuration file.
        """
        return ["stacked", "intermediate", "derived_name"]

    def derive(
        self,
        df: pd.DataFrame,
        datamodule: DataModule,
    ) -> Tuple[np.ndarray, List]:
        """
        The method automatically checks input column names and the DataFrame, calls the :meth:`._derive` method, and
        checks the output of the derived data.

        Parameters
        ----------
        df:
            The tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance. Data-derivers might use information in the
            DataModule, but would not change its contents.

        Returns
        -------
        np.ndarray
            A ndarray of derived data
        List
            Names of each column in the derived data.
        """
        for arg_name in self._required_cols():
            self._check_exist(df, arg_name)
        values = self._derive(df, datamodule)
        self._check_values(values)
        if self.derived_dtype is None:
            self.derived_dtype = values.dtype
        else:
            values = values.astype(self.derived_dtype)
        names = (
            self._generate_col_names(values.shape[-1])
            if "col_names" not in self.kwargs
            else self.kwargs["col_names"]
        )
        self.last_derived_col_names = names
        return values, names

    def _derive(
        self,
        df: pd.DataFrame,
        datamodule: DataModule,
    ) -> np.ndarray:
        """
        The main function for a data-deriver.

        Parameters
        ----------
        df:
            The tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance. Data-derivers might use information in the DataModule, but
            would not change its contents.

        Returns
        -------
        np.ndarray
            The derived data. If it is one-dimensional, use reshape(-1, 1) to transform it into two-dimensional.
        """
        raise NotImplementedError

    def _derived_names(self) -> List[str]:
        """
        Default names for each column of the derived data.

        Returns
        -------
        List
            A list of column names of the derived data.
        """
        raise NotImplementedError

    def _generate_col_names(self, length: int) -> List[str]:
        """
        Use the ``derived_name`` argument to generate column names for each column of the derived data.

        Parameters
        ----------
        length:
            The number of columns of the derived data.

        Returns
        -------
        List
            Automatically generated column names.
        """
        try:
            names = self._derived_names()
        except:
            derived_name = self.kwargs["derived_name"]
            names = (
                [f"{derived_name}-{idx}" for idx in range(length)]
                if length > 1
                else [derived_name]
            )
        return names

    def _required_cols(self) -> List[str]:
        """
        Required column names in the tabular dataset by the data-deriver. Whether these names exist in the tabular
        dataset will be checked in :meth:`~AbstractDeriver.derive`.

        Returns
        -------
        List
            Column names required by the data-deriver in the tabular dataset.
        """
        raise NotImplementedError

    def _check_exist(self, df: pd.DataFrame, name: str):
        """
        Check whether the required column name exists in the tabular dataset.

        Parameters
        ----------
        df:
            The tabular dataset.
        name:
            The name of argument or a column name in the input arguments.
        """
        if self.kwargs[name] not in df.columns:
            raise Exception(
                f"Derivation: {name} is not a valid column in df for deriver {self.__class__.__name__}."
            )

    def _check_values(self, values: np.ndarray) -> None:
        """
        Check whether the returned derived data is two-dimensional.

        Parameters
        ----------
        values:
            The derived data returned by :meth:`~AbstractDeriver._derive`.
        """
        if len(values.shape) == 1:
            raise Exception(
                f"Derivation: {self.__class__.__name__} returns a one dimensional numpy.ndarray. Use reshape(-1, 1) to "
                f"transform into 2D."
            )


class AbstractImputer(AbstractDataStep):
    """
    The base class for all data-imputers. Data-imputers replace NaNs in the input tabular dataset.
    """

    def __init__(self, **kwargs):
        super(AbstractImputer, self).__init__(**kwargs)
        self.record_imputed_features = None

    def fit_transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Record feature names in the datamodule, fit the imputer and transform the input dataframe. This should be
        performed on the training and validation sets. Missing values in categorical features are filled by "UNK".
        Continuous features that are totally missing will not be imputed.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by imputers.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        data = input_data.copy()
        self._record_features(input_data=input_data, datamodule=datamodule)
        for feature, dtype in self.record_cat_dtypes.items():
            data[feature] = data[feature].fillna(
                -1 if issubclass(dtype, numbers.Number) else "UNK"
            )
        return (
            self._fit_transform(data, datamodule)
            if len(self.record_cont_features) > 0
            else data
        )

    def transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Restore feature names in the datamodule using recorded features, and transform the input tabular data using the fitted
        imputer. This should be performed on the testing set.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by imputers.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        data = input_data.copy()
        data = self._restore_features(input_data=data, datamodule=datamodule)
        for feature, dtype in self.record_cat_dtypes.items():
            data[feature] = data[feature].fillna(
                -1 if issubclass(dtype, numbers.Number) else "UNK"
            )
        return (
            self._transform(data, datamodule)
            if len(self.record_cont_features) > 0
            else data
        )

    def _fit_transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Fit the imputer and transform the input dataframe. This should be performed on the training and validation sets.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by imputers.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        raise NotImplementedError

    def _transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Transform the input tabular data using the fitted imputer. This should perform on the testing dataset.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by imputers.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        raise NotImplementedError

    def _get_impute_features(
        self, cont_feature_names: List[str], data: pd.DataFrame
    ) -> List[str]:
        """
        Get continuous feature names that can be imputed, i.e. those not totally missing.

        Parameters
        ----------
        cont_feature_names:
            Names of continuous features.
        data:
            The input tabular dataset.

        Returns
        -------
        List
            Names of continuous features that can be imputed.
        """
        all_missing_idx = np.where(
            np.isnan(data[cont_feature_names].values).all(axis=0)
        )[0]
        impute_features = [
            x for idx, x in enumerate(cont_feature_names) if idx not in all_missing_idx
        ]
        self.record_imputed_features = impute_features
        return impute_features


class AbstractSklearnImputer(AbstractImputer):
    """
    A base class for sklearn-style imputers that has ``fit_transform`` and ``transform`` methods that return
    ``np.ndarray``.
    """

    def _fit_transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        impute_features = self._get_impute_features(
            datamodule.cont_feature_names, input_data
        )
        imputer = self._new_imputer()
        # https://github.com/scikit-learn/scikit-learn/issues/16426
        # SimpleImputer reduces the number of features without giving messages. The issue is fixed in
        # scikit-learn==1.2.0 by an argument "keep_empty_features"; however, autogluon==0.6.1 requires
        # scikit-learn<1.2.0.
        res = imputer.fit_transform(input_data[impute_features])
        if type(res) == pd.DataFrame:
            res = res.values
        input_data[impute_features] = res.astype(np.float64)

        self.transformer = imputer
        return input_data

    def _transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        res = self.transformer.transform(input_data[self.record_imputed_features])
        if type(res) == pd.DataFrame:
            res = res.values
        input_data[self.record_imputed_features] = res.astype(np.float64)
        return input_data

    def _new_imputer(self):
        """
        Get a sklearn-style imputer that has ``fit_transform`` and ``transform`` methods that return
        ``np.ndarray``.

        Returns
        -------
        Any
            A instance of a sklearn-style imputer, such as ``sklearn.impute.SimpleImputer``.
        """
        raise NotImplementedError


class AbstractProcessor(AbstractDataStep):
    """
    The base class for data-processors that change the content of the tabular dataset. The class is only directly used
    for those who reduce the number of data points.

    Notes
    -----
    If any attribute of the datamodule is set by the processor in :meth:`~AbstractProcessor._fit_transform`, the
    processor is responsible for restoring the set attribute when _transform is called. For instance, in the wrapper
    methods :meth:`~AbstractProcessor.fit_transform` and :meth:`~AbstractProcessor.transform`, we have implemented
    recording feature names and restoring them.
    """

    def fit_transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Record feature names in the datamodule, fit the processor, and call :meth:`~AbstractProcessor._fit_transform` to
        transform the input data. This should be performed on the training set.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by
            processors.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        data = input_data.copy()
        self._record_features(input_data, datamodule)
        return self._fit_transform(data, datamodule)

    def transform(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Restore feature names in datamodule using recorded features and call :meth:`~AbstractProcessor._transform`
        to transform the input data. This should be performed on the validation and testing sets.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by
            processors.

        Returns
        -------
        pd.DataFrame
            A transformed tabular dataset.
        """
        data = input_data.copy()
        data = self._restore_features(input_data=data, datamodule=datamodule)
        return self._transform(data, datamodule)

    def _fit_transform(
        self, data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _transform(self, data: pd.DataFrame, datamodule: DataModule) -> pd.DataFrame:
        raise NotImplementedError


class AbstractAugmenter(AbstractProcessor):
    """
    A kind of data processor that increases the number of data points.
    """

    def _fit_transform(
        self, data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        already_augmented = max([np.max(data.index) - len(datamodule.df) + 1, 0])
        ###############################
        # Here is the augmentation part
        augmented = self._get_augmented(data, datamodule)
        ###############################
        augmented.reset_index(drop=True, inplace=True)
        augmented.index = (
            np.array(augmented.index) + len(datamodule.df) + already_augmented
        )
        data = pd.concat([data, augmented], axis=0)
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule) -> pd.DataFrame:
        # Do not do anything to the testing data.
        return data

    def _get_augmented(
        self, data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        """
        Return a DataFrame that contains augmented (new) data points based on the training and validation sets.

        Parameters
        ----------
        data
            The combined training and validation sets.
        datamodule
            A :class:`~tabensemb.data.datamodule.DataModule` instance.

        Returns
        -------
        pd.DataFrame
            Augmented data points.
        """
        raise NotImplementedError


class AbstractTransformer(AbstractProcessor):
    """
    The base class for data-processors that change values of the tabular dataset.
    """

    def __init__(self, **kwargs):
        super(AbstractTransformer, self).__init__(**kwargs)
        self.transformer = None

    def var_slip(self, feature_name, x) -> Union[int, float, Any]:
        """
        See how the transformer performs on a value of a feature.

        Parameters
        ----------
        feature_name:
            The investigated feature.
        x:
            The value of the feature.

        Returns
        -------
        int or float
            The transformed value of the feature value x.
        """
        zero_data = pd.DataFrame(
            data=np.array(
                [
                    0 if feature_name != record_feature else x
                    for record_feature in self.record_cont_features
                ]
            ).reshape(1, -1),
            columns=self.record_cont_features,
        )
        try:
            trans_res = self.transformer.transform(zero_data)
        except:
            trans_res = zero_data.values
        return trans_res[0, self.record_cont_features.index(feature_name)]


class AbstractFeatureSelector(AbstractProcessor):
    """
    The base class for data-processors that reduce the number of features.
    """

    def _fit_transform(
        self, data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        if len(datamodule.all_feature_names) > 0:
            retain_features = list(self._get_feature_names_out(data, datamodule))
            removed_features = list(
                np.setdiff1d(datamodule.all_feature_names, retain_features)
            )
            if len(removed_features) > 0:
                datamodule.cont_feature_names = [
                    x for x in datamodule.cont_feature_names if x in retain_features
                ]
                datamodule.cat_feature_names = [
                    x for x in datamodule.cat_feature_names if x in retain_features
                ]
                print(
                    f"{len(removed_features)} features removed: {removed_features}. {len(retain_features)} features "
                    f"retained: {retain_features}."
                )
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule) -> pd.DataFrame:
        return data

    def _get_feature_names_out(
        self, input_data: pd.DataFrame, datamodule: DataModule
    ) -> List[str]:
        """
        Get selected features.

        Parameters
        ----------
        input_data:
            A tabular dataset.
        datamodule:
            A :class:`~tabensemb.data.datamodule.DataModule` instance that contains necessary information required by
            processors.

        Returns
        -------
        List
            A list of selected features.
        """
        raise NotImplementedError


class AbstractScaler(AbstractTransformer):
    """
    This is a marker for scaling processors like a standard scaler or a normalizer.
    """

    pass


class AbstractSplitter:
    """
    The base class for data-splitters that split the dataset and return training, validation, and testing indices.

    Attributes
    ----------
    support_cv
    """

    def __init__(
        self,
        train_val_test: Optional[Union[List, np.ndarray]] = None,
        cv: int = -1,
    ):
        self.train_val_test = (
            np.array([0.6, 0.2, 0.2])
            if train_val_test is None
            else np.array(train_val_test)
        )
        self.train_val_test /= np.sum(self.train_val_test)
        self.cv_generator = None
        self.cv = cv

    def split(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        cv: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset. It will call :meth:`~AbstractSplitter._split` and check its results.

        Parameters
        ----------
        df:
            The input tabular dataset.
        cont_feature_names:
            Names of continuous features.
        cat_feature_names:
            Names of categorical features.
        label_name:
            The name of the label.
        cv:
            The total number of cross-validation runs.

        Returns
        -------
        np.ndarray
            Indices of the training, validation, and testing datasets.
        """
        cv = self.cv if cv is None or cv <= 1 else cv
        if cv > 1 and self.support_cv:
            train_indices, val_indices, test_indices = self._next_cv(
                df, cont_feature_names, cat_feature_names, label_name, cv
            )
        else:
            if cv > 1:
                warnings.warn(
                    f"{self.__class__.__name__} does not support cross validation splitting."
                )
            train_indices, val_indices, test_indices = self._split(
                df, cont_feature_names, cat_feature_names, label_name
            )
        self._check_split(train_indices, val_indices, test_indices)
        return train_indices, val_indices, test_indices

    def reset_cv(self, cv: int = -1):
        self.cv_generator = None
        self.cv = cv

    def _split(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def support_cv(self):
        """
        Whether the :meth:`~AbstractSplitter._next_cv` is implemented.
        """
        return False

    def _next_cv(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        cv: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the next fold of indices of training, validation, and testing sets.

        Parameters
        ----------
        df:
            The input tabular dataset.
        cont_feature_names:
            Names of continuous features.
        cat_feature_names:
            Names of categorical features.
        label_name:
            The name of the label.
        cv:
            The total number of cross-validation runs.

        Returns
        -------
        np.ndarray
            Indices of the training, validation, and testing dataset.
        """
        raise NotImplementedError

    def _sklearn_k_fold(self, data, cv) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a ``sklearn.model_selection.KFold`` instance and return its ``__next__()`` result.

        Notes
        -----
        The returned values are fold indices of the input ``data`` argument, but not the fold.

        Parameters
        ----------
        data
            An Iterable whose index will be split by KFold.
        cv
            The total number of cross-validation runs.

        Returns
        -------
        np.ndarray
            Indices of the training, validation, and testing datasets of the current fold.
        """
        if self.cv_generator is None or cv != self.cv:
            if cv != self.cv and self.cv > 1:
                warnings.warn(
                    f"The input {cv}-fold is not consistent with the previous setting {self.cv}-fold. "
                    f"Starting a new {cv}-fold generator."
                )
            self.cv_generator = PickleAbleGenerator(
                KFold(n_splits=cv, shuffle=True).split(data)
            )
            self.cv = cv
        try:
            train_indices, test_indices = self.cv_generator.__next__()
        except:
            warnings.warn(f"{cv}-fold exceeded. Starting a new {cv}-fold generator.")
            self.cv_generator = PickleAbleGenerator(
                KFold(n_splits=cv, shuffle=True).split(data)
            )
            train_indices, test_indices = self.cv_generator.__next__()
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=len(data) // cv if cv > 2 else len(train_indices) // 2,
            shuffle=True,
        )
        return train_indices, val_indices, test_indices

    @staticmethod
    def _check_split(train_indices, val_indices, test_indices):
        """
        Check whether split indices overlap with each other.

        Parameters
        ----------
        train_indices:
            Indices of the training dataset.
        val_indices:
            Indices of the validation dataset.
        test_indices
            Indices of the testing dataset.
        """

        def individual_check(indices, name):
            if not issubclass(type(indices), np.ndarray):
                raise Exception(
                    f"The class of {name}_indices {type(indices)} is not the subclass of numpy.ndarray."
                )
            if len(indices.shape) != 1:
                raise Exception(
                    f"{name}_indices is not one dimensional. Use numpy.ndarray.flatten() to convert."
                )

        def intersect_check(a_indices, b_indices, a_name, b_name):
            if len(np.intersect1d(a_indices, b_indices)) != 0:
                raise Exception(
                    f"There exists intersection {np.intersect1d(a_indices, b_indices)} between {a_name}_indices "
                    f"and {b_name}_indices."
                )

        individual_check(train_indices, "train")
        individual_check(val_indices, "val")
        individual_check(test_indices, "test")

        intersect_check(train_indices, val_indices, "train", "val")
        intersect_check(train_indices, test_indices, "train", "test")
        intersect_check(val_indices, test_indices, "val", "test")

    def _check_exist(self, df, arg, name):
        if arg not in df.columns:
            raise Exception(
                f"Splitter: {name} is not a valid column in df for splitter {self.__class__.__name__}."
            )
