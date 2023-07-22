import warnings
from tabensemb.utils import *
from tabensemb.data import (
    AbstractProcessor,
    AbstractFeatureSelector,
    AbstractTransformer,
    AbstractScaler,
    AbstractAugmenter,
)
from tabensemb.data import DataModule
import inspect
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.preprocessing import OrdinalEncoder
from typing import Type
from .utils import get_corr_sets


class SampleDataAugmentor(AbstractAugmenter):
    """
    This is a sample of data augmentation, which is not reasonable at all and is only used to test data augmentation.
    """

    def _get_augmented(
        self, data: pd.DataFrame, datamodule: DataModule, **kwargs
    ) -> pd.DataFrame:
        augmented = data.loc[data.index[-2:], :].copy()
        return augmented


class FeatureValueSelector(AbstractProcessor):
    """
    Select data with the specified feature value. Required arguments:

    feature: str
        The feature that will be filtered.
    value: float
        The specified feature value.
    """

    def __init__(self):
        super(FeatureValueSelector, self).__init__()

    def _fit_transform(
        self,
        data: pd.DataFrame,
        datamodule: DataModule,
        feature=None,
        value=None,
        **kwargs,
    ):
        if feature is None or value is None:
            raise Exception(
                'FeatureValueSelector requires arguments "feature" and "value".'
            )
        if value not in list(data[feature]):
            raise Exception(
                f"Value {value} not available for feature {feature}. Select from {data[feature].unique()}"
            )
        where_value = data.index[np.where(data[feature] == value)[0]]
        data = data.loc[where_value, :]
        self.feature, self.value = feature, value
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        if datamodule.training:
            if self.value not in list(data[self.feature]):
                raise Exception(
                    f"Value {self.value} not available for feature {self.feature}. Select from {data[self.feature].unique()}"
                )
            where_value = data.index[np.where(data[self.feature] == self.value)[0]]
            data = data.loc[where_value, :]
        else:
            if self.value not in list(data[self.feature]):
                warnings.warn(
                    f"Value {self.value} not available for feature {self.feature} selected by "
                    f"{self.__class__.__name__}."
                )
        return data


class IQRRemover(AbstractProcessor):
    """
    Remove outliers using IQR strategy. Outliers are those
    out of the range [25-percentile - 1.5 * IQR, 75-percentile + 1.5 * IQR], where IQR = 75-percentile - 25-percentile.
    """

    def __init__(self):
        super(IQRRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        print(f"Removing outliers by IQR. Original size: {len(data)}, ", end="")
        for feature in list(datamodule.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            Q1 = np.percentile(data[feature].dropna(axis=0), 25, method="midpoint")
            Q3 = np.percentile(data[feature].dropna(axis=0), 75, method="midpoint")
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = data.index[np.where(data[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = data.index[np.where(data[feature] <= (Q1 - 1.5 * IQR))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        return data


class StdRemover(AbstractProcessor):
    """
    Remove outliers using standard error strategy. Outliers are those out of the range of 3sigma.
    """

    def __init__(self):
        super(StdRemover, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        print(f"Removing outliers by std. Original size: {len(data)}, ", end="")
        for feature in list(datamodule.args["feature_names_type"].keys()):
            if pd.isna(data[feature]).all():
                raise Exception(f"All values of {feature} are NaN.")
            m = np.mean(data[feature].dropna(axis=0))
            std = np.std(data[feature].dropna(axis=0))
            if std == 0:
                continue
            upper = data.index[np.where(data[feature] >= (m + 3 * std))[0]]
            lower = data.index[np.where(data[feature] <= (m - 3 * std))[0]]

            data = data.drop(upper)
            data = data.drop(lower)
        print(f"Final size: {len(data)}.")
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        return data


class NaNFeatureRemover(AbstractFeatureSelector):
    """
    Remove features that contain no valid value.
    """

    def __init__(self):
        super(NaNFeatureRemover, self).__init__()

    def _get_feature_names_out(self, data, datamodule, **kwargs):
        retain_features = []
        all_missing_idx = np.where(
            pd.isna(data[datamodule.all_feature_names]).values.all(axis=0)
        )[0]
        for idx, feature in enumerate(datamodule.all_feature_names):
            if idx not in all_missing_idx:
                retain_features.append(feature)
        return retain_features


class RFEFeatureSelector(AbstractFeatureSelector):
    """
    Select features using recursive feature elimination, adapted from the implementation of RFECV in sklearn.
    Available arguments:

    n_estimators: int
        The number of trees used in random forests.
    step: int
        The number of eliminated features at each step.
    min_features_to_select: int
        The minimum number of features.
    method: str
        The method of calculating importance. "auto" for default impurity-based method implemented in
        RandomForestRegressor, and "shap" for SHAP value (which may slow down the program but is more accurate).
    """

    def __init__(self):
        super(RFEFeatureSelector, self).__init__()

    def _get_feature_names_out(
        self,
        data,
        datamodule,
        n_estimators=100,
        step=1,
        verbose=0,
        min_features_to_select=1,
        method="auto",
        **kwargs,
    ):
        from tabensemb.utils.processors.rfecv import ExtendRFECV
        import shap

        cv = KFold(5)

        def importance_getter(estimator, data):
            np.random.seed(0)
            selected_data = data.loc[
                np.random.choice(
                    np.arange(data.shape[0]),
                    size=min(100, data.shape[0]),
                    replace=False,
                ),
                :,
            ]
            return np.mean(
                np.abs(shap.Explainer(estimator)(selected_data).values),
                axis=0,
            )

        rfecv = ExtendRFECV(
            estimator=datamodule.get_base_predictor(
                categorical=False,
                n_estimators=100,
                n_jobs=-1,
                random_state=0,
            ),
            step=step,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=min_features_to_select,
            n_jobs=-1,
            verbose=verbose,
            importance_getter=importance_getter if method == "shap" else method,
        )
        if len(datamodule.label_name) > 1:
            warnings.warn(
                f"Multi-target task is not supported by {self.__class__.__name__}. Only the first label is used."
            )
        data.columns = [str(x) for x in data.columns]
        rfecv.fit(
            data[datamodule.all_feature_names],
            data[datamodule.label_name[0]].values.flatten(),
        )
        retain_features = list(rfecv.get_feature_names_out())
        return retain_features


class VarianceFeatureSelector(AbstractFeatureSelector):
    """
    Remove features that almost (by a certain fraction) contain an identical value. Required arguments:

    thres: float
        If more than thres * 100 percent of values are the same, the feature is removed.
    """

    def __init__(self):
        super(VarianceFeatureSelector, self).__init__()

    def _get_feature_names_out(self, data, datamodule, thres=0.8, **kwargs):
        sel = VarianceThreshold(threshold=(thres * (1 - thres)))
        sel.fit(
            data[datamodule.all_feature_names],
            data[datamodule.label_name].values,  # Ignored.
        )
        retain_features = list(sel.get_feature_names_out())
        return retain_features


class CorrFeatureSelector(AbstractFeatureSelector):
    """
    Select features that are not correlated (in the sense of Pearson correlation). Correlated features will be ranked
    by SHAP using RandomForestRegressor, and the feature with the highest importance will be selected.
    Required arguments:

    thres:
        The threshold of pearson correlation.
    n_estimators:
        The number of trees used in random forests.
    """

    def __init__(self):
        super(CorrFeatureSelector, self).__init__()

    def _get_feature_names_out(
        self, data, datamodule, thres=0.8, n_estimators=100, **kwargs
    ):
        import shap

        abs_corr = datamodule.cal_corr(imputed=False, features_only=True).abs()
        where_corr = np.where(abs_corr > thres)
        corr_feature, corr_sets = get_corr_sets(
            where_corr, datamodule.cont_feature_names
        )
        rf = datamodule.get_base_predictor(
            categorical=False, n_estimators=n_estimators, n_jobs=-1, random_state=0
        )
        rf.fit(
            data[datamodule.all_feature_names],
            data[datamodule.label_name].values,
        )

        explainer = shap.Explainer(rf)
        shap_values = explainer(
            data.loc[
                np.random.choice(
                    np.array(data.index), size=min([100, len(data)]), replace=False
                ),
                datamodule.all_feature_names,
            ]
        )

        retain_features = list(
            np.setdiff1d(datamodule.cont_feature_names, corr_feature)
        )
        attr = np.mean(np.abs(shap_values.values), axis=0)
        print("Correlated features (Ranked by SHAP):")
        for corr_set in corr_sets:
            set_shap = [attr[datamodule.all_feature_names.index(x)] for x in corr_set]
            max_shap_feature = corr_set[set_shap.index(np.max(set_shap))]
            retain_features += [max_shap_feature]
            order = np.array(set_shap).argsort()
            corr_set_dict = {}
            for idx in order[::-1]:
                corr_set_dict[corr_set[idx]] = set_shap[idx]
            print(pretty(corr_set_dict))
        retain_features += datamodule.cat_feature_names
        return retain_features


class StandardScaler(AbstractTransformer, AbstractScaler):
    """
    The standard scaler implemented using StandardScaler from sklearn.
    """

    def __init__(self):
        super(StandardScaler, self).__init__()

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        scaler = skStandardScaler()
        data.loc[:, datamodule.cont_feature_names] = scaler.fit_transform(
            data.loc[:, datamodule.cont_feature_names]
        ).astype(np.float32)

        self.transformer = scaler
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        data.loc[:, datamodule.cont_feature_names] = self.transformer.transform(
            data.loc[:, datamodule.cont_feature_names]
        ).astype(np.float32)
        return data


class CategoricalOrdinalEncoder(AbstractTransformer):
    """
    The categorical feature encoder that transform string values to unique integer values, implemented using
    OrdinalEncoder from sklearn.
    """

    def __init__(self):
        super(CategoricalOrdinalEncoder, self).__init__()
        self.record_feature_mapping = None

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        data.loc[:, datamodule.cat_feature_names] = oe.fit_transform(
            data.loc[:, datamodule.cat_feature_names]
        ).astype(int)
        for feature, categories in zip(datamodule.cat_feature_names, oe.categories_):
            datamodule.cat_feature_mapping[feature] = categories
        self.transformer = oe
        self.record_feature_mapping = cp(datamodule.cat_feature_mapping)
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule, **kwargs):
        datamodule.cat_feature_mapping = cp(self.record_feature_mapping)
        try:
            res = self.transformer.transform(data.loc[:, datamodule.cat_feature_names])
            for idx, cat_feature in enumerate(datamodule.cat_feature_names):
                res[:, idx] = np.nan_to_num(
                    res[:, idx],
                    nan=list(self.record_feature_mapping[cat_feature]).index("UNK")
                    if "UNK" in self.record_feature_mapping.keys()
                    else len(self.record_feature_mapping),
                )
            data.loc[:, datamodule.cat_feature_names] = res.astype(int)
        except Exception as e:
            try:
                # Categorical features are already transformed.
                self.transformer.inverse_transform(
                    data.loc[:, datamodule.cat_feature_names]
                )
                return data
            except:
                raise Exception(
                    f"Categorical features are not compatible with the fitted OrdinalEncoder."
                )
        return data

    def var_slip(self, feature_name, x):
        return x


processor_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractProcessor):
        processor_mapping[name] = cls


def get_data_processor(name: str) -> Type[AbstractProcessor]:
    if name not in processor_mapping.keys():
        raise Exception(f"Data processor {name} not implemented.")
    elif not issubclass(processor_mapping[name], AbstractProcessor):
        raise Exception(f"{name} is not the subclass of AbstractProcessor.")
    else:
        return processor_mapping[name]
