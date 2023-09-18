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
from typing import Type
from .utils import get_corr_sets, OrdinalEncoder


class SampleDataAugmentor(AbstractAugmenter):
    """
    This is a sample of data augmentation, which is not reasonable at all and is only used to test data augmentation.
    """

    def _get_augmented(
        self, data: pd.DataFrame, datamodule: DataModule
    ) -> pd.DataFrame:
        augmented = data.loc[data.index[-2:], :].copy()
        return augmented


class FeatureValueSelector(AbstractProcessor):
    """
    Select data with the specified feature value.

    Parameters
    ----------
    feature: str
        The feature that will be filtered.
    value: float
        The selected feature value.

    Notes
    -----
    The ``FeatureValueSelector`` will not change anything in the upcoming dataset, which means that the value in the
    upcoming set may exceed the range you expect. A typical error can be "IndexError: index out of range in self" from
    ``torch.embedding`` because of categorical features.
    """

    def _required_kwargs(self):
        return ["feature", "value"]

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule):
        feature = self.kwargs["feature"]
        value = self.kwargs["value"]
        indices = datamodule.select_by_value(selection={feature: value}, df=data)
        indices_retain_order = np.array([i for i in data.index if i in indices])
        data = data.loc[indices_retain_order, :]
        self.feature, self.value = feature, value
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule):
        indices = datamodule.select_by_value(
            selection={self.feature: self.value}, df=data
        )
        indices_retain_order = np.array([i for i in data.index if i in indices])
        if datamodule.training:
            if len(indices) == 0:
                raise Exception(
                    f"Value {self.value} not available for feature {self.feature}. Select from {data[self.feature].unique()}"
                )
            data = data.loc[indices_retain_order, :]
        else:
            if len(indices) == 0:
                warnings.warn(
                    f"Value {self.value} not available for feature {self.feature} selected by "
                    f"{self.__class__.__name__}."
                )
        return data


class IQRRemover(AbstractProcessor):
    """
    Remove outliers using the IQR strategy. Outliers are those
    out of the range [25-percentile - 1.5 * IQR, 75-percentile + 1.5 * IQR], where IQR = 75-percentile - 25-percentile.
    """

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule):
        print(f"Removing outliers by IQR. Original size: {len(data)}, ", end="")
        for feature in datamodule.extract_original_cont_feature_names(
            datamodule.cont_feature_names
        ):
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

    def _transform(self, data: pd.DataFrame, datamodule: DataModule):
        return data


class StdRemover(AbstractProcessor):
    """
    Remove outliers using the standard error strategy. Outliers are those out of the range of 3sigma.
    """

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule):
        print(f"Removing outliers by std. Original size: {len(data)}, ", end="")
        for feature in datamodule.extract_original_cont_feature_names(
            datamodule.cont_feature_names
        ):
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

    def _transform(self, data: pd.DataFrame, datamodule: DataModule):
        return data


class NaNFeatureRemover(AbstractFeatureSelector):
    """
    Remove features that contain no valid value.
    """

    def _get_feature_names_out(self, data, datamodule):
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

    def _defaults(self):
        return dict(
            n_estimators=100, step=1, verbose=0, min_features_to_select=1, method="auto"
        )

    def _get_feature_names_out(self, data, datamodule):
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
            # RFECV does not support categorical encoding. The estimator should have `coef_` or `feature_importances_`
            # so pipeline is not valid if importance_getter=="auto". shap can not handle a pipeline either.
            estimator=datamodule.get_base_predictor(
                categorical=False,
                n_estimators=self.kwargs["n_estimators"],
                n_jobs=-1,
                random_state=0,
            ),
            step=self.kwargs["step"],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=self.kwargs["min_features_to_select"],
            n_jobs=-1,
            verbose=self.kwargs["verbose"],
            importance_getter=importance_getter
            if self.kwargs["method"] == "shap"
            else self.kwargs["method"],
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
    Remove features that almost (by a certain fraction) contain an identical value.

    Parameters
    ----------
    thres: float
        If more than thres * 100 percent of values are the same, the feature is removed.
    """

    def _defaults(self):
        return dict(thres=0.8)

    def _get_feature_names_out(self, data, datamodule):
        thres = self.kwargs["thres"]
        sel = VarianceThreshold(threshold=(thres * (1 - thres)))
        sel.fit(
            data[datamodule.all_feature_names],
            data[datamodule.label_name].values.flatten()
            if len(datamodule.label_name) == 1
            else data[datamodule.label_name].values,  # Ignored.
        )
        retain_features = list(sel.get_feature_names_out())
        return retain_features


class CorrFeatureSelector(AbstractFeatureSelector):
    """
    Select features that are not correlated (in the sense of Pearson correlation). Correlated features will be ranked
    by SHAP using RandomForestRegressor, and the feature with the highest importance will be selected.

    Parameters
    ----------
    thres:
        The threshold of the Pearson correlation coefficient.
    n_estimators:
        The number of trees used in random forests.
    """

    def _defaults(self):
        return dict(thres=0.8, n_estimators=100)

    def _get_feature_names_out(self, data, datamodule):
        import shap

        abs_corr = datamodule.cal_corr(imputed=False, features_only=True).abs()
        where_corr = np.where(abs_corr > self.kwargs["thres"])
        corr_feature, corr_sets = get_corr_sets(
            where_corr, datamodule.cont_feature_names
        )
        rf = datamodule.get_base_predictor(
            categorical=False,
            n_estimators=self.kwargs["n_estimators"],
            n_jobs=-1,
            random_state=0,
        )
        rf.fit(
            data[datamodule.all_feature_names],
            data[datamodule.label_name].values.flatten()
            if len(datamodule.label_name) == 1
            else data[datamodule.label_name].values,
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


class StandardScaler(AbstractScaler):
    """
    A standard scaler implemented using StandardScaler from sklearn.
    """

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule):
        scaler = skStandardScaler()
        if len(datamodule.cont_feature_names) > 0:
            data[datamodule.cont_feature_names] = scaler.fit_transform(
                data[datamodule.cont_feature_names]
            ).astype(np.float64)

        self.transformer = scaler
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule):
        if len(datamodule.cont_feature_names) > 0:
            data[datamodule.cont_feature_names] = self.transformer.transform(
                data[datamodule.cont_feature_names]
            ).astype(np.float64)
        return data


class CategoricalOrdinalEncoder(AbstractTransformer):
    """
    A categorical feature encoder that transforms string values to unique integer values.
    See :class:`tabensemb.data.utils.OrdinalEncoder` for details.
    """

    def __init__(self, **kwargs):
        super(CategoricalOrdinalEncoder, self).__init__(**kwargs)
        self.record_feature_mapping = None

    def _fit_transform(self, data: pd.DataFrame, datamodule: DataModule):
        oe = OrdinalEncoder()
        data = oe.fit(data[datamodule.cat_feature_names]).transform(data)
        datamodule.cat_feature_mapping = {
            key: np.array(val) for key, val in oe.mapping.items()
        }
        self.transformer = oe
        self.record_feature_mapping = cp(datamodule.cat_feature_mapping)
        return data

    def _transform(self, data: pd.DataFrame, datamodule: DataModule):
        datamodule.cat_feature_mapping = cp(self.record_feature_mapping)
        data = self.transformer.transform(data)
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
