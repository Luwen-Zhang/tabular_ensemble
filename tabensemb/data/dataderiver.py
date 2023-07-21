from tabensemb.utils import *
from tabensemb.data import AbstractDeriver
import inspect
from typing import Type
from .utils import get_corr_sets


class RelativeDeriver(AbstractDeriver):
    """
    Dividing a feature by another to derive a new feature. Required arguments are:

    absolute_col: str
        The feature that needs to be divided.
    relative2_col: str
        The feature that acts as the denominator.
    """

    def __init__(self):
        super(RelativeDeriver, self).__init__()

    def _required_cols(self, **kwargs):
        return ["absolute_col", "relative2_col"]

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False)

    def _derive(
        self,
        df,
        datamodule,
        **kwargs,
    ):
        absolute_col = kwargs["absolute_col"]
        relative2_col = kwargs["relative2_col"]

        relative = df[absolute_col] / df[relative2_col]
        relative = relative.values.reshape(-1, 1)

        return relative


class SampleWeightDeriver(AbstractDeriver):
    """
    Derive weight for each sample in the dataset.
    """

    def __init__(self):
        super(SampleWeightDeriver, self).__init__()
        self.percentile_dict = {}
        self.unique_vals = {}
        self.feature_weight = {}
        self.denominator = None

    def _required_cols(self, **kwargs):
        return []

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        datamodule,
        **kwargs,
    ):
        if datamodule.training:
            self.percentile_dict = {}
            self.unique_vals = {}
            self.feature_weight = {}
            self.denominator = None
        train_idx = datamodule.train_indices
        cont_feature_names = datamodule.cont_feature_names
        cat_feature_names = datamodule.cat_feature_names
        weight = pd.DataFrame(
            index=df.index, columns=["weight"], data=np.ones((len(df), 1))
        )
        for feature in cont_feature_names:
            if feature == kwargs["derived_name"]:
                continue
            # We can only calculate distributions based on known data, i.e. the training set.
            if datamodule.training:
                Q1 = np.percentile(
                    df.loc[train_idx, feature].dropna(axis=0), 25, method="midpoint"
                )
                Q3 = np.percentile(
                    df.loc[train_idx, feature].dropna(axis=0), 75, method="midpoint"
                )
                self.percentile_dict[feature] = (Q1, Q3)
            else:
                Q1, Q3 = self.percentile_dict[feature]
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            upper = df.index[np.where(df[feature] >= (Q3 + 1.5 * IQR))[0]]
            lower = df.index[np.where(df[feature] <= (Q1 - 1.5 * IQR))[0]]
            idx = np.union1d(upper, lower)
            if len(idx) == 0:
                continue
            if datamodule.training:
                train_upper = train_idx[
                    np.where(df.loc[train_idx, feature] >= (Q3 + 1.5 * IQR))[0]
                ]
                train_lower = train_idx[
                    np.where(df.loc[train_idx, feature] <= (Q1 - 1.5 * IQR))[0]
                ]
                train_outlier = np.union1d(train_upper, train_lower)
                p_outlier = len(train_outlier) / len(train_idx)
                feature_weight = -np.log10(p_outlier)
                self.feature_weight[feature] = feature_weight
            elif feature in self.feature_weight.keys():
                feature_weight = self.feature_weight[feature]
            else:
                continue
            weight.loc[idx, "weight"] = weight.loc[idx, "weight"] * (
                1.0 + 0.1 * feature_weight
            )

        for feature in cat_feature_names:
            if datamodule.training:
                all_cnts = df[feature].value_counts()
                unique_values = np.array(all_cnts.index)
                train_cnts = df.loc[train_idx, feature].value_counts()
                fitted_train_cnts = np.array(
                    [
                        train_cnts[x] if x in train_cnts.index else 0.0
                        for x in unique_values
                    ]
                )
                p_unique_values = fitted_train_cnts / len(train_idx)
                feature_weight = np.abs(
                    np.log10(p_unique_values) - np.log10(max(p_unique_values))
                )
                self.unique_vals[feature] = unique_values
                self.feature_weight[feature] = feature_weight
            elif feature in self.unique_vals.keys():
                unique_values = self.unique_vals[feature]
                feature_weight = self.feature_weight[feature]
            else:
                continue
            for value, w in zip(unique_values, feature_weight):
                where_value = df.index[np.where(df[feature] == value)[0]]
                weight.loc[where_value, "weight"] = weight.loc[
                    where_value, "weight"
                ] * (1.0 + 0.1 * w)

        if datamodule.training:
            self.denominator = 1 / np.sum(weight.values) * len(df)
        weight = weight.values * self.denominator
        return weight


class UnscaledDataDeriver(AbstractDeriver):
    """
    Record unscaled data in DataModule.derived_data so that TorchModels can access it.
    """

    def _required_cols(self, **kwargs):
        return []

    def _required_params(self, **kwargs):
        return []

    def _defaults(self):
        return dict(stacked=False, intermediate=False)

    def _derive(
        self,
        df,
        datamodule,
        **kwargs,
    ):
        if kwargs["stacked"]:
            raise Exception(
                f"{self.__class__.__name__} can not derive stacked features (behavior when "
                f"``datamodule._force_features=True`` is not defined)."
            )
        return df[datamodule.cont_feature_names].values


deriver_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractDeriver):
        deriver_mapping[name] = cls


def get_data_deriver(name: str) -> Type[AbstractDeriver]:
    if name not in deriver_mapping.keys():
        raise Exception(f"Data deriver {name} not implemented.")
    elif not issubclass(deriver_mapping[name], AbstractDeriver):
        raise Exception(f"{name} is not the subclass of AbstractDeriver.")
    else:
        return deriver_mapping[name]
