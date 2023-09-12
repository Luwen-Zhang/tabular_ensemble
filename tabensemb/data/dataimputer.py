import numpy as np
from tabensemb.utils import *
from tabensemb.data import AbstractImputer, AbstractSklearnImputer
import inspect
import sklearn.exceptions
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from typing import Type
from .datamodule import DataModule


class MiceLightgbmImputer(AbstractImputer):
    """
    An implementation of MICE with lightgbm.

    Parameters
    ----------
    **kwargs
        Arguments for ``miceforest.ImputationKernel.mice``
    """

    def _defaults(self):
        return dict(iterations=2, n_estimators=1)

    def _fit_transform(
        self, input_data: pd.DataFrame, datamodule: DataModule, **kwargs
    ):
        import miceforest as mf

        impute_features = self._get_impute_features(
            datamodule.cont_feature_names, input_data
        )
        no_nan = not np.any(np.isnan(input_data[impute_features].values))
        imputer = mf.ImputationKernel(
            input_data[impute_features], random_state=0, train_nonmissing=no_nan
        )
        imputer.mice(**self.kwargs)
        input_data[impute_features] = imputer.complete_data().values.astype(np.float64)
        imputer.compile_candidate_preds()
        self.transformer = imputer
        return input_data

    def _transform(self, input_data: pd.DataFrame, datamodule: DataModule, **kwargs):
        input_data[self.record_imputed_features] = (
            self.transformer.impute_new_data(
                new_data=input_data[self.record_imputed_features]
            )
            .complete_data()
            .values.astype(np.float64)
        )
        return input_data


class MiceImputer(AbstractSklearnImputer):
    """
    An implementation of MICE by sklearn.

    Parameters
    ----------
    **kwargs
        Arguments for ``sklearn.impute.IterativeImputer``
    """

    def _defaults(self):
        return {
            "max_iter": 1000,
            "random_state": 0,
            "tol": 1e-3,
            "sample_posterior": False,
        }

    def _new_imputer(self):
        # https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_sklearn_ice.py
        warnings.simplefilter(
            action="ignore", category=sklearn.exceptions.ConvergenceWarning
        )
        return IterativeImputer(**self.kwargs)


class MissForestImputer(AbstractSklearnImputer):
    """
    MICE-Random forest implemented using sklearn.

    Parameters
    ----------
    **kwargs
        Arguments for ``sklearn.ensemble.RandomForestRegressor``
    """

    def _defaults(self):
        return dict(
            n_estimators=1,
            max_depth=3,
            random_state=0,
            bootstrap=True,
            n_jobs=-1,
        )

    def _new_imputer(self):
        warnings.simplefilter(
            action="ignore", category=sklearn.exceptions.ConvergenceWarning
        )
        estimator_rf = RandomForestRegressor(**self.kwargs)
        return IterativeImputer(estimator=estimator_rf, random_state=0, max_iter=10)


class GainImputer(AbstractSklearnImputer):
    """
    Imputation using GAIN.

    Parameters
    ----------
    **kwargs
        Arguments for :class:`~tabensemb.utils.imputers.gain.GainImputation`
    """

    def _new_imputer(self):
        from tabensemb.utils.imputers.gain import GainImputation

        return GainImputation(**self.kwargs)


class MeanImputer(AbstractSklearnImputer):
    """
    Imputation with average values implemented using sklearn's SimpleImputer.

    Parameters
    ----------
    **kwargs
        Arguments for ``sklearn.impute.SimpleImputer`` (except for ``strategy``)
    """

    def _new_imputer(self):
        return SimpleImputer(strategy="mean", **self.kwargs)


class MedianImputer(AbstractSklearnImputer):
    """
    Imputation with median values implemented using sklearn's SimpleImputer.

    Parameters
    ----------
    **kwargs
        Arguments for ``sklearn.impute.SimpleImputer`` (except for ``strategy``)
    """

    def _new_imputer(self):
        return SimpleImputer(strategy="median", **self.kwargs)


class ModeImputer(AbstractSklearnImputer):
    """
    Imputation with mode values implemented using sklearn's SimpleImputer.

    Parameters
    ----------
    **kwargs
        Arguments for ``sklearn.impute.SimpleImputer`` (except for ``strategy``)
    """

    def _new_imputer(self):
        return SimpleImputer(strategy="most_frequent", **self.kwargs)


imputer_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractImputer):
        imputer_mapping[name] = cls


def get_data_imputer(name: str) -> Type[AbstractImputer]:
    if name not in imputer_mapping.keys():
        raise Exception(f"Data imputer {name} not implemented.")
    elif not issubclass(imputer_mapping[name], AbstractImputer):
        raise Exception(f"{name} is not the subclass of AbstractImputer.")
    else:
        return imputer_mapping[name]
