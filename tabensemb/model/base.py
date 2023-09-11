import os
import pickle
import types
import warnings
import numpy as np
import pandas as pd
import torch.optim.optimizer
import tabensemb
from tabensemb.utils import *
from tabensemb.trainer import Trainer, save_trainer
from tabensemb.data import DataModule
import skopt
from skopt import gp_minimize
import torch.utils.data as Data
import torch.nn as nn
from copy import deepcopy as cp
from typing import *
from skopt.space import Real, Integer, Categorical
import time
import pytorch_lightning as pl
from functools import partial
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from collections.abc import Iterable
from captum.attr import FeaturePermutation
import traceback
import math
import inspect


class AbstractModel:
    """
    The base class for all model bases.

    Attributes
    ----------
    exclude_models
        The names of models that should not be trained.
    init_params
        Arguments passed to :meth:`__init__`. See :meth:`save_kwargs`.
    limit_batch_size
        If ``batch_size // len(training set) < limit_batch_size``, the ``batch_size`` is forced to be
        ``len(training set)`` to avoid potential numerical issues. For Tabnet, this is extremely important because a
        small batch may cause NaNs and further CUDA device-side assert in the sparsemax function. Set to -1 to turn off
        this check (NOT RECOMMENDED!!). Note: Setting ``drop_last=True`` for ``torch.utils.data.DataLoader`` is fine,
        but I think (i) having access to all data points in one epoch is beneficial for some models, (ii) If using a
        large dataset and a large ``batch_size``, it is possible that the last batch is so large that contains
        essential information, (iii) the user should have full control for this. If you want to use ``drop_last`` in
        your code, use the ``original_batch_size`` in ``kwargs`` passed to :class:`AbstractModel` methods.
    model
        A dictionary of models.
    model_params
        Hyperparameters that contain all keys in :meth:`_initial_values` for each model. In cross validation runs, the
        parameters in the previous run will be loaded for the current run.
    model_subset
        The names of models selected to be trained in the model base.
    program
        The name of the model base.
    root
        The place where all files of the model base are stored.
    store_in_harddisk
        Whether to save models in the hard disk.
    trainer
        A :class:`tabensemb.trainer.Trainer` instance.
    device
    """

    def __init__(
        self,
        trainer: Trainer,
        program: str = None,
        model_subset: List[str] = None,
        exclude_models: List[str] = None,
        store_in_harddisk: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        trainer:
            A :class:`~tabensemb.trainer.Trainer` instance that contains all information and datasets and will be
            linked to the model base. The trainer has loaded configs and data.
        program:
            The name of the model base. If None, the name from :meth:`_get_program_name` is used.
        model_subset:
            The names of models selected to be trained in the model base.
        exclude_models:
            The names of models that should not be trained. Only one of ``model_subset`` and ``exclude_models`` can
            be specified.
        store_in_harddisk:
            Whether to save models in the hard disk. If the global setting
            ``tabensemb.setting["low_memory"]`` is True, True is used.
        **kwargs:
            Ignored.
        """
        self.trainer = trainer
        if not hasattr(trainer, "args"):
            raise Exception(f"trainer.load_config is not called.")
        self.model = None
        self.leaderboard = None
        self.model_subset = model_subset
        self.exclude_models = exclude_models
        if self.model_subset is not None and self.exclude_models is not None:
            raise Exception(
                f"Only one of model_subset and exclude_models can be specified."
            )
        self.store_in_harddisk = (
            True if tabensemb.setting["low_memory"] else store_in_harddisk
        )
        self.program = self._get_program_name() if program is None else program
        self.init_params = {}
        self.model_params = {}
        self.save_kwargs(d=self.init_params, ignore=["trainer", "self", "frame"])
        self._check_space()
        self._mkdir()
        self.limit_batch_size = 6

    def save_kwargs(self, d: dict = None, ignore: List[str] = None):
        """
        Save all args and kwargs of the caller except for those in ``ignore``. It will trace back to the top caller that
        has the same method name and the same class of ``self`` as that of the current frame. For example, in nested
        __init__ calls of inherited classes, it will trace back to the first __init__ call and record kwargs layer by
        layer until it reaches the current caller.

        Notes
        -----
        It will be automatically called in :meth:`__init__`.

        Parameters
        ----------
        d
            The dictionary to save params.
        ignore
            kwargs names to be ignored

        Returns
        -------
        dict
            The dictionary with the recorded kwargs
        """
        ignore = [] if ignore is None else ignore
        d = {} if d is None else d
        caller_frame = inspect.currentframe().f_back
        caller_function = inspect.getframeinfo(caller_frame).function
        nest_init_params = []
        while (
            isinstance(caller_frame, types.FrameType)
            and "self" in caller_frame.f_locals.keys()
            and isinstance(caller_frame.f_locals["self"], self.__class__)
            and inspect.getframeinfo(caller_frame).function == caller_function
        ):
            argvalues = inspect.getargvalues(caller_frame)
            kwargs = {
                key: argvalues.locals[key]
                for key in argvalues.args
                if key not in ignore
            }
            nest_init_params.append(kwargs)
            caller_frame = caller_frame.f_back
        for params in nest_init_params[::-1]:
            d.update(params)
        return d

    def reset(self):
        """
        Reset the model base by calling __init__ with the recorded kwargs from :meth:`save_kwargs`.
        """
        self.__init__(trainer=self.trainer, **self.init_params)

    @property
    def device(self):
        """
        The device set in the linked :class:`~tabensemb.trainer.Trainer`.

        Returns
        -------
        str
            "cpu" or "cuda"
        """
        return self.trainer.device

    def fit(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        model_subset: List[str] = None,
        derived_data: Dict[str, np.ndarray] = None,
        verbose: bool = True,
        warm_start: bool = False,
        bayes_opt: bool = False,
    ):
        """
        Fit all models using a tabular dataset.

        Notes
        -----
        The loaded dataset in the linked :class:`~tabensemb.trainer.Trainer` will be replaced.

        Parameters
        ----------
        df:
            A tabular dataset.
        cont_feature_names:
            The names of continuous features.
        cat_feature_names:
            The names of categorical features.
        label_name:
            The names of targets.
        model_subset:
            The names of a subset of all available models (in :meth:`get_model_names`). Only these models will be
            trained.
        derived_data:
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`. If None,
            unstacked data will be re-derived.
        verbose:
            Verbosity.
        warm_start:
            Finetune models based on previous trained models.
        bayes_opt:
            Whether to perform Gaussian-process-based Bayesian Hyperparameter Optimization for each model.
        """
        self.trainer.set_status(training=True)
        trainer_state = cp(self.trainer)
        self.trainer.datamodule.set_data(
            df,
            cont_feature_names=cont_feature_names,
            cat_feature_names=cat_feature_names,
            label_name=label_name,
            derived_data=derived_data,
            warm_start=warm_start if self._trained else False,
            verbose=verbose,
            all_training=True,
        )
        if bayes_opt != self.trainer.args["bayes_opt"]:
            self.trainer.args["bayes_opt"] = bayes_opt
            if verbose:
                print(
                    f"The argument bayes_opt of fit() conflicts with Trainer.bayes_opt. Use the former one."
                )
        self.train(
            dump_trainer=False,
            verbose=verbose,
            model_subset=model_subset,
            warm_start=warm_start if self._trained else False,
        )
        self.trainer.load_state(trainer_state)
        self.trainer.set_status(training=False)

    def train(self, *args, stderr_to_stdout=False, **kwargs):
        """
        Train the model base using the dataset in the linked :class:`~tabensemb.trainer.Trainer` directly.

        Parameters
        ----------
        *args:
            Arguments of :meth:`_train`.
        stderr_to_stdout:
            Redirect stderr to stdout. Useful for notebooks.
        **kwargs:
            Arguments of :meth:`_train`.
        """
        self.trainer.set_status(training=True)
        verbose = "verbose" not in kwargs.keys() or kwargs["verbose"]
        if verbose:
            print(f"\n-------------Run {self.program}-------------\n")
        with PlainText(disable=not stderr_to_stdout):
            self._train(*args, **kwargs)
        if self.model is None or len(self.model) == 0:
            warnings.warn(f"No model has been trained for {self.__class__.__name__}.")
        if verbose:
            print(f"\n-------------{self.program} End-------------\n")
        self.trainer.set_status(training=False)

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str,
        model: Any = None,
        derived_data: dict = None,
        ignore_absence: bool = False,
        proba: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Make inferences on a new dataset using the selected model.

        Parameters
        ----------
        df:
            A new tabular dataset.
        model_name:
            A selected name of a model, which is already trained.
        model:
            The model returned by :meth:`_new_model`. If None, the model will be loaded from ``self.model``.
        derived_data:
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`. If None,
            unstacked data will be re-derived.
        ignore_absence:
            Whether to ignore absent keys in ``derived_data``. Use True only when the model does not use derived_data.
        proba:
            Return probabilities instead of predicted classes for classification models.
        **kwargs:
            Arguments of :meth:`_predict`.

        Returns
        -------
        np.ndarray
            Predicted target. Always 2d np.ndarray.
        """
        self.trainer.set_status(training=False)
        if self.model is None:
            raise Exception("Run fit() before predict().")
        if model_name not in self.get_model_names():
            raise Exception(
                f"Model {model_name} is not available. Select among {self.get_model_names()}"
            )
        df, derived_data = self.trainer.datamodule.prepare_new_data(
            df, derived_data, ignore_absence
        )
        res = self._predict(
            df,
            model_name,
            derived_data,
            model=model,
            **kwargs,
        )
        if self.trainer.datamodule.task == "regression" or proba:
            return res
        else:
            return self.trainer.datamodule.label_ordinal_encoder.inverse_transform(
                convert_proba_to_target(res, self.trainer.datamodule.task)
            )

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """
        Predict probabilities of each class.

        Parameters
        ----------
        args
            Positional arguments of :meth:`predict`.
        kwargs
            Arguments of :meth:`predict`, except for ``proba``.

        Returns
        -------
        np.ndarray
            For binary tasks, a (n_samples, 1) np.ndarray is returned as the probability of positive. For multiclass
            tasks, a (n_samples, n_classes) np.ndarray is returned.
        """
        if self.trainer.datamodule.task == "regression":
            raise Exception(f"Calling predict_proba on regression models.")
        if "proba" in kwargs.keys():
            del kwargs["proba"]
        return self.predict(*args, proba=True, **kwargs)

    def detach_model(self, model_name: str, program: str = None) -> "AbstractModel":
        """
        Detach the chosen model to a separate model base with the same linked :class:`~tabensemb.trainer.Trainer`.

        Parameters
        ----------
        model_name:
            The name of the model to be detached.
        program:
            The new name of the detached model base. If the name is the same as the original one, the detached model is
            stored in memory to avoid overwriting the original model.

        Returns
        -------
        AbstractModel
            An AbstractModel containing the chosen model.
        """
        if not isinstance(self.model, dict) and not isinstance(self.model, ModelDict):
            raise Exception(f"The modelbase does not support model detaching.")
        program = program if program is not None else self.program
        tmp_model = cp(self)
        tmp_model.trainer = self.trainer
        tmp_model.program = program
        tmp_model.model_subset = [model_name]
        if tmp_model.store_in_harddisk and program != self.program:
            tmp_model._mkdir()
            tmp_model.model = ModelDict(path=tmp_model.root)
        else:
            tmp_model.store_in_harddisk = False
            tmp_model.model = {}
        tmp_model.model[model_name] = cp(self.model[model_name])
        if model_name in self.model_params.keys():
            tmp_model.model_params[model_name] = cp(self.model_params[model_name])
        return tmp_model

    def set_path(self, path: Union[os.PathLike, str]):
        """
        Set the path of the model base (usually a trained one), including the paths of its models. It is used when
        migrating models to another directory.

        Parameters
        ----------
        path
            The path of the model base.
        """
        if hasattr(self, "root"):
            self.root = path
        if self.store_in_harddisk:
            if hasattr(self, "model") and self.model is not None:
                self.model.root = path
                for name in self.model.model_path.keys():
                    self.model.model_path[name] = os.path.join(self.root, name) + ".pkl"

    def new_model(self, model_name: str, verbose: bool, **kwargs):
        """
        A wrapper method to generate a new model while keeping the random seed constant.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to generate the model. It contains all arguments in :meth:`_initial_values`.

        Returns
        -------
        Any
            A new model (without any restriction to its type). It will be passed to :meth:`_train_single_model` and
            :meth:`_pred_single_model`.

        See Also
        --------
        :meth:`_new_model`
        """
        set_random_seed(tabensemb.setting["random_seed"])
        required_models = self._get_required_models(model_name=model_name)
        if required_models is not None:
            kwargs["required_models"] = required_models
        return self._new_model(model_name=model_name, verbose=verbose, **kwargs)

    def cal_feature_importance(
        self, model_name, method, **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate feature importance using a specified model.

        Parameters
        ----------
        model_name
            The selected model in the model base.
        method
            The method to calculate importance. "permutation" or "shap".
        kwargs
            Ignored.

        Returns
        -------
        np.ndarray
            Values of feature importance.
        list
            Corresponding feature names.
        """
        datamodule = self.trainer.datamodule
        all_feature_names = self.trainer.all_feature_names
        label_name = self.trainer.label_name
        if method == "permutation":
            attr = np.zeros((len(all_feature_names),))
            test_data = datamodule.X_test
            base_pred = self.predict(
                test_data,
                derived_data=datamodule.D_test,
                model_name=model_name,
            )
            base_metric = metric_sklearn(
                test_data[label_name].values, base_pred, metric="rmse"
            )
            for idx, feature in enumerate(all_feature_names):
                df = test_data.copy()
                shuffled = df.loc[:, feature].values
                np.random.shuffle(shuffled)
                df.loc[:, feature] = shuffled
                perm_pred = self.predict(
                    df,
                    derived_data=datamodule.derive_unstacked(df),
                    model_name=model_name,
                )
                attr[idx] = np.abs(
                    metric_sklearn(df[label_name].values, perm_pred, metric="rmse")
                    - base_metric
                )
            attr /= np.sum(attr)
        elif method == "shap":
            attr = self.cal_shap(model_name=model_name)
        else:
            raise NotImplementedError
        importance_names = cp(all_feature_names)
        return attr, importance_names

    def cal_shap(self, model_name: str, **kwargs) -> np.ndarray:
        """
        Calculate SHAP values using a specified model. ``shap.KernelExplainer`` is called, and ``shap.kmeans`` is
        called to summarize the training data to 10 samples as the background data and 10 random samples in the testing
        set are explained, which will bias the results.

        Parameters
        ----------
        model_name
            The selected model in the model base.
        kwargs
            Ignored for the compatibility of TorchModel.

        Returns
        -------
        attr
            The SHAP values.
        """
        import shap

        trainer_df = self.trainer.df
        train_indices = self.trainer.train_indices
        test_indices = self.trainer.test_indices
        all_feature_names = self.trainer.all_feature_names
        datamodule = self.trainer.datamodule
        background_data = shap.kmeans(
            trainer_df.loc[train_indices, all_feature_names], 10
        )
        warnings.filterwarnings(
            "ignore",
            message="The default of 'normalize' will be set to False in version 1.2 and deprecated in version 1.4.",
        )

        def func(data):
            df = pd.DataFrame(columns=all_feature_names, data=data)
            return self.predict(
                df,
                model_name=model_name,
                derived_data=datamodule.derive_unstacked(df, categorical_only=True),
                ignore_absence=True,
            ).flatten()

        test_indices = np.random.choice(test_indices, size=10, replace=False)
        test_data = trainer_df.loc[test_indices, all_feature_names].copy()
        shap_values = shap.KernelExplainer(func, background_data).shap_values(test_data)
        attr = (
            np.concatenate(
                [np.mean(np.abs(shap_values[0]), axis=0)]
                + [np.mean(np.abs(x), axis=0) for x in shap_values[1:]],
            )
            if type(shap_values) == list and len(shap_values) > 1
            else np.mean(np.abs(shap_values), axis=0)
        )
        return attr

    def _check_params(self, model_name, **kwargs):
        """
        Check the validity of hyperparameters. This is implemented originally for batch_size because TabNet crashes
        when batch_size is small under certain situations.

        Parameters
        ----------
        model_name
            The name of a selected model.
        kwargs
            Parameters to generate the model. It contains all arguments in :meth:`_initial_values`.

        Returns
        -------
        dict
            The checked kwargs.
        """
        if "batch_size" in kwargs.keys():
            batch_size = kwargs["batch_size"]
            kwargs["original_batch_size"] = batch_size
            n_train = len(self.trainer.train_indices)
            limit_batch_size = self.limit_batch_size
            if limit_batch_size == -1:
                if 1 < n_train % batch_size < 4 or batch_size < 4:
                    warnings.warn(
                        f"Using batch_size={batch_size} and len(training set)={n_train}, which will make the mini "
                        f"batch extremely small. A very small batch may cause unexpected numerical issue, especially "
                        f"for TabNet. However, the attribute `limit_batch_size` is set to -1."
                    )
                if n_train % batch_size == 1:
                    raise Exception(
                        f"Using batch_size={batch_size} and len(training set)={n_train}, which will make the "
                        f"mini batch illegal. However, the attribute `limit_batch_size` is set to -1."
                    )
            if -1 < limit_batch_size < 2:
                warnings.warn(
                    f"limit_batch_size={limit_batch_size} is illegal. Use limit_batch_size=2 instead."
                )
                limit_batch_size = 2
            new_batch_size = batch_size
            if model_name == "TabNet":
                _new_batch_size = 64
                if new_batch_size < _new_batch_size:
                    warnings.warn(
                        f"For TabNet, using small batch_size ({new_batch_size}) may trigger CUDA device-side assert. "
                        f"Using batch_size={_new_batch_size} instead."
                    )
                    new_batch_size = _new_batch_size
            if 0 < n_train % new_batch_size < limit_batch_size:
                _new_batch_size = int(math.ceil(n_train / (n_train // new_batch_size)))
                warnings.warn(
                    f"Using batch_size={new_batch_size} and len(training set)={n_train}, which will make the mini batch "
                    f"smaller than limit_batch_size={limit_batch_size}. Using batch_size={_new_batch_size} instead."
                )
                new_batch_size = _new_batch_size
            kwargs["batch_size"] = new_batch_size
        return kwargs

    def _get_required_models(self, model_name) -> Union[Dict, None]:
        """
        Extract models specified in :meth:`required_models`.

        Parameters
        ----------
        model_name
            The name of the model.

        Returns
        -------
        dict or None
            A dictionary of extracted models required by the model.
        """
        required_model_names = self.required_models(model_name)
        if required_model_names is not None:
            required_models = {}
            for name in required_model_names:
                if name == model_name:
                    raise Exception(f"The model {model_name} is required by itself.")
                if name in self._get_model_names():
                    if name not in self.model.keys():
                        raise Exception(
                            f"Model {name} is required for model {model_name}, but is not trained."
                        )
                    required_models[name] = self.model[name]
                elif name.startswith("EXTERN_"):
                    spl = name.split("_")
                    if len(spl) not in [3, 4] or (len(spl) == 4 and spl[-1] != "WRAP"):
                        raise Exception(
                            f"Unrecognized required model name {name} from external model bases."
                        )
                    program, ext_model_name = spl[1], spl[2]
                    wrap = spl[-1] == "WRAP"
                    try:
                        modelbase = self.trainer.get_modelbase(program=program)
                    except:
                        if self.trainer.training:
                            raise Exception(
                                f"Model base {program} is required for model {model_name}, but does not exist."
                            )
                        else:
                            raise Exception(
                                f"Model base {program} is required for model {model_name}, but does not exist. It is "
                                f"mainly caused by model detaching and is currently not supported for models that "
                                f"requires other models."
                            )
                    try:
                        detached_model = modelbase.detach_model(
                            model_name=ext_model_name
                        )
                    except Exception as e:
                        raise Exception(
                            f"Model {ext_model_name} can not be detached from model base {program}. Exception:\n{e}"
                        )
                    if wrap:
                        from .pytorch_tabular import (
                            PytorchTabular,
                            PytorchTabularWrapper,
                        )
                        from .widedeep import WideDeep, WideDeepWrapper

                        if isinstance(detached_model, PytorchTabular):
                            detached_model = PytorchTabularWrapper(detached_model)
                        elif isinstance(detached_model, WideDeep):
                            detached_model = WideDeepWrapper(detached_model)
                        elif isinstance(detached_model, TorchModel):
                            detached_model = TorchModelWrapper(detached_model)
                        else:
                            raise Exception(
                                f"{type(detached_model)} does not support wrapping. Supported model bases "
                                f"are PytorchTabular, WideDeep, and TorchModels."
                            )
                    required_models[name] = detached_model
                else:
                    raise Exception(
                        f"Unrecognized model name {name} required by {model_name}."
                    )
            return required_models
        else:
            return None

    def required_models(self, model_name: str) -> Union[List[str], None]:
        """
        The names of models required by the requested model. If not None and the required model is
        trained, the required model will be passed to :meth:`_new_model`.
        If models from other model bases are required, the name should be
        ``EXTERN_{Name of the model base}_{Name of the model}``

        Notes
        -----
        For :class:`TorchModel`, if the required model is in the :class:`TorchModel` itself, the
        :class:`.AbstractNN` is passed to :meth:`_new_model`; if the required model is in another model base, the
        :class:`.AbstractModel` is passed.
        """
        return None

    def inspect_attr(
        self,
        model_name: str,
        attributes: List[str],
        df=None,
        derived_data=None,
        to_numpy=True,
    ) -> Dict[str, Any]:
        """
        Get attributes of the model after evaluating the model on training, validation, and testing sets respectively.
        If ``df`` is given, values after evaluating the given set are returned.

        Parameters
        ----------
        model_name
            The name of the inspected model.
        attributes
            The requested attributes. If the model does not have the attribute, None is returned.
        df
            The tabular dataset.
        derived_data:
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`. If None,
            unstacked data will be re-derived.
        to_numpy
            If True, call ``numpy()`` if the attribute is a torch.Tensor.

        Returns
        -------
        dict
            A dict with keys ``train``, ``val``, and ``test`` if ``df`` is not given, and each value contains
            the attributes requested. If ``df`` is given, a dict with a single key ``USER_INPUT`` and the corresponding
            value contains the attributes. The prediction is also included with the key ``prediction``.
        """

        def to_cpu(attr):
            if isinstance(attr, nn.Module):
                attr = attr.to("cpu")
            elif isinstance(attr, torch.Tensor):
                attr = attr.detach().cpu()
                if to_numpy:
                    attr = attr.numpy()
            return attr

        data = self.trainer.datamodule
        model = self.model[model_name]
        if df is None:
            inspect_dict = {part: {} for part in ["train", "val", "test"]}
            for X, D, part in [
                (data.X_train, data.D_train, "train"),
                (data.X_val, data.D_val, "val"),
                (data.X_test, data.D_test, "test"),
            ]:
                prediction = self._predict(
                    X, derived_data=D, model_name=model_name, model=model
                )
                for attr in attributes:
                    inspect_dict[part][attr] = to_cpu(cp(getattr(model, attr, None)))
                inspect_dict[part]["prediction"] = prediction
        else:
            inspect_dict = {"USER_INPUT": {}}
            prediction = self.predict(
                df, model_name=model_name, derived_data=derived_data, model=model
            )
            for attr in attributes:
                inspect_dict["USER_INPUT"][attr] = to_cpu(
                    cp(getattr(model, attr, None))
                )
            inspect_dict["USER_INPUT"]["prediction"] = prediction
        return inspect_dict

    def _predict_all(
        self, verbose: bool = True, test_data_only: bool = False
    ) -> Dict[str, Dict]:
        """
        Make inferences on training/validation/testing datasets to evaluate the performance of all models.

        Parameters
        ----------
        verbose:
            Verbosity.
        test_data_only:
            Whether to predict only the testing set. If True, the whole dataset will be evaluated.

        Returns
        -------
        dict
            A dict of results. Its keys are "Training", "Testing", and "Validation". Its values are tuples containing
            predicted values and ground truth values
        """
        self.trainer.set_status(training=False)
        self._check_train_status()

        model_names = self.get_model_names()
        data = self.trainer.datamodule

        predictions = {}
        tc = TqdmController()
        tc.disable_tqdm()
        for idx, model_name in enumerate(model_names):
            if verbose:
                print(model_name, f"{idx + 1}/{len(model_names)}")
            if not test_data_only:
                y_train_pred = self._predict(
                    data.X_train,
                    derived_data=data.D_train,
                    model_name=model_name,
                )
                y_val_pred = self._predict(
                    data.X_val, derived_data=data.D_val, model_name=model_name
                )
                y_train = data.y_train
                y_val = data.y_val
            else:
                y_train_pred = y_train = None
                y_val_pred = y_val = None

            y_test_pred = self._predict(
                data.X_test, derived_data=data.D_test, model_name=model_name
            )

            predictions[model_name] = {
                "Training": (y_train_pred, y_train),
                "Testing": (y_test_pred, data.y_test),
                "Validation": (y_val_pred, y_val),
            }

        tc.enable_tqdm()
        return predictions

    def _predict(
        self,
        df: pd.DataFrame,
        model_name: str,
        derived_data: Dict = None,
        model: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make prediction based on a tabular dataset using the selected model.

        Parameters
        ----------
        df:
            A new tabular dataset that has the same structure as ``self.trainer.datamodule.X_test``.
        model_name:
            A name of a selected model, which is already trained. It is used to process the input data if any specific
            routine is defined for this model in :meth:`~AbstractModel._data_preprocess`.
        derived_data:
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked` that has the
            same structure as ``self.trainer.datamodule.D_test``.
        model:
            The model returned by :meth:`_new_model`. If None, the model will be loaded from ``self.model``.
        **kwargs:
            Ignored.

        Returns
        -------
        np.ndarray
            Prediction of the target.
        """
        self.trainer.set_status(training=False)
        X_test = self._data_preprocess(df, derived_data, model_name=model_name)
        return self._pred_single_model(
            self.model[model_name] if model is None else model,
            X_test=X_test,
            verbose=False,
        )

    def _custom_training_params(self, model_name) -> Dict:
        """
        Customized training settings to override settings in the configuration. The configuration will be restored after
        training the model.

        Parameters
        ----------
        model_name
            The name of a selected model

        Returns
        -------
        dict
            A dict of training parameters.
        """
        return {}

    def _train(
        self,
        model_subset: List[str] = None,
        dump_trainer: bool = True,
        verbose: bool = True,
        warm_start: bool = False,
        **kwargs,
    ):
        """
        The basic framework of training models, including processing the dataset, training each model (with/without
        bayesian hyperparameter optimization), and evaluating them on the dataset.

        Parameters
        ----------
        model_subset:
            The names of a subset of all available models (in :func:`get_model_names`). Only these models will be
            trained.
        dump_trainer:
            Whether to save the trainer after models are trained.
        verbose:
            Verbosity.
        warm_start:
            Finetune models based on previous trained models.
        **kwargs:
            Ignored.
        """
        self.trainer.set_status(training=True)
        if self.model is None:
            if self.store_in_harddisk:
                self.model = ModelDict(path=self.root)
            else:
                self.model = {}
        for model_name in (
            self.get_model_names() if model_subset is None else model_subset
        ):
            if verbose:
                print(f"Training {model_name}")
            data = self._train_data_preprocess(model_name)
            tmp_params = self._get_params(model_name, verbose=verbose)
            space = self._space(model_name=model_name)

            original_args = self.trainer.args
            args = self.trainer.args.copy()
            args.update(self._custom_training_params(model_name))
            self.trainer.args = args

            do_bayes_opt = args["bayes_opt"] and not warm_start
            total_epoch = args["epoch"] if not tabensemb.setting["debug_mode"] else 2
            if do_bayes_opt and len(space) > 0:
                min_calls = len(space)
                bayes_calls = (
                    max([args["bayes_calls"], min_calls])
                    if not tabensemb.setting["debug_mode"]
                    else min_calls
                )
                callback = BayesCallback(total=bayes_calls)
                global _bayes_objective

                @skopt.utils.use_named_args(space)
                def _bayes_objective(**params):
                    params = self._check_params(model_name, **params)
                    try:
                        with HiddenPrints():
                            model = self.new_model(
                                model_name=model_name, verbose=False, **params
                            )

                            self._train_single_model(
                                model,
                                epoch=args["bayes_epoch"]
                                if not tabensemb.setting["debug_mode"]
                                else 1,
                                X_train=data["X_train"],
                                y_train=data["y_train"],
                                X_val=data["X_val"],
                                y_val=data["y_val"],
                                verbose=False,
                                warm_start=False,
                                in_bayes_opt=True,
                                **params,
                            )

                        res = self._bayes_eval(
                            model,
                            data["X_train"],
                            data["y_train"],
                            data["X_val"],
                            data["y_val"],
                        )
                    except Exception as e:
                        joint_trackback = "".join(
                            traceback.format_exception(e.__class__, e, e.__traceback__)
                        )
                        print(f"An exception occurs when evaluating a bayes call:")
                        print(joint_trackback)
                        print("with the following parameters:")
                        print(params)
                        if (
                            model_name == "TabNet"
                            and "CUDA error: device-side assert triggered"
                            in joint_trackback
                        ):
                            print(
                                "You are using TabNet and a CUDA device-side assert is triggered. You encountered\n"
                                "the same issue as I did. For TabNet, it is really weird that if a batch is extremely\n"
                                "small (less than 5 maybe), during back-propagation, the gradient of its embedding\n"
                                "may contain NaN, which, in the next step, causes CUDA device-side assert in\n"
                                "sparsemax. See these two issues:\n"
                                "https://github.com/dreamquark-ai/tabnet/issues/135\n"
                                "https://github.com/dreamquark-ai/tabnet/issues/432\n"
                            )
                        if (
                            "CUDA error: device-side assert triggered"
                            in joint_trackback
                        ):
                            raise ValueError(
                                "A CUDA device-side assert is triggered. Unfortunately, CUDA device-side assert will\n"
                                "make the entire GPU session not accessible, the whole hyperparameter optimization\n"
                                "process invalid, and the final model training raising an exception. The error is\n"
                                "just re-raised because currently there is no way to restart the GPU session and\n"
                                "continue the HPO process. Please tell me if there is a solution."
                            )
                        print(f"Returning a large value instead.")
                        res = 100
                    # If a result from one bayes opt iteration is very large (over 10000) caused by instability of the
                    # model, it can not be fully reproduced during another execution and has error (though small, it
                    # disturbs bayes optimization).
                    if res > 1000:
                        print(
                            f"The loss value ({res}) is greater than 1000 and 1000 will be returned. Consider "
                            f"debugging such instability of the model, or check whether the loss value is normalized by"
                            f"the number of samples."
                        )
                        return 1000
                    # To guarantee reproducibility on different machines.
                    return round(res, 4)

                with warnings.catch_warnings():
                    # To obtain clean progress bar.
                    warnings.filterwarnings(
                        "ignore",
                        message="The objective has been evaluated at this point before",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="`pytorch_lightning.utilities.cloud_io.get_filesystem` has been deprecated in v1.8.0 and will be removed in v1.10.0.",
                    )
                    if (
                        "batch_size" in tmp_params.keys()
                        and "original_batch_size" in tmp_params.keys()
                    ):
                        tmp_params["batch_size"] = tmp_params["original_batch_size"]
                    result = gp_minimize(
                        _bayes_objective,
                        space,
                        n_calls=bayes_calls,
                        n_initial_points=10
                        if not tabensemb.setting["debug_mode"]
                        else 0,
                        callback=callback.call,
                        random_state=0,
                        x0=[tmp_params[s.name] for s in space],
                    )
                opt_params = {s.name: val for s, val in zip(space, result.x)}
                params = tmp_params.copy()
                params.update(opt_params)
                params = self._check_params(model_name, **params)
                self.model_params[model_name] = cp(params)
                callback.close()
                skopt.dump(
                    result,
                    add_postfix(os.path.join(self.root, f"{model_name}_skopt.pt")),
                )
                tmp_params = self._get_params(
                    model_name=model_name, verbose=verbose
                )  # to announce the optimized params.
            elif do_bayes_opt and len(space) == 0:
                warnings.warn(
                    f"No hyperparameter space defined for model {model_name}."
                )

            tmp_params = self._check_params(model_name, **tmp_params)
            if not warm_start or (
                warm_start and (not self._trained or not self._support_warm_start)
            ):
                if warm_start and not self._support_warm_start:
                    warnings.warn(
                        f"{self.__class__.__name__} does not support warm_start."
                    )
                model = self.new_model(
                    model_name=model_name, verbose=verbose, **tmp_params
                )
            else:
                model = self.model[model_name]

            self._train_single_model(
                model,
                epoch=total_epoch,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                verbose=verbose,
                warm_start=warm_start,
                in_bayes_opt=False,
                **tmp_params,
            )

            def pred_set(X, y, name):
                pred = self._pred_single_model(model, X, verbose=False)
                metric, loss = self._default_metric_sklearn(y, pred)
                if verbose:
                    print(f"{name} {metric} loss: {loss:.5f}")

            pred_set(data["X_train"], data["y_train"], "Training")
            pred_set(data["X_val"], data["y_val"], "Validation")
            pred_set(data["X_test"], data["y_test"], "Testing")
            self.model[model_name] = model
            torch.cuda.empty_cache()
            self.trainer.args = original_args

        self.trainer.set_status(training=False)
        if dump_trainer:
            save_trainer(self.trainer)

    def _default_metric_sklearn(self, y_true, y_pred):
        """
        Calculate MSE loss for regression tasks and log loss for classification tasks using sklearn APIs.

        Parameters
        ----------
        y_true
            Ground truth values.
        y_pred
            Predicted values.

        Returns
        -------
        str
            "mse" for regression tasks and "log_loss" for classification tasks.
        float
            MSE loss for regression tasks and log loss for classification tasks
        """
        task = self.trainer.datamodule.task
        if task == "regression":
            metric = "mse"
            loss = auto_metric_sklearn(y_true, y_pred, metric, "regression")
        elif task == "binary":
            metric = "log_loss"
            loss = auto_metric_sklearn(y_true, y_pred, metric, "binary")
        elif task == "multiclass":
            metric = "log_loss"
            loss = auto_metric_sklearn(y_true, y_pred, metric, "multiclass")
        else:
            raise NotImplementedError
        return metric, loss

    def _bayes_eval(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        """
        Evaluate the model for Bayesian optimization iterations. The larger one of the training loss and the validation
        loss is returned by default.

        Parameters
        ----------
        model
            The model returned by :meth:`_new_model`.
        X_train
            The training data from :meth:`_train_data_preprocess`.
        y_train
            The target of the training data from :meth:`_train_data_preprocess`.
        X_val
            The validation data from :meth:`_train_data_preprocess`.
        y_val
            The target of the validation data from :meth:`_train_data_preprocess`.

        Returns
        -------
        float
            The metric of the Bayesian hyperparameter optimization iteration.
        """
        y_val_pred = self._pred_single_model(model, X_val, verbose=False)
        _, val_loss = self._default_metric_sklearn(y_val, y_val_pred)
        y_train_pred = self._pred_single_model(model, X_train, verbose=False)
        _, train_loss = self._default_metric_sklearn(y_train, y_train_pred)
        return max([train_loss, val_loss])

    def _check_train_status(self):
        """
        Raise exception if _predict is called and the model base is not trained.
        """
        if not self._trained:
            raise Exception(
                f"{self.program} not trained, run {self.__class__.__name__}.train() first."
            )

    def _get_params(self, model_name: str, verbose=True) -> Dict[str, Any]:
        """
        Load default parameters or optimized parameters (if Bayesian optimization is performed) of the selected model.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity

        Returns
        -------
        dict
            A dict of parameters that contains all keys in :meth:`_initial_values`.
        """
        if model_name not in self.model_params.keys():
            return self._initial_values(model_name=model_name)
        else:
            if verbose:
                print(f"Previous params loaded: {self.model_params[model_name]}")
            return self.model_params[model_name]

    @property
    def _trained(self) -> bool:
        """
        True if :meth:`train` has been called, otherwise False.
        """
        if self.model is None:
            return False
        else:
            return True

    @property
    def _support_warm_start(self) -> bool:
        """
        If the model base cannot finetune a model, this is set to False.
        """
        return True

    def _check_space(self):
        """
        Check if all parameters defined in :meth:`_initial_values` have corresponding search spaces defined in
        :meth:`_space`.
        """
        any_mismatch = False
        for model_name in self.get_model_names():
            tmp_params = self._get_params(model_name, verbose=False)
            space = self._space(model_name=model_name)
            if len(space) == 0:
                continue
            not_exist = [s.name for s in space if s.name not in tmp_params.keys()]
            if len(not_exist) > 0:
                print(
                    f"{not_exist} are defined for {self.program} - {model_name} in _space but are not defined in "
                    f"_initial_values."
                )
                any_mismatch = True
        if any_mismatch:
            raise Exception(f"Defined spaces and initial values do not match.")

    def _mkdir(self):
        """
        Create a directory for the model base under the root of the linked :class:`~tabensemb.trainer.Trainer`.
        """
        self.root = os.path.join(self.trainer.project_root, self.program)
        if not os.path.exists(self.root):
            os.mkdir(self.root)

    def get_model_names(self) -> List[str]:
        """
        Get names of available models based on :meth:`_get_model_names` and the arguments ``model_subset`` or
        ``exclude_models`` of :meth:`__init__`.

        Returns
        -------
        list
            Names of available models.
        """
        if self.model_subset is not None:
            for model in self.model_subset:
                if model not in self._get_model_names():
                    raise Exception(f"Model {model} not available for {self.program}.")
            res = self.model_subset
        elif self.exclude_models is not None:
            names = self._get_model_names()
            used_names = [x for x in names if x not in self.exclude_models]
            res = used_names
        else:
            res = self._get_model_names()
        res = [x for x in res if self._conditional_validity(x)]
        return res

    @staticmethod
    def _get_model_names() -> List[str]:
        """
        Get names of all available models implemented in the model base.

        Returns
        -------
        list
            Names of available models.
        """
        raise NotImplementedError

    def _get_program_name(self) -> str:
        """
        Get the default name of the model base.

        Returns
        -------
        str
            The default name of the model base.
        """
        raise NotImplementedError

    # The following methods are for the default _train and _predict methods. If users directly overload _train and
    # _predict, the following methods are not required to be implemented.
    def _new_model(self, model_name: str, verbose: bool, **kwargs):
        """
        Generate a new selected model based on kwargs.

        Parameters
        ----------
        model_name:
            The name of a selected model.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to generate the model returned by :meth:`_get_params`. It contains all arguments in
            :meth:`_initial_values`. If any model is required, which is defined in :meth:`required_models`, there will
            be a named argument "required_models" containing required models extracted by :meth:`_get_required_models`.

        Returns
        -------
        model:
            A new model (without any restriction to its type). It will be passed to :meth:`_train_single_model` and
            :meth:`_pred_single_model`.

        See Also
        --------
        :meth:`new_model`
        """
        raise NotImplementedError

    def _train_data_preprocess(self, model_name) -> Union[DataModule, dict]:
        """
        Processing the data from ``self.trainer.datamodule`` for training.

        Parameters
        ----------
        model_name:
            The name of a selected model.

        Returns
        -------
        dict
            A dictionary that has the following keys: X_train, y_train, X_val, y_val, X_test, y_test.
            Those with postfixes ``_train`` or ``_val`` will be passed to :meth:`_train_single_model` and
            :meth:`_bayes_eval`. All of them will be passed to :meth:`_pred_single_model` for evaluation.

        Notes
        -----
        ``self.trainer.datamodule.X_train/val/test`` are not scaled. To scale the df,
        run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        raise NotImplementedError

    def _data_preprocess(
        self, df: pd.DataFrame, derived_data: Dict[str, np.ndarray], model_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Perform the same preprocessing as in :meth:`_train_data_preprocess` on a new dataset.

        Parameters
        ----------
        df:
            The new tabular dataset that has the same structure as ``self.trainer.datamodule.X_test``
        derived_data:
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`. If None,
            unstacked data will be re-derived.
        model_name:
            The name of a selected model.

        Returns
        -------
        Any
            The processed data that has the same structure as X_test from :meth:`_train_data_preprocess`.

        Notes
        -----
        The input df is not scaled. To scale the df, run ``df = datamodule.data_transform(df, scaler_only=True)``
        """
        raise NotImplementedError

    def _train_single_model(
        self,
        model: Any,
        epoch: Optional[int],
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        y_val: Any,
        verbose: bool,
        warm_start: bool,
        in_bayes_opt: bool,
        **kwargs,
    ):
        """
        Training the model (initialized in :meth:`_new_model`).

        Parameters
        ----------
        model:
            The model returned by :meth:`_new_model`.
        epoch:
            Total epochs to train the model.
        X_train:
            The training data from :func:`_train_data_preprocess`.
        y_train:
            The training target from :func:`_train_data_preprocess`.
        X_val:
            The validation data from :func:`_train_data_preprocess`.
        y_val:
            The validation target from :func:`_train_data_preprocess`.
        verbose:
            Verbosity.
        warm_start:
            Finetune models based on previous trained models.
        in_bayes_opt:
            Whether is in a Bayesian optimization loop.
        **kwargs:
            Parameters to train the model returned by :meth:`_get_params`. It contains all arguments in
            :meth:`_initial_values`.
        """
        raise NotImplementedError

    def _pred_single_model(
        self, model: Any, X_test: Any, verbose: bool, **kwargs
    ) -> np.ndarray:
        """
        Predict using the model trained in :meth:`_train_single_model`.

        Parameters
        ----------
        model:
            The model returned by :meth:`_new_model` and trained in :meth:`_train_single_model`.
        X_test:
            The data from :meth:`_data_preprocess` or :meth:`_train_data_preprocess`.
        verbose:
            Verbosity.
        **kwargs:
            Parameters to train the model returned by :meth:`_get_params`. It contains all arguments in
            :meth:`_initial_values`.

        Returns
        -------
        np.ndarray
            Prediction of the target.

        Notes
        -----
        For deep learning models with mini-batch training (dataloaders), if an :class:`AbstractWrapper` will be used to extract
        hidden representations, the ``batch_size`` when inferring should be the length of the dataset. See
        :meth:`tabensemb.model.PytorchTabular._pred_single_model` and :meth:`tabensemb.model.WideDeep._pred_single_model`.
        """
        raise NotImplementedError

    def _space(self, model_name: str) -> List[Union[Integer, Real, Categorical]]:
        """
        A list of ``scikit-optimize`` search spaces for the selected model. It should contain all parameters
        defined in :meth:`_initial_values`.

        Parameters
        ----------
        model_name:
            The name of a selected model that is currently going through Bayesian optimization.

        Returns
        -------
        list
            A list of ``skopt.space``.
        """
        raise NotImplementedError

    def _initial_values(self, model_name: str) -> Dict[str, Union[int, float]]:
        """
        Initial values of hyperparameters to be optimized.

        Parameters
        ----------
        model_name:
            The name of a selected model.

        Returns
        -------
        dict
            A dict of initial hyperparameters.
        """
        raise NotImplementedError

    def _conditional_validity(self, model_name: str) -> bool:
        """
        Check the validity of a model.

        Parameters
        ----------
        model_name:
            The name of a model in :meth:`_get_model_names`.

        Returns
        -------
        bool
            Whether the model can be trained under certain settings.
        """
        return True


class BayesCallback:
    """
    Print information when performing Bayesian optimization.
    """

    def __init__(self, total):
        self.total = total
        self.cnt = 0
        self.init_time = time.time()
        self.postfix = {
            "ls": 1e8,
            "param": [],
            "min ls": 1e8,
            "min param": [],
            "min at": 0,
        }

    def call(self, result):
        self.postfix["ls"] = result.func_vals[-1]
        self.postfix["param"] = [
            round(x, 5) if hasattr(x, "__round__") else x for x in result.x_iters[-1]
        ]
        if result.fun < self.postfix["min ls"]:
            self.postfix["min ls"] = result.fun
            self.postfix["min param"] = [
                round(x, 5) if hasattr(x, "__round__") else x for x in result.x
            ]
            self.postfix["min at"] = len(result.func_vals)
        self.cnt += 1
        tot_time = time.time() - self.init_time
        print(
            f"Bayes-opt {self.cnt}/{self.total}, tot {tot_time:.2f}s, avg {tot_time/self.cnt:.2f}s/it: {self.postfix}"
        )

    def close(self):
        torch.cuda.empty_cache()


class TorchModel(AbstractModel):
    """
    The class for PyTorch-like models. Some abstract methods in :class:`AbstractModel` are implemented.
    """

    def cal_feature_importance(self, model_name, method, call_general_method=False):
        """
        Calculate feature importance using a specified model. ``captum`` or ``shap`` is called.

        Parameters
        ----------
        model_name
            The selected model in the model base.
        method
            The method to calculate importance. "permutation" or "shap".
        call_general_method
            Call the general feature importance calculation :meth:`AbstractModel.cal_feature_importance` instead of the
            optimized procedure for deep learning models. This is useful when calculating the feature importance of
            models that require other models.

        Returns
        -------
        attr
            Values of feature importance.
        importance_names
            Corresponding feature names. All features including derived unstacked features will be included.
        """
        if call_general_method:
            return super(TorchModel, self).cal_feature_importance(model_name, method)

        label_data = self.trainer.label_data
        test_indices = self.trainer.test_indices
        test_label = label_data.loc[test_indices, :].values
        trainer_datamodule = self.trainer.datamodule

        # This is decomposed from _data_preprocess (The first part)
        tensors, df, derived_data, custom_datamodule = self._prepare_tensors(
            trainer_datamodule.df.loc[test_indices, :],
            trainer_datamodule.get_derived_data_slice(
                trainer_datamodule.derived_data, test_indices
            ),
            model_name,
        )
        X = tensors[0]
        D = tensors[1:-1]
        y = tensors[-1]
        cont_feature_names = custom_datamodule.cont_feature_names
        cat_feature_names = custom_datamodule.cat_feature_names

        if method == "permutation":
            if self.required_models(model_name) is not None:
                warnings.warn(
                    f"Calculating permutation importance for models that require other models. Results of required "
                    f"models come from un-permuted data. If this is not acceptable, pass `call_general_method=True`."
                )

            def forward_func(_X, *_D):
                # This is decomposed from _data_preprocess (The second part)
                _tensors = (_X, *_D, torch.ones_like(y))
                dataset = self._generate_dataset_from_tensors(
                    _tensors, df, derived_data, model_name
                )
                # This is decomposed from _predict
                prediction = self._pred_single_model(
                    self.model[model_name],
                    X_test=dataset,
                    verbose=False,
                )
                loss = float(metric_sklearn(test_label, prediction, "mse"))
                return loss

            feature_perm = FeaturePermutation(forward_func)
            attr = [x.cpu().numpy().flatten() for x in feature_perm.attribute((X, *D))]
            attr = np.abs(np.concatenate(attr))
        elif method == "shap":
            attr = self.cal_shap(model_name=model_name)
        else:
            raise NotImplementedError
        dims = [x.shape for x in derived_data.values()]
        importance_names = cp(cont_feature_names)
        for key_idx, key in enumerate(derived_data.keys()):
            importance_names += (
                [
                    f"{key} (dim {i})" if dims[key_idx][-1] > 1 else key
                    for i in range(dims[key_idx][-1])
                ]
                if key != "categorical"
                else cat_feature_names
            )
        return attr, importance_names

    def cal_shap(self, model_name: str, call_general_method=False) -> np.ndarray:
        """
        Calculate SHAP values using a specified model. If the model base is a :class:`TorchModel`, the
        ``shap.DeepExplainer`` is used.

        Parameters
        ----------
        model_name
            The selected model in the model base.
        call_general_method
            Call the general shap calculation :meth:`AbstractModel.cal_shap` instead of the
            optimized procedure for deep learning models. This is useful when calculating the feature importance of
            models that require other models.

        Returns
        -------
        attr
            The SHAP values. All features including derived unstacked features will be included.
        """
        if self.required_models(model_name) is not None:
            raise Exception(
                f"Calculating shap for models that require other models is not supported, because "
                f"shap.DeepExplainer directly calls forward passing a series of tensors, and required models may "
                f"use DataFrames, NDArrays, etc. Pass `call_general_method=True` to use shap.KernelExplainer."
            )
        import shap

        train_indices = self.trainer.train_indices
        test_indices = self.trainer.test_indices
        datamodule = self.trainer.datamodule
        if "categorical" in datamodule.derived_data.keys():
            warnings.warn(
                f"shap.DeepExplainer cannot handle categorical features because their gradients (as float dtype) are "
                f"zero, and integers can not require_grad. If shap values of categorical values are needed, pass "
                f"`call_general_method=True` to use shap.KernelExplainer."
            )

        bk_indices = np.random.choice(
            train_indices,
            size=min([100, len(train_indices)]),
            replace=False,
        )
        tensors, _, _, _ = self._prepare_tensors(
            datamodule.df.loc[bk_indices, :],
            datamodule.get_derived_data_slice(datamodule.derived_data, bk_indices),
            model_name,
        )
        X_train_bk = tensors[0]
        D_train_bk = tensors[1:-1]
        background_data = [X_train_bk, *D_train_bk]

        tensors, _, _, _ = self._prepare_tensors(
            datamodule.df.loc[test_indices, :],
            datamodule.get_derived_data_slice(datamodule.derived_data, test_indices),
            model_name,
        )
        X_test = tensors[0]
        D_test = tensors[1:-1]
        test_data = [X_test, *D_test]

        with global_setting({"test_with_no_grad": False}):
            explainer = shap.DeepExplainer(self.model[model_name], background_data)

            with HiddenPrints():
                shap_values = explainer.shap_values(test_data)

        attr = (
            np.concatenate(
                [np.mean(np.abs(shap_values[0]), axis=0)]
                + [np.mean(np.abs(x), axis=0) for x in shap_values[1:]],
            )
            if type(shap_values) == list and len(shap_values) > 1
            else np.mean(np.abs(shap_values[0]), axis=0)
        )
        return attr

    def _train_data_preprocess(self, model_name):
        datamodule = self._prepare_custom_datamodule(model_name)
        datamodule.update_dataset()
        train_dataset, val_dataset, test_dataset = self._generate_dataset(
            datamodule, model_name
        )
        return {
            "X_train": train_dataset,
            "y_train": datamodule.y_train,
            "X_val": val_dataset,
            "y_val": datamodule.y_val,
            "X_test": test_dataset,
            "y_test": datamodule.y_test,
        }

    def _prepare_custom_datamodule(self, model_name: str) -> DataModule:
        """
        Change this method if a customized preprocessing stage is needed. See :class:`tabensemb.model.CatEmbed` for example.

        See Also
        --------
        :meth:`_run_custom_data_module`
        """
        return self.trainer.datamodule

    def _generate_dataset(self, datamodule: DataModule, model_name: str):
        """
        Generate ``torch.utils.data.Dataset`` for training.

        Parameters
        ----------
        datamodule
            The :class:`tabensemb.data.datamodule.DataModule` returned by :meth:`_prepare_custom_datamodule`.
        model_name
            The name of the selected model.

        Returns
        -------
        torch.utils.data.Dataset

        See Also
        --------
        :meth:`_generate_dataset_from_tensors`
        """
        required_models = self._get_required_models(model_name)
        if required_models is None:
            train_dataset, val_dataset, test_dataset = (
                datamodule.train_dataset,
                datamodule.val_dataset,
                datamodule.test_dataset,
            )
        else:
            dataset = self._generate_dataset_for_required_models(
                df=datamodule.df,
                derived_data=datamodule.derived_data,
                tensors=datamodule.tensors,
                required_models=required_models,
            )
            train_dataset, val_dataset, test_dataset = datamodule.generate_subset(
                dataset
            )
        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def get_full_name_from_required_model(required_model, model_name=None):
        """
        Get the name of a required model to store or access data in ``derived_tensors`` passed to
        :meth:`AbstractNN._forward`.

        Parameters
        ----------
        required_model
            A required model specified in :meth:`AbstractModel.required_models` and extracted by
            :meth:`AbstractModel._get_required_models`.
        model_name
            The name of the required model. It is necessary if the model comes from the same model base.

        Returns
        -------
        str
            The name of a required model
        """
        if isinstance(required_model, AbstractWrapper) or isinstance(
            required_model, AbstractModel
        ):
            name = required_model.get_model_names()[0]
            full_name = f"EXTERN_{required_model.program}_{name}"
        elif isinstance(required_model, nn.Module):
            if model_name is None:
                raise Exception(
                    f"If the required model comes from the same model base, `model_name` should be "
                    f"provided when calling `call_required_model.`"
                )
            full_name = model_name
        else:
            raise Exception(
                f"The required model should be a nn.Module, an AbstractWrapper, or an AbstractModel, but got"
                f"{type(required_model)} instead."
            )
        return full_name

    def _generate_dataset_for_required_models(
        self, df, derived_data, tensors, required_models
    ):
        """
        Call :meth:`AbstractModel._data_preprocess` to generate the dataset, output, and hidden representations for the
        required model

        Parameters
        ----------
        df
            The new tabular dataset that has the same structure as ``self.trainer.datamodule.X_test``
        derived_data
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`.
        tensors
            Tensors stored in a :class:`tabensemb.data.datamodule.DataModule` and obtained by
            :meth:`tabensemb.data.datamodule.DataModule.update_dataset`
        required_models
            Required models specified in :meth:`AbstractModel.required_models` and extracted by
            :meth:`AbstractModel._get_required_models`.

        Returns
        -------
        torch.utils.data.Dataset
        """
        full_data_required_models = {}
        for name, mod in required_models.items():
            full_name = TorchModel.get_full_name_from_required_model(
                mod, model_name=name
            )
            if not isinstance(mod, AbstractNN):
                data = mod._data_preprocess(
                    df=df,
                    derived_data=derived_data,
                    model_name=full_name.split("_")[-1],
                )
                full_data_required_models[full_name] = data
                res = AbstractNN.call_required_model(
                    mod, None, {"data_required_models": {full_name: data}}
                )
                if isinstance(res, torch.Tensor):
                    res = res.detach().to("cpu")
                full_data_required_models[full_name + "_pred"] = res
                if isinstance(mod, AbstractWrapper):
                    hidden = mod.hidden_representation.detach().to("cpu")
                    full_data_required_models[full_name + "_hidden"] = hidden
            else:
                mod.eval()
                with torch.no_grad():
                    res = mod(*tensors).detach().to("cpu")
                hidden = mod.hidden_representation.detach().to("cpu")
                full_data_required_models[full_name + "_pred"] = res
                full_data_required_models[full_name + "_hidden"] = hidden
        tensor_dataset = Data.TensorDataset(*tensors)
        dict_df_dataset = DictMixDataset(full_data_required_models)
        dataset = DictDataset(
            ListDataset([tensor_dataset, dict_df_dataset]),
            keys=["self", "required"],
        )
        return dataset

    def _run_custom_data_module(self, df, derived_data, model_name):
        """
        Change this method if a customized preprocessing stage is implemented in :meth:`_prepare_custom_datamodule`.
        See :class:`tabensemb.model.CatEmbed` for example.

        See Also
        --------
        :meth:`_prepare_custom_datamodule`
        """
        return df, derived_data, self.trainer.datamodule

    def _prepare_tensors(self, df, derived_data, model_name):
        """
        Transform the upcoming dataset into Tensors that has the same structures as those stored in a
        :class:`tabensemb.data.datamodule.DataModule` and obtained by
        :meth:`tabensemb.data.datamodule.DataModule.update_dataset`.

        Parameters
        ----------
        df
            The new tabular dataset that has the same structure as ``self.trainer.datamodule.X_test``
        derived_data
            Unstacked data derived from :meth:`tabensemb.data.datamodule.DataModule.derive_unstacked`.
        model_name
            The name of the selected model.

        Returns
        -------
        A tuple of torch.Tensor
            Transformed tensors.
        pd.DataFrame
            The transformed dataset after running :meth:`_run_custom_data_module`.
        dict
            The derived unstacked data after running :meth:`_run_custom_data_module`
        DataModule
            The :class:`tabensemb.data.datamodule.DataModule` returned by :meth:`_prepare_custom_data_module`

        See Also
        --------
        :meth:`tabensemb.data.datamodule.DataModule.update_dataset`
        """
        df, derived_data, datamodule = self._run_custom_data_module(
            df, derived_data, model_name
        )
        scaled_df = datamodule.data_transform(df, scaler_only=True)
        X, D, y = datamodule.generate_tensors(scaled_df, derived_data)
        tensors = (X, *D, y)
        return tensors, df, derived_data, datamodule

    def _generate_dataset_from_tensors(self, tensors, df, derived_data, model_name):
        """
        Perform the same preprocessing as in :meth:`_generate_dataset` on a new dataset.

        Parameters
        ----------
        tensors
            Tensors that has the same structures as those stored in a :class:`tabensemb.data.datamodule.DataModule` and
            obtained by :meth:`tabensemb.data.datamodule.DataModule.update_dataset`.
        df
            The transformed dataset after running :meth:`_run_custom_data_module`.
        derived_data
            The derived unstacked data after running :meth:`_run_custom_data_module`
        model_name
            The name of the selected model.

        Returns
        -------
        torch.utils.data.Dataset

        See Also
        --------
        :meth:`_generate_dataset`
        """
        required_models = self._get_required_models(model_name)
        if required_models is None:
            dataset = Data.TensorDataset(*tensors)
        else:
            dataset = self._generate_dataset_for_required_models(
                df=df.reset_index(drop=True),  # Use the unscaled one here
                derived_data=derived_data,
                tensors=tensors,
                required_models=required_models,
            )
        return dataset

    def _data_preprocess(self, df, derived_data, model_name):
        # In _train_data_preprocess:
        # 1. prepare_custom_datamodule + DataModule.update_dataset
        # 2. _generate_dataset using tensors in DataModule
        # In _data_preprocess
        # 0. Check the label(s).
        # 1. _prepare_tensors = _run_custom_data_module + update_dataset
        # 2. _generate_dataset_from_tensors is very similar to _generate_dataset, but does not split it into three parts.
        df = df.copy()
        for label in self.trainer.label_name:
            if label not in df.columns:
                # Just to create a placeholder for datamodule.generate_tensors.
                df[label] = np.zeros_like(self.trainer.df[label].values)[: len(df)]
        tensors, df, derived_data, _ = self._prepare_tensors(
            df, derived_data, model_name
        )
        dataset = self._generate_dataset_from_tensors(
            tensors, df, derived_data, model_name
        )
        return dataset

    def _train_single_model(
        self,
        model: "AbstractNN",
        epoch,
        X_train,
        y_train,
        X_val,
        y_val,
        verbose,
        warm_start,
        in_bayes_opt,
        **kwargs,
    ):
        """
        ``pytorch_lightning`` implementation of training a pytorch model.

        See Also
        --------
        :meth:`AbstractModel._train_single_model`
        """
        if not isinstance(model, AbstractNN):
            raise Exception(
                f"_new_model must return an AbstractNN instance, but got {model}."
            )

        warnings.filterwarnings(
            "ignore", "The dataloader, val_dataloader 0, does not have many workers"
        )
        warnings.filterwarnings(
            "ignore", "The dataloader, train_dataloader, does not have many workers"
        )
        warnings.filterwarnings("ignore", "Checkpoint directory")

        train_loader = Data.DataLoader(
            X_train,
            batch_size=int(kwargs["batch_size"]),
            sampler=torch.utils.data.RandomSampler(
                data_source=X_train, replacement=False
            ),
            pin_memory=True and self.device != "cpu",
        )
        val_loader = Data.DataLoader(
            X_val,
            batch_size=len(X_val),
            pin_memory=True and self.device != "cpu",
        )

        es_callback = EarlyStopping(
            monitor="early_stopping_eval",
            min_delta=0.001,
            patience=self.trainer.static_params["patience"],
            mode="min",
        )
        ckpt_callback = ModelCheckpoint(
            monitor="early_stopping_eval",
            dirpath=self.root,
            filename="early_stopping_ckpt",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
        )
        trainer = pl.Trainer(
            max_epochs=epoch,
            min_epochs=1,
            callbacks=[
                PytorchLightningLossCallback(verbose=True, total_epoch=epoch),
                es_callback,
                ckpt_callback,
            ],
            fast_dev_run=False,
            max_time=None,
            gpus=None,
            accelerator="cpu" if self.device == "cpu" else "auto",
            devices=None,
            accumulate_grad_batches=1,
            auto_lr_find=False,
            # auto_select_gpus=True,
            check_val_every_n_epoch=1,
            gradient_clip_val=0.0,
            overfit_batches=0.0,
            deterministic=False,
            profiler=None,
            logger=False,
            track_grad_norm=-1,
            precision=32,
            enable_checkpointing=True,
            enable_progress_bar=False,
        )

        ckpt_path = os.path.join(self.root, "early_stopping_ckpt.ckpt")
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

        with HiddenPrints(
            disable_std=not verbose,
            disable_logging=not verbose,
        ):
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

        model.to("cpu")
        model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
        trainer.strategy.remove_checkpoint(
            os.path.join(self.root, "early_stopping_ckpt.ckpt")
        )
        # pl.Trainer is not pickle-able. When pickling, "ReferenceError: weakly-referenced object no longer exists."
        # may be raised occasionally. Set the trainer to None.
        # https://deepforest.readthedocs.io/en/latest/FAQ.html
        model.trainer = None
        torch.cuda.empty_cache()

    def _pred_single_model(self, model: "AbstractNN", X_test, verbose, **kwargs):
        test_loader = Data.DataLoader(
            X_test,
            batch_size=len(X_test),
            shuffle=False,
            pin_memory=True and self.device != "cpu",
        )
        model.to(self.device)
        y_test_pred, _, _ = model.test_epoch(test_loader, **kwargs)
        model.to("cpu")
        torch.cuda.empty_cache()
        return y_test_pred

    def _space(self, model_name):
        return self.trainer.SPACE

    def _initial_values(self, model_name):
        return self.trainer.chosen_params

    def count_params(self, model_name, trainable_only=False):
        """
        Count the number of parameters in a ``torch.nn.Module``

        Parameters
        ----------
        model_name
            The name of the selected model
        trainable_only
            Only count trainable (requires_grad=True) parameters.

        Returns
        -------
        float
            The number of parameters
        """
        if self.model is not None and model_name in self.model.keys():
            model = self.model[model_name]
        else:
            self._prepare_custom_datamodule(model_name)
            model = self.new_model(
                model_name, verbose=False, **self._get_params(model_name, verbose=False)
            )
        return sum(
            p.numel()
            for p in model.parameters()
            if (p.requires_grad if trainable_only else True)
        )


class AbstractWrapper:
    """
    For those required deep learning models, this is a wrapper to make them have hidden information like
    ``hidden_representation`` or something else extracted from the forward process.

    Attributes
    ----------
    hidden_rep_dim
    hidden_representation
    """

    def __init__(self, model: AbstractModel):
        if len(model.get_model_names()) > 1:
            raise Exception(
                f"More than one model is included in the input model base: {model.get_model_names()}."
            )
        self.wrapped_model = model
        self.model_name = self.wrapped_model.get_model_names()[0]
        self.original_forward = None
        self.wrap_forward()

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            return getattr(self.wrapped_model, item)

    def eval(self):
        pass

    def __call__(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simulate ``AbstractNN._forward`` by calling ``AbstractNN.call_required_model``.
        """
        return AbstractNN.call_required_model(self.wrapped_model, x, derived_tensors)

    def wrap_forward(self):
        """
        Override the forward method of a torch.nn.Module to record hidden representations.
        """
        raise NotImplementedError

    def reset_forward(self):
        """
        Reset the overridden forward method of the torch.nn.Module to ensure pickling compatibility.
        """
        raise NotImplementedError

    @property
    def hidden_rep_dim(self):
        """
        The dimension of :meth:`hidden_representation`.
        """
        raise NotImplementedError

    @property
    def hidden_representation(self):
        """
        The extracted information of a deep learning model when forward-passing a batch. It is usually the input of the
        last output layer (usually a linear layer or an MLP).
        """
        raise NotImplementedError

    def __getstate__(self):
        self.reset_forward()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.wrap_forward()


class TorchModelWrapper(AbstractWrapper):
    def __init__(self, model: TorchModel):
        super(TorchModelWrapper, self).__init__(model=model)

    def wrap_forward(self):
        pass

    def reset_forward(self):
        pass

    @property
    def hidden_rep_dim(self):
        return self.wrapped_model.model[self.model_name].hidden_rep_dim

    @property
    def hidden_representation(self):
        return self.wrapped_model.model[self.model_name].hidden_representation


class AbstractNN(pl.LightningModule):
    """
    A subclass of ``pytorch_lightning.LightningModule`` that is compatible with :class:`TorchModel` and has implemented
    training and inferencing steps.

    Attributes
    ----------
    default_loss_fn
        The name of the default loss function returned by :meth:`get_loss_fn`
    default_output_norm
        The name of the default output normalization returned by :meth:`get_output_norm`
    cont_feature_names
        The names of continuous features
    cat_feature_names
        The names of categorical features
    n_cont
        The number of continuous features
    n_cat
        The number of categorical features
    derived_feature_names
        The keys of derived unstacked features.
    derived_feature_dims
        The dimensions of derived unstacked features
    task
        "regression", "binary", or "multiclass"
    n_outputs
        The number of outputs. Note that for classification tasks, logits are returned instead of probabilities.
        For binary classification, the logit for the positive class is returned.
    cat_num_unique
        The number of unique values for each categorical feature.
    hidden_representation
        The extracted information of a deep learning model when forward-passing a batch.
        It is usually the input of the last output layer (usually a linear layer or an MLP). It should be manually
        recorded in :meth:`_forward`.
    hidden_rep_dim
        The dimension of :attr:`hidden_representation`. It should be manually set in :meth:`__init__`.
    device
    training
    """

    def __init__(self, datamodule: DataModule, **kwargs):
        """
        Record useful information for initializing and training models.

        Parameters
        ----------
        datamodule:
            A :class:`tabensemb.data.datamodule.DataModule` instance.
        """
        super(AbstractNN, self).__init__()
        self.default_loss_fn = self.get_loss_fn(datamodule.loss, datamodule.task)
        self.default_output_norm = self.get_output_norm(datamodule.task)
        self.cont_feature_names = cp(datamodule.cont_feature_names)
        self.cat_feature_names = cp(datamodule.cat_feature_names)
        self.n_cont = len(self.cont_feature_names)
        self.n_cat = len(self.cat_feature_names)
        self.derived_feature_names = list(datamodule.derived_data.keys())
        self.derived_feature_dims = datamodule.get_derived_data_sizes()
        self.derived_feature_names_dims = {}
        self.automatic_optimization = False
        self.hidden_representation = None
        self.hidden_rep_dim = None
        self.task = datamodule.task
        self.n_inputs = len(self.cont_feature_names)
        task_outputs = {
            "regression": len(datamodule.label_name),
            "multiclass": datamodule.n_classes[0],
            "binary": 1,
        }
        if self.task in task_outputs.keys():
            self.n_outputs = task_outputs[self.task]
        else:
            raise Exception(f"Unsupported type of task {self.task}")
        self.cat_num_unique = datamodule.cat_num_unique
        if len(kwargs) > 0:
            self.save_hyperparameters(
                *list(kwargs.keys()),
                ignore=["trainer", "datamodule", "required_models"],
            )
        for name, dim in zip(
            datamodule.derived_data.keys(), datamodule.get_derived_data_sizes()
        ):
            self.derived_feature_names_dims[name] = dim
        self._device_var = nn.Parameter(torch.empty(0, requires_grad=False))

    @staticmethod
    def get_output_norm(task) -> torch.nn.Module:
        """
        The operation on the output of ``forward`` in training/validation/testing steps. This will not affect the input
        of loss functions.

        Parameters
        ----------
        task
            "regression", "multiclass", or "binary"

        Returns
        -------
        nn.Module
            The operation on the output.
        """
        task_norm = {
            "regression": torch.nn.Identity(),
            "multiclass": torch.nn.Softmax(dim=-1),
            "binary": torch.nn.Sigmoid(),
        }
        if task in task_norm.keys():
            return task_norm[task]
        else:
            raise Exception(f"Unrecognized task {task}.")

    @staticmethod
    def get_loss_fn(loss, task) -> torch.nn.Module:
        """
        The loss function for the output of ``forward`` and the target.

        Parameters
        ----------
        loss
            "cross_entropy", "mae", or "mse"
        task
            "regression", "multiclass", or "binary"

        Returns
        -------
        nn.Module
            The loss function.
        """
        if task in ["binary", "multiclass"] and loss != "cross_entropy":
            raise Exception(
                f"Only cross entropy loss is supported for classification tasks."
            )
        if task == "binary":
            return torch.nn.BCEWithLogitsLoss()
        elif task == "multiclass":
            return torch.nn.CrossEntropyLoss()
        elif task == "regression":
            mapping = {
                "mse": torch.nn.MSELoss(),
                "mae": torch.nn.L1Loss(),
            }
            return mapping[loss]
        else:
            raise Exception(f"Unrecognized task {task}.")

    @property
    def device(self):
        """
        The device where the model is.
        """
        return self._device_var.device

    def forward(
        self,
        *tensors: torch.Tensor,
        data_required_models: Dict[str, pd.DataFrame] = None,
    ) -> torch.Tensor:
        """
        A wrapper of the original forward of ``nn.Module`` for compatibility concerns.

        Parameters
        ----------
        tensors:
            Input tensors to the torch model. They have the same structures as the ``tensors`` stored in
            :meth:`tabensemb.data.datamodule.DataModule`
        data_required_models:
            The datasets for required models processed by their own :meth:`AbstractModel._train_data_preprocess` or
            :meth:`AbstractModel._data_preprocess` methods. See :meth:`TorchModel._generate_dataset_for_required_models`

        Returns
        -------
        torch.Tensor
            The output from :meth:`_forward`.
        """
        with torch.no_grad() if tabensemb.setting[
            "test_with_no_grad"
        ] and not self.training else torch_with_grad():
            x = tensors[0]
            additional_tensors = tensors[1:]
            if len(additional_tensors) > 0 and type(additional_tensors[0]) == dict:
                derived_tensors = additional_tensors[0]
            else:
                derived_tensors = {}
                for tensor, name in zip(additional_tensors, self.derived_feature_names):
                    derived_tensors[name] = tensor
            if data_required_models is not None:
                derived_tensors["data_required_models"] = data_required_models
            res = self._forward(x, derived_tensors)
            if len(res.shape) == 1:
                res = res.view(-1, 1)
            if self.hidden_representation is None:
                self.hidden_representation = res
            if self.hidden_rep_dim is None:
                self.hidden_rep_dim = res.shape[1]
            return res

    def _forward(
        self, x: torch.Tensor, derived_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        The real forward method.

        Parameters
        ----------
        x
            A tensor that contains continuous features.
        derived_tensors
            It mostly has the same structure as the derived unstacked data ``derived_data`` stored in a
            :class:`tabensemb.data.datamodule.DataModule`. If some models are required (defined in
            :meth:`AbstractModel.required_models`), there will be a key named "data_required_models" containing the
            data batch, the output, and possibly the hidden representation of the required models.

        Returns
        -------
        torch.Tensor
            The output of the model.

        Notes
        -----
        For classification tasks, DO NOT turn logits into probabilities here because we have already
        implemented this later. See also :meth:`output_norm`.
        """
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: Any):
        if type(batch) == dict:
            tensors, data_required_models = batch["self"], batch["required"]
        else:
            tensors, data_required_models = batch, None
        self.cal_zero_grad()
        yhat = tensors[-1]
        data = tensors[0]
        additional_tensors = [x for x in tensors[1 : len(tensors) - 1]]
        y = self(
            *([data] + additional_tensors), data_required_models=data_required_models
        )
        y, yhat = self.before_loss_fn(y, yhat)
        loss = self.loss_fn(y, yhat, *([data] + additional_tensors))
        self.cal_backward_step(loss)
        default_loss = self.default_loss_fn(y, yhat)
        self.log(
            "train_loss_verbose",
            default_loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=y.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) == dict:
            tensors, data_required_models = batch["self"], batch["required"]
        else:
            tensors, data_required_models = batch, None
        with torch.no_grad():
            yhat = tensors[-1]
            data = tensors[0]
            additional_tensors = [x for x in tensors[1 : len(tensors) - 1]]
            y = self(
                *([data] + additional_tensors),
                data_required_models=data_required_models,
            )
            y, yhat = self.before_loss_fn(y, yhat)
            y_out = self.output_norm(y)
            loss = self.default_loss_fn(y, yhat)
            self.log(
                "valid_loss_verbose",
                loss.item(),
                on_step=False,
                on_epoch=True,
                batch_size=y.shape[0],
            )
        return yhat, y_out

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def test_epoch(
        self, test_loader: Data.DataLoader, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate a torch.nn.Module model in a single epoch.

        Parameters
        ----------
        test_loader:
            The ``DataLoader`` of the testing dataset.
        **kwargs:
            Parameters to train the model returned by :meth:`AbstractModel._get_params`. It contains all arguments in
            :meth:`AbstractModel._initial_values`.

        Returns
        -------
        np.ndarray
            The prediction. Always a 2d ``torch.Tensor``.
        np.ndarray
            The ground truth. Always a 2d ``torch.Tensor``.
        float
            The default loss :meth:`get_loss_fn` of the model on the testing dataset.
        """
        self.eval()
        pred = []
        truth = []
        with torch.no_grad() if tabensemb.setting[
            "test_with_no_grad"
        ] else torch_with_grad():
            # print(test_dataset)
            avg_loss = 0
            for idx, batch in enumerate(test_loader):
                if type(batch) == dict:
                    tensors, data_required_models = batch["self"], batch["required"]
                else:
                    tensors, data_required_models = batch, None
                yhat = tensors[-1].to(self.device)
                data = tensors[0].to(self.device)
                additional_tensors = [
                    x.to(self.device) for x in tensors[1 : len(tensors) - 1]
                ]
                y = self(
                    *([data] + additional_tensors),
                    data_required_models=data_required_models,
                )
                y, yhat = self.before_loss_fn(y, yhat)
                y_out = self.output_norm(y)
                loss = self.default_loss_fn(y, yhat)
                avg_loss += loss.item() * len(y)
                pred.append(y_out.cpu().detach().numpy())
                truth.append(yhat.cpu().detach().numpy())
            avg_loss /= len(test_loader.dataset)
        all_pred = np.concatenate(pred, axis=0)
        all_truth = np.concatenate(truth, axis=0)
        if len(all_pred.shape) == 1:
            all_pred = all_pred.reshape(-1, 1)
        if len(all_truth.shape) == 1:
            all_truth = all_truth.reshape(-1, 1)
        return all_pred, all_truth, avg_loss

    def before_loss_fn(self, y, yhat):
        """
        Treatments on the prediction and the ground truth before passing them to :meth:`loss_fn`.

        Parameters
        ----------
        y
            The prediction from :meth:`forward`.
        yhat
            The ground truth.

        Returns
        -------
        torch.Tensor
            The processed prediction
        torch.Tensor
            The processed ground truth
        """
        if self.task == "binary":
            y = torch.flatten(y)
            yhat = torch.flatten(yhat)
        elif self.task == "multiclass":
            yhat = torch.flatten(yhat).long()
        return y, yhat

    def loss_fn(self, y_pred, y_true, *data, **kwargs):
        """
        User defined loss function.

        Parameters
        ----------
        y_true:
            The ground truth.
        y_pred:
            The predictions from the model (from :meth:`forward` and after :meth:`before_loss_fn`).
        *data:
            Tensors of continuous data and derived unstacked data.
        **kwargs:
            Parameters to train the model returned by :meth:`AbstractModel._get_params`. It contains all arguments in
            :meth:`AbstractModel._initial_values`.

        Returns
        -------
        torch.Tensor
            A torch-like loss.

        Notes
        -----
        Other attributes in ``self`` can also be used to calculate loss values.
        """
        return self.default_loss_fn(y_pred, y_true)

    def output_norm(self, y_pred):
        """
        User defined operation before output. This is not related to the input of :meth:`loss_fn`.

        Parameters
        ----------
        y_pred
            The prediction from the model (from :meth:`forward` and after :meth:`before_loss_fn`).

        Returns
        -------
        torch.Tensor
            The modified prediction.
        """
        return self.default_output_norm(y_pred)

    def cal_zero_grad(self):
        """
        Call zero_grad of optimizers initialized in :meth:`configure_optimizers`.
        """
        opt = self.optimizers()
        if isinstance(opt, list):
            for o in opt:
                o.zero_grad()
        else:
            opt.zero_grad()

    def cal_backward_step(self, loss):
        """
        Perform the backward propagation and optimization steps.

        Parameters
        ----------
        loss
            The loss returned by :meth:`loss_fn`.

        Notes
        -----
        Other attributes recorded in :meth:`loss_fn` can be also used.
        """
        self.manual_backward(loss)
        opt = self.optimizers()
        opt.step()

    def set_requires_grad(
        self, model: nn.Module, requires_grad: bool = None, state=None
    ):
        """
        Set or reset requires_grad states of a ``nn.Module``.

        Parameters
        ----------
        model
            A ``nn.Module`` model.
        requires_grad
            The requires_grad state for all parameters in the model.
        state
            The recorded state when calling this method with the argument ``required_grad`` given.

        Returns
        -------
        list
            The recorded state that can be used as the argument "state" to restore requires_grad states of the same
            model. Returned when the argument ``requires_grad`` is given.
        """
        if (requires_grad is None and state is None) or (
            requires_grad is not None and state is not None
        ):
            raise Exception(
                f"One of `requires_grad` and `state` should be specified to determine the action. If `requires_grad` is "
                f"not None, requires_grad of all parameters in the model is set. If state is not None, state of "
                f"requires_grad in the model is restored."
            )
        if state is not None:
            for s, param in zip(state, model.parameters()):
                param.requires_grad_(s)
        else:
            state = []
            for param in model.parameters():
                state.append(param.requires_grad)
                param.requires_grad_(requires_grad)
            return state

    def _early_stopping_eval(self, train_loss: float, val_loss: float) -> float:
        """
        Calculate the loss value (criteria) for early stopping. The validation loss is returned, but note that
        ``0.0 * train_loss`` is added to the returned value so that NaNs in the training set can be detected by
        ``EarlyStopping``.

        Parameters
        ----------
        train_loss
            The training loss from :attr:`default_loss_fn` of the epoch.
        val_loss
            The validation loss from :attr:`default_loss_fn` of the epoch.

        Returns
        -------
        float
            The early stopping evaluation.
        """
        return val_loss + 0.0 * train_loss

    @staticmethod
    def _test_required_model(
        n_inputs: int,
        required_model: Union[AbstractModel, "AbstractNN", AbstractWrapper],
    ) -> Tuple[bool, int]:
        """
        Test whether a required model has the attribute ``hidden_rep_dim`` and find its value.

        Parameters
        ----------
        n_inputs
            The dimension of the input (i.e. the ``x`` of :meth:`_forward`)
        required_model
            A required model specified in :meth:`AbstractModel.required_models` and extracted by
            :meth:`AbstractModel._get_required_models`.

        Returns
        -------
        bool
            Whether the required model has the attribute ``hidden_rep_dim``
        int
             The dimension of the hidden representation. If the required model does not have the attribute
             ``hidden_rep_dim``, ``1+n_inputs`` is returned.

        Notes
        -----
        For an ``AbstractNN``, whether the hidden representation (:attr:`hidden_representation`) is recorded is not
        guaranteed.
        """
        if isinstance(required_model, AbstractWrapper):
            hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            use_hidden_rep = True
        elif not hasattr(required_model, "hidden_representation") or not hasattr(
            required_model, "hidden_rep_dim"
        ):
            if not hasattr(required_model, "hidden_rep_dim"):
                print(
                    f"`hidden_rep_dim` is not given. The output of the backbone and the input features are used instead."
                )
                hidden_rep_dim = 1 + n_inputs
            else:
                hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            if not hasattr(required_model, "hidden_representation") or not hasattr(
                required_model, "hidden_rep_dim"
            ):
                print(
                    f"The backbone should have an attribute called `hidden_representation` that records the "
                    f"final output of the hidden layer, and `hidden_rep_dim` that records the dim of "
                    f"`hidden_representation`. Now the output of the backbone is used instead."
                )
            use_hidden_rep = False
        else:
            hidden_rep_dim = getattr(required_model, "hidden_rep_dim")
            use_hidden_rep = True
        return use_hidden_rep, hidden_rep_dim

    @staticmethod
    def call_required_model(
        required_model, x, derived_tensors, model_name=None
    ) -> torch.Tensor:
        """
        Call a required model and return its result. Predictions and hidden representations are generated before
        training using this method.

        Parameters
        ----------
        required_model
            A required model specified in :meth:`AbstractModel.required_models` and extracted by
            :meth:`AbstractModel._get_required_models`.
        x
            See :meth:`_forward`.
        derived_tensors
            See :meth:`_forward`.
        model_name
            The name of the required model. It is necessary if the model comes from the same model base.

        Returns
        -------
        torch.Tensor
            The result of the required model.

        Notes
        -----
        If you want to run the required model and further train it, pass a copied
        ``derived_tensors`` after removing the ``{MODEL_NAME}_pred`` item in its ``data_required_models`` item.
        """
        device = x.device if x is not None else "cpu"
        full_name = TorchModel.get_full_name_from_required_model(
            required_model, model_name
        )
        if full_name + "_pred" in derived_tensors["data_required_models"].keys():
            dl_pred = derived_tensors["data_required_models"][full_name + "_pred"][
                0
            ].to(device)
        else:
            dl_pred = None

        if dl_pred is None:
            # This will only happen when generating datasets before training.
            if isinstance(required_model, nn.Module) or isinstance(
                required_model, AbstractWrapper
            ):
                required_model.eval()
                dl_pred = required_model(x, derived_tensors)
            elif isinstance(required_model, AbstractModel):
                # _pred_single_model might disturb random sampling of dataloaders because
                # in torch.utils.data._BaseDataLoaderIter.__init__, the following line uses random:
                # self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
                name = required_model.get_model_names()[0]
                ml_pred = required_model._pred_single_model(
                    required_model.model[name],
                    X_test=derived_tensors["data_required_models"][full_name],
                    verbose=False,
                )
                dl_pred = torch.tensor(ml_pred, device=device)

        return dl_pred

    @staticmethod
    def get_hidden_state(
        required_model, x, derived_tensors, model_name=None
    ) -> Union[torch.Tensor, None]:
        """
        The input of the last layer of a deep learning model, i.e. the hidden representation, whose dimension is
        (batch_size, required_model.hidden_rep_dim). The definition can be different for different models, depending on
        the different implementations of :class:`AbstractWrapper` for different model bases.

        Parameters
        ----------
        required_model
            A required model specified in :meth:`AbstractModel.required_models` and extracted by
            :meth:`AbstractModel._get_required_models`.
        x
            See :meth:`_forward`.
        derived_tensors
            See :meth:`_forward`.
        model_name
            The name of the required model. It is necessary if the model comes from the same model base.

        Returns
        -------
        torch.Tensor
            The input of the last layer of a deep learning model.
        """
        device = x.device if x is not None else "cpu"
        full_name = TorchModel.get_full_name_from_required_model(
            required_model, model_name=model_name
        )
        if full_name + "_hidden" in derived_tensors["data_required_models"].keys():
            hidden = derived_tensors["data_required_models"][full_name + "_hidden"][
                0
            ].to(device)
        else:
            hidden = required_model.hidden_representation.to(device)
        return hidden


class ModelDict:
    def __init__(self, path):
        self.root = path
        self.model_path = {}

    def __setitem__(self, key, value):
        self.model_path[key] = os.path.join(self.root, key) + ".pkl"
        with open(self.model_path[key], "wb") as file:
            pickle.dump((key, value), file, pickle.HIGHEST_PROTOCOL)
        del value
        torch.cuda.empty_cache()

    def __getitem__(self, item):
        torch.cuda.empty_cache()
        with open(self.model_path[item], "rb") as file:
            key, model = pickle.load(file)
        return model

    def __len__(self):
        return len(self.model_path)

    def keys(self):
        return self.model_path.keys()


def init_weights(m, nonlinearity="leaky_relu"):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)


class AdaptiveDropout(nn.Module):
    keep_dropout = False
    global_p = None

    def __init__(self, p):
        super(AdaptiveDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(
            x,
            p=self.p if AdaptiveDropout.global_p is None else AdaptiveDropout.global_p,
            training=self.training or AdaptiveDropout.keep_dropout,
        )

    @classmethod
    def set(cls, state: bool):
        cls.keep_dropout = state


class KeepDropout:
    def __init__(self, p=None):
        self.p = p

    def __enter__(self):
        self.state = AdaptiveDropout.keep_dropout
        AdaptiveDropout.set(True)
        if self.p is not None:
            AdaptiveDropout.global_p = self.p

    def __exit__(self, exc_type, exc_val, exc_tb):
        AdaptiveDropout.set(self.state)
        AdaptiveDropout.global_p = None


def get_sequential(
    layers,
    n_inputs,
    n_outputs,
    act_func,
    dropout=0,
    use_norm=True,
    norm_type="batch",
    out_activate=False,
    out_norm_dropout=False,
    adaptive_dropout=False,
):
    net = nn.Sequential()
    if norm_type == "batch":
        norm = nn.BatchNorm1d
    elif norm_type == "layer":
        norm = nn.LayerNorm
    else:
        raise Exception(f"Normalization {norm_type} not implemented.")
    if act_func == nn.ReLU:
        nonlinearity = "relu"
    elif act_func == nn.LeakyReLU:
        nonlinearity = "leaky_relu"
    else:
        nonlinearity = "leaky_relu"
    if adaptive_dropout:
        dp = AdaptiveDropout
    else:
        dp = nn.Dropout
    if len(layers) > 0:
        if use_norm:
            net.add_module(f"norm_0", norm(n_inputs))
        net.add_module(
            "input", get_linear(n_inputs, layers[0], nonlinearity=nonlinearity)
        )
        net.add_module("activate_0", act_func())
        if dropout != 0:
            net.add_module(f"dropout_0", dp(dropout))
        for idx in range(1, len(layers)):
            if use_norm:
                net.add_module(f"norm_{idx}", norm(layers[idx - 1]))
            net.add_module(
                str(idx),
                get_linear(layers[idx - 1], layers[idx], nonlinearity=nonlinearity),
            )
            net.add_module(f"activate_{idx}", act_func())
            if dropout != 0:
                net.add_module(f"dropout_{idx}", dp(dropout))
        if out_norm_dropout and use_norm:
            net.add_module(f"norm_out", norm(layers[-1]))
        net.add_module(
            "output", get_linear(layers[-1], n_outputs, nonlinearity=nonlinearity)
        )
        if out_activate:
            net.add_module("activate_out", act_func())
        if out_norm_dropout and dropout != 0:
            net.add_module(f"dropout_out", dp(dropout))
    else:
        if use_norm:
            net.add_module("norm", norm(n_inputs))
        net.add_module("single_layer", nn.Linear(n_inputs, n_outputs))
        net.add_module("activate", act_func())
        if dropout != 0:
            net.add_module("dropout", dp(dropout))

    net.apply(partial(init_weights, nonlinearity=nonlinearity))
    return net


def get_linear(n_inputs, n_outputs, nonlinearity="leaky_relu"):
    linear = nn.Linear(n_inputs, n_outputs)
    init_weights(linear, nonlinearity=nonlinearity)
    return linear


class PytorchLightningLossCallback(Callback):
    def __init__(self, verbose, total_epoch):
        super(PytorchLightningLossCallback, self).__init__()
        self.val_ls = []
        self.es_val_ls = []
        self.verbose = verbose
        self.total_epoch = total_epoch
        self.start_time = 0

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        train_loss = logs["train_loss_verbose"].detach().cpu().numpy()
        val_loss = logs["valid_loss_verbose"].detach().cpu().numpy()
        self.val_ls.append(val_loss)
        if hasattr(pl_module, "_early_stopping_eval"):
            early_stopping_eval = pl_module._early_stopping_eval(
                trainer.logged_metrics["train_loss_verbose"],
                trainer.logged_metrics["valid_loss_verbose"],
            ).item()
            pl_module.log("early_stopping_eval", early_stopping_eval)
            self.es_val_ls.append(early_stopping_eval)
        else:
            early_stopping_eval = None
        epoch = trainer.current_epoch
        if (
            (epoch + 1) % tabensemb.setting["verbose_per_epoch"] == 0 or epoch == 0
        ) and self.verbose:
            if early_stopping_eval is not None:
                print(
                    f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                    f"Min val loss: {np.min(self.val_ls):.4f}, Min ES val loss: {np.min(self.es_val_ls):.4f}, "
                    f"Epoch time: {time.time()-self.start_time:.3f}s."
                )
            else:
                print(
                    f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                    f"Min val loss: {np.min(self.val_ls):.4f}, Epoch time: {time.time() - self.start_time:.3f}s."
                )


class DataFrameDataset(Data.Dataset):
    def __init__(self, df: pd.DataFrame):
        # If predicting for a new dataframe, the index might be a mess.
        self.df = df.reset_index(drop=True)
        self.df_dict = {
            key: row[1] for key, row in zip(self.df.index, self.df.iterrows())
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df_dict[item]


class NDArrayDataset(Data.Dataset):
    def __init__(self, array: np.ndarray):
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, item):
        return self.array[item]


class SubsetDataset(Data.Dataset):
    def __init__(self, dataset: Data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return Data.Subset(self.dataset, [item])


class ListDataset(Data.Dataset):
    def __init__(self, datasets: List[Data.Dataset]):
        self.datasets = datasets
        for dataset in self.datasets:
            if len(dataset) != len(self.datasets[0]):
                raise Exception(f"All datasets should have the equal length.")

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item):
        return [dataset.__getitem__(item) for dataset in self.datasets]


class DictDataset(Data.Dataset):
    def __init__(self, ls_dataset: ListDataset, keys: List[str]):
        self.keys = keys
        self.datasets = ls_dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return {
            key: data for key, data in zip(self.keys, self.datasets.__getitem__(item))
        }


class DictDataFrameDataset(DictDataset):
    def __init__(self, dict_dfs: Dict[str, pd.DataFrame]):
        keys = list(dict_dfs.keys())
        df_ls = list(dict_dfs.values())
        ls_dataset = ListDataset([DataFrameDataset(df) for df in df_ls])
        super(DictDataFrameDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)


class DictNDArrayDataset(DictDataset):
    def __init__(self, dict_array: Dict[str, np.ndarray]):
        keys = list(dict_array.keys())
        array_ls = list(dict_array.values())
        ls_dataset = ListDataset([NDArrayDataset(array) for array in array_ls])
        super(DictNDArrayDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)


class DictMixDataset(DictDataset):
    def __init__(self, dict_mix: Dict[str, Union[pd.DataFrame, np.ndarray]]):
        keys = list(dict_mix.keys())
        item_ls = list(dict_mix.values())
        ls_data = []
        for item in item_ls:
            if isinstance(item, pd.DataFrame):
                ls_data.append(DataFrameDataset(item))
            elif isinstance(item, np.ndarray):
                ls_data.append(NDArrayDataset(item))
            elif isinstance(item, torch.Tensor):
                ls_data.append(Data.TensorDataset(item))
            elif isinstance(item, Data.Dataset):
                ls_data.append(SubsetDataset(item))
            else:
                raise Exception(
                    f"Generating a mixed type dataset for type {type(item)}."
                )

        ls_dataset = ListDataset(ls_data)
        super(DictMixDataset, self).__init__(ls_dataset=ls_dataset, keys=keys)
