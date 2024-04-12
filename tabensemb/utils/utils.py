"""
All utilities used in the project.
"""

import os
import os.path
import sys
import warnings
import logging
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
import torch.optim
from distutils.spawn import find_executable
from importlib import import_module, reload
from functools import partialmethod, partial
import itertools
from copy import deepcopy as cp
from torch.autograd.grad_mode import _DecoratorContextManager
from typing import Any
import tabensemb
from typing import Dict
from sklearn.metrics import *
from io import StringIO

sns.reset_defaults()
# matplotlib.use("Agg")
if find_executable("latex") and tabensemb.setting["matplotlib_usetex"]:
    matplotlib.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True

global_sns_palette = sns.color_palette("deep")

global_palette = [
    "#ea232c",
    "#314190",
    "#2caf91",
    "#e49633",
    "#166135",
    "#50b9aa",
    "#0d5a9b",
    "#fcb760",
    "#6a77ac",
    "#322051",
    "#c24135",
    "#aa602d",
    "#eea86f",
    "#c56f9c",
    "#cd552e",
    "#ebde4e",
    "#96235d",
    "#f8b68a",
    "#c0e3df",
    "#000000",
    "#662b3d",
    "#eb3882",
    "#1a8e7c",
    "#a89351",
    "#a6cf79",
    "#f6d761",
    "#50abde",
]

global_marker = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "D", "d"]


def is_notebook() -> bool:
    """
    Check whether the current environment is a notebook.

    Returns
    -------
    bool
        True if in a notebook.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def set_random_seed(seed=0):
    """
    Set random seeds of pytorch (including cuda and dataloaders), numpy, and random.

    Parameters
    ----------
    seed
        The random seed.
    """
    set_torch(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    """
    For the argument ``worker_init_fn`` of ``torch.utils.data.DataLoader``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_torch(seed=0):
    """
    Set the random seed of pytorch, CUDA, and ``torch.utils.data.DataLoader``.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    dl = reload_module("torch.utils.data").DataLoader

    if not dl.__init__.__name__ == "_method":
        # Actually, setting generator improves reproducibility, but torch._C.Generator does not support pickling.
        # https://pytorch.org/docs/stable/notes/randomness.html
        # https://github.com/pytorch/pytorch/issues/43672
        dl.__init__ = partialmethod(dl.__init__, worker_init_fn=seed_worker)


def metric_sklearn(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """
    Calculate metrics using ``sklearn`` APIs. The format of ``y_true`` and ``y_pred`` should follow the requirement of
    ``metric`` (See https://scikit-learn.org/stable/modules/model_evaluation.html), so we recommend using
    :func:`auto_metric_sklearn` to automatically deal with different metrics.

    Parameters
    ----------
    y_true
        An array of ground truth values.
    y_pred
        An array of predictions.
    metric
        Use ``tabensemb.utils.utils.REGRESSION_METRICS``, ``tabensemb.utils.utils.BINARY_METRICS``, and
        ``tabensemb.utils.utils.MULTICLASS_METRICS`` to check all available metrics for regression, binary, and
        multiclass tasks respectively.

    Returns
    -------
    float
        The metric.

    See Also
    --------
    :func:`auto_metric_sklearn`
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
        y_true = y_true.flatten()
    if not np.all(np.isfinite(y_pred)):
        if tabensemb.setting["warn_nan_metric"]:
            warnings.warn(
                f"NaNs exist in the tested prediction. A large value (100) is returned instead."
                f"To disable this and raise an Exception, turn the global setting `warn_nan_metric` to False."
            )
            return 100
        else:
            raise Exception(
                f"NaNs exist in the tested prediction. To ignore this and return a large value (100) instead, turn "
                f"the global setting `warn_nan_metric` to True"
            )

    mapping = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
        "r2": r2_score,
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2_score": r2_score,
        "median_absolute_error": median_absolute_error,
        "max_error": max_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_poisson_deviance": mean_poisson_deviance,
        "mean_gamma_deviance": mean_gamma_deviance,
        "mean_pinball_loss": mean_pinball_loss,
        "accuracy_score": accuracy_score,
        "top_k_accuracy_score": top_k_accuracy_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "average_precision_score": average_precision_score,
        "precision_score": partial(precision_score, zero_division=0),
        "recall_score": partial(recall_score, zero_division=0),
        "log_loss": partial(log_loss),
        "balanced_accuracy_score": balanced_accuracy_score,
        "explained_variance_score": explained_variance_score,
        "brier_score_loss": brier_score_loss,
        "jaccard_score": jaccard_score,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "cohen_kappa_score": cohen_kappa_score,
        "hamming_loss": hamming_loss,
        "matthews_corrcoef": matthews_corrcoef,
        "zero_one_loss": zero_one_loss,
        "precision_score_macro": partial(
            precision_score, average="macro", zero_division=0
        ),
        "precision_score_micro": partial(
            precision_score, average="micro", zero_division=0
        ),
        "precision_score_weighted": partial(
            precision_score, average="weighted", zero_division=0
        ),
        "recall_score_macro": partial(recall_score, average="macro", zero_division=0),
        "recall_score_micro": partial(recall_score, average="micro", zero_division=0),
        "recall_score_weighted": partial(
            recall_score, average="weighted", zero_division=0
        ),
        "f1_score_macro": partial(f1_score, average="macro"),
        "f1_score_micro": partial(f1_score, average="micro"),
        "f1_score_weighted": partial(f1_score, average="weighted"),
        "jaccard_score_macro": partial(jaccard_score, average="macro"),
        "jaccard_score_micro": partial(jaccard_score, average="micro"),
        "jaccard_score_weighted": partial(jaccard_score, average="weighted"),
        "roc_auc_score_ovr_macro": partial(
            roc_auc_score, average="macro", multi_class="ovr"
        ),
        "roc_auc_score_ovr_weighted": partial(
            roc_auc_score, average="weighted", multi_class="ovr"
        ),
        "roc_auc_score_ovo": partial(
            roc_auc_score, average="weighted", multi_class="ovo"
        ),
    }
    if metric in mapping.keys():
        return mapping[metric](y_true, y_pred)
    elif metric == "rmse_conserv":
        y_pred = np.array(cp(y_pred)).reshape(-1, 1)
        y_true = np.array(cp(y_true)).reshape(-1, 1)
        where_not_conserv = y_pred > y_true
        if np.any(where_not_conserv):
            return mean_squared_error(
                y_true[where_not_conserv], y_pred[where_not_conserv]
            )
        else:
            return 0.0
    else:
        raise Exception(f"Metric {metric} not implemented.")


def convert_proba_to_target(y_pred: np.ndarray, task) -> np.ndarray:
    """
    Convert probabilities of classes to the class of each sample.

    Parameters
    ----------
    y_pred
        An array of predicted probabilities. For binary, it should be the probability of the positive class.
    task
        "multiclass" or "binary".

    Returns
    -------
    np.ndarray
        The class of each sample. 2d array (the second dimension is 1) for multiclass tasks. 0-1 array (1d or 2d
        depending on the input ``y_pred``) for binary tasks.
    """
    if task == "regression":
        raise Exception(f"Not supported for regressions tasks.")
    elif task == "multiclass":
        return np.argmax(y_pred, axis=-1).reshape(-1, 1)
    elif task == "binary":
        return (y_pred > 0.5).astype(int)
    else:
        raise Exception(f"Unrecognized task {task}.")


def convert_target_to_indicator(y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert the class of each sample to class indicator.

    Parameters
    ----------
    y_pred
        The class of each sample (not probabilities). It should be a 1d array or a 2d array whose second dimension
        is 1.
    n_classes
        The number of classes.

    Returns
    -------
    np.ndarray
        An array of (n_samples, n_classes) where, at each entry, 1 indicates that the sample belongs to this class.
    """
    indicator = np.zeros((y_pred.shape[0], n_classes))
    indicator[np.arange(y_pred.shape[0]), y_pred.flatten()] = 1
    return indicator


REGRESSION_METRICS = [
    "rmse",
    "mse",
    "mae",
    "mape",
    "r2",
    # "mean_squared_log_error",
    "median_absolute_error",
    # "max_error",
    "explained_variance_score",
    # "mean_poisson_deviance",
    # "mean_gamma_deviance",
]

_BINARY_USE_TARGET_METRICS = [
    "f1_score",
    "precision_score",
    "recall_score",
    "jaccard_score",
    "accuracy_score",
    "balanced_accuracy_score",
    "cohen_kappa_score",
    "hamming_loss",
    "matthews_corrcoef",
    "zero_one_loss",
]
_BINARY_USE_PROB_METRICS = [
    "roc_auc_score",
    "log_loss",
    "brier_score_loss",
]
_BINARY_USE_INDICATOR_METRICS = ["average_precision_score"]

BINARY_METRICS = (
    _BINARY_USE_TARGET_METRICS
    + _BINARY_USE_PROB_METRICS
    + _BINARY_USE_INDICATOR_METRICS
)

_MULTICLASS_USE_TARGET_METRICS = [
    "accuracy_score",
    "balanced_accuracy_score",
    "cohen_kappa_score",
    "hamming_loss",
    "matthews_corrcoef",
    "zero_one_loss",
    "precision_score_macro",
    "precision_score_micro",
    "precision_score_weighted",
    "recall_score_macro",
    "recall_score_micro",
    "recall_score_weighted",
    "f1_score_macro",
    "f1_score_micro",
    "f1_score_weighted",
    "jaccard_score_macro",
    "jaccard_score_micro",
    "jaccard_score_weighted",
]
_MULTICLASS_USE_PROB_METRICS = [
    "top_k_accuracy_score",
    "log_loss",
    "roc_auc_score_ovr_macro",
    "roc_auc_score_ovr_weighted",
    "roc_auc_score_ovo",
]
MULTICLASS_METRICS = _MULTICLASS_USE_TARGET_METRICS + _MULTICLASS_USE_PROB_METRICS


def auto_metric_sklearn(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str, task: str
) -> float:
    """
    Calculate metrics using ``sklearn`` APIs. It automatically deals with different requirements of input shapes for
    different metrics.

    Parameters
    ----------
    y_true
        An array of ground truth values. For classification, it should be the class of each sample. It can be 1d or 2d
        (the second dimension is 1) for classification tasks.
    y_pred
        An array of predictions. For classification, it should be the probabilities of classes. It can be 1d or 2d
        (the second dimension is 1) for binary classification tasks.
    metric
        Use ``tabensemb.utils.utils.REGRESSION_METRICS``, ``tabensemb.utils.utils.BINARY_METRICS``, and
        ``tabensemb.utils.utils.MULTICLASS_METRICS`` to check all available metrics for regression, binary, and
        multiclass tasks respectively.
    task
        "regression", "multiclass", or "binary".

    Returns
    -------
    float
        The metric.
    """
    if task not in ["binary", "multiclass", "regression"]:
        raise Exception(f"Task {task} does not support auto metrics.")
    if task in ["multiclass", "binary"] and not (
        len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1)
    ):
        raise Exception(
            f"Expecting a 1d or 2d (the second dimension is 1) y_true, but got y_true with shape {y_true.shape}."
        )
    if task == "binary" and not (
        len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1)
    ):
        raise Exception(
            f"Expecting the probability of the positive class, but got y_pred with shape {y_pred.shape}."
        )
    if task == "binary":
        y_pred = y_pred.flatten()
    # For classification tasks, y_pred is proba, y_true is an integer array
    if task == "regression":
        return metric_sklearn(y_true, y_pred, metric)
    elif task == "binary":
        if metric in _BINARY_USE_TARGET_METRICS:
            return metric_sklearn(
                y_true, convert_proba_to_target(y_pred, "binary"), metric
            )
        elif metric in _BINARY_USE_PROB_METRICS:
            return metric_sklearn(y_true, y_pred, metric)
        elif metric in _BINARY_USE_INDICATOR_METRICS:
            y_pred_extend = y_pred.reshape(-1, 1)
            y_pred_2d = np.concatenate([1 - y_pred_extend, y_pred_extend], axis=-1)
            n_classes = len(np.unique(y_true))
            y_true_indicator = convert_target_to_indicator(y_true, n_classes)
            return metric_sklearn(y_true_indicator, y_pred_2d, metric)
        else:
            raise NotImplementedError
    elif task == "multiclass":
        if metric in _MULTICLASS_USE_TARGET_METRICS:
            return metric_sklearn(
                y_true, convert_proba_to_target(y_pred, task="multiclass"), metric
            )
        if metric in _MULTICLASS_USE_PROB_METRICS:
            return metric_sklearn(y_true, y_pred, metric)
        else:
            raise NotImplementedError


def str_to_dataframe(s, sep=",", names=None, check_nan_on=None) -> pd.DataFrame:
    """
    Convert a .csv type of string to a dataframe.

    Parameters
    ----------
    s
        A .csv type of string.
    sep
        The delimiter.
    names
        Column labels.
    check_nan_on
        Numerical column labels to detect invalid values and replace them with ``np.nan``.

    Returns
    -------
    pd.DataFrame
        The converted dataframe.
    """
    df = pd.read_csv(StringIO(s), names=names, sep=sep)
    if names is not None:
        if len(df.columns) != len(names) or (
            df.dtypes[names[0]] == object and pd.isna(df[names[1:]]).all().all()
        ):
            raise Exception(
                f"pd.read_csv can not handle the delimiters. Consider specifying `sep`."
            )

    if check_nan_on is not None:
        is_object = df[check_nan_on].dtypes == object
        object_features = is_object.index[np.where(is_object)[0]]
        if len(object_features) > 0:
            print(
                f"Unknown values are detected in {list(object_features)}. They will be treated as np.nan."
            )
        for feature in object_features:
            is_nan = np.array(
                list(map(lambda x: not x.replace(".", "").isnumeric(), df[feature]))
            )
            df.loc[is_nan, feature] = np.nan
    return df


def get_figsize(n, max_col, width_per_item, height_per_item, max_width):
    """
    Calculate the ``figsize`` argument of ``matplotlib`` for a figure with subplots.

    Parameters
    ----------
    n
        The number of subplots.
    max_col
        The maximum number of columns.
    width_per_item
        The width of each column if only one row is needed.
    height_per_item
        The height of each row.
    max_width
        The width of the figure if multiple rows are needed.

    Returns
    -------
    tuple
        The ``figsize`` argument of ``matplotlib``
    int
        The number of columns of the figure
    int
        The number of rows of the figure
    """
    if n > max_col:
        width = max_col
        if n % max_col == 0:
            height = n // max_col
        else:
            height = n // max_col + 1
        figsize = (max_width, height_per_item * height)
    else:
        figsize = (width_per_item * n, height_per_item)
        width = n
        height = 1
    return figsize, width, height


def check_stream():
    """
    A utility of :func:`HiddenPrints`.
    """
    if not isinstance(sys.stdout, tabensemb.Stream) or not isinstance(
        sys.stderr, tabensemb.Stream
    ):
        return False
    return True


class HiddenPrints:
    """
    A context manager that can temporarily hide all ``sys.stdout`` outputs and ``logging`` outputs.
    It works better when ``sys.stdout`` is not changed after ``tabensemb`` is imported.
    """

    def __init__(self, disable_logging: bool = True, disable_std: bool = True):
        """
        Parameters
        ----------
        disable_logging
            Hide ``logging`` outputs
        disable_std
            Hide ``sys.stdout`` outputs
        """
        self.disable_logging = disable_logging
        self.disable_std = disable_std

    def __enter__(self):
        if self.disable_std:
            if check_stream():
                self._stream = tabensemb.stdout_stream.stream
                self._null_stream = open(os.devnull, "w")
                tabensemb.stdout_stream.set_stream(self._null_stream)
                self._path = tabensemb.stdout_stream.path
                tabensemb.stdout_stream.set_path(None)
            else:
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
        if self.disable_logging:
            self.logging_state = logging.root.manager.disable
            logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_std:
            if check_stream():
                self._null_stream.close()
                tabensemb.stdout_stream.set_stream(self._stream)
                tabensemb.stdout_stream.set_path(self._path)
            else:
                sys.stdout.close()
                sys.stdout = self._original_stdout
        if self.disable_logging:
            logging.disable(self.logging_state)


class PlainText:
    """
    A context manager that can temporarily redirect all ``sys.stderr`` outputs to ``sys.stdout``.
    It works better when ``sys.stdout`` and ``sys.stderr`` are not changed after ``tabensemb`` is imported.
    """

    def __init__(self, disable=False):
        self.disable = disable

    def __enter__(self):
        if not self.disable:
            if check_stream():
                self._stream = tabensemb.stderr_stream.stream
                tabensemb.stderr_stream.set_stream("stdout")
            else:
                self._original_stderr = sys.stderr
                sys.stderr = sys.stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable:
            if check_stream():
                tabensemb.stderr_stream.set_stream(self._stream)
            else:
                sys.stderr = self._original_stderr


class global_setting:
    """
    A context manager that temporarily changes the global setting ``tabensemb.setting``.
    """

    def __init__(self, setting: Dict):
        self.setting = setting
        self.original = None

    def __enter__(self):
        self.original = tabensemb.setting.copy()
        tabensemb.setting.update(self.setting)

    def __exit__(self, exc_type, exc_val, exc_tb):
        tabensemb.setting.update(self.original)


class HiddenPltShow:
    """
    A context manager that temporarily hide all ``matplotlib.pyplot.show()``.
    """

    def __init__(self):
        pass

    def __enter__(self):
        def nullfunc(*args, **kwargs):
            pass

        self.original = plt.show
        plt.show = nullfunc

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self.original


def reload_module(name):
    """
    Re-import the module.

    Parameters
    ----------
    name
        The name of the module
    """
    if name not in sys.modules:
        mod = import_module(name)
    else:
        mod = reload(sys.modules.get(name))
    return mod


class TqdmController:
    """
    A controller of ``tqdm`` progress bars, including ``tqdm.tqdm``, ``tqdm.notebook.tqdm``, and ``tqdm.auto.tqdm``.
    """

    def __init__(self):
        self.original_init = {}
        self.disabled = False

    def disable_tqdm(self):
        def disable_one(name):
            tq = reload_module(name).tqdm
            self.original_init[name] = tq.__init__
            tq.__init__ = partialmethod(tq.__init__, disable=True)

        disable_one("tqdm")
        disable_one("tqdm.notebook")
        disable_one("tqdm.auto")
        self.disabled = True

    def enable_tqdm(self):
        def enable_one(name):
            tq = reload_module(name).tqdm
            tq.__init__ = self.original_init[name]

        if self.disabled:
            enable_one("tqdm")
            enable_one("tqdm.notebook")
            enable_one("tqdm.auto")
            self.disabled = False


def debugger_is_active() -> bool:
    """
    Return True if the debugger is currently active
    """
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def gini(x: np.ndarray, w: np.ndarray = None) -> float:
    """
    Calculate the gini index of a feature.
    https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python

    Parameters
    ----------
    x
        The values of a feature.
    w
        The weights of samples.

    Returns
    -------
    float
        The gini index of the feature.
    """
    x = np.asarray(x)
    w = w[np.isfinite(x)] if w is not None else None
    x = x[np.isfinite(x)]
    if len(np.unique(x)) == 1:
        return np.nan
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def pretty(value, htchar="\t", lfchar="\n", indent=0):
    """
    Represent a dictionary, a list, or a tuple by a string.
    https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries

    Parameters
    ----------
    value
        A dictionary, a list, or a tuple to be formatted.
    htchar
        The string for indents.
    lfchar
        The string between two lines.
    indent
        The number of indents.

    Returns
    -------
    str
        The formatted representation of ``value``.
    """
    nlch = lfchar + htchar * (indent + 1)
    if isinstance(value, dict):
        items = [
            nlch + repr(key) + ": " + pretty(value[key], htchar, lfchar, indent + 1)
            for key in value
        ]
        return "{%s}" % (",".join(items) + lfchar + htchar * indent)
    elif isinstance(value, list):
        items = [nlch + pretty(item, htchar, lfchar, indent + 1) for item in value]
        return "[%s]" % (",".join(items) + lfchar + htchar * indent)
    elif isinstance(value, tuple):
        items = [nlch + pretty(item, htchar, lfchar, indent + 1) for item in value]
        return "(%s)" % (",".join(items) + lfchar + htchar * indent)
    else:
        return repr(value)


def update_defaults_by_kwargs(defaults: Dict = None, kwargs: Dict = None):
    defaults = defaults if defaults is not None else {}
    defaults.update({} if kwargs is None else kwargs)
    return defaults


class Logger:
    """
    Capture all outputs to a log file while still printing it. It works as a utility of :class:`Logging`.
    https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    """

    def __init__(self, path, stream):
        self.terminal = stream
        self.path = path

    def write(self, message):
        self.terminal.write(message)
        with open(self.path, "ab") as log:
            log.write(message.encode("utf-8"))

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


class Logging:
    """
    Capture all outputs to a log file while still printing it.
    """

    def enter(self, path):
        if check_stream():
            tabensemb.stdout_stream.set_path(path)
            tabensemb.stderr_stream.set_path(path)
        else:
            self.out_logger = Logger(path, sys.stdout)
            self.err_logger = Logger(path, sys.stderr)
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = self.out_logger
            sys.stderr = self.err_logger

    def exit(self):
        if check_stream():
            tabensemb.stdout_stream.set_path(None)
            tabensemb.stderr_stream.set_path(None)
        else:
            sys.stdout = self._stdout
            sys.stderr = self._stderr


def add_postfix(path):
    """
    If the input path exists, add a postfix ``f"-I{n}"`` to it, where ``n`` increases if ``path`` ends with ``f"-I{n}"``.

    Parameters
    ----------
    path
        A path to a folder or a file that will be created.

    Returns
    -------
    str
        A path that can be created without conflict.
    """
    postfix_iter = itertools.count()
    s = cp(path)
    root, ext = os.path.splitext(s)
    is_folder = len(ext) == 0
    last_cnt = postfix_iter.__next__()
    while os.path.exists(s) if is_folder else os.path.isfile(s):
        root_split = list(os.path.split(root))
        last_postfix = f"-I{last_cnt}"
        last_cnt = postfix_iter.__next__()
        if root_split[-1].endswith(last_postfix):
            # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
            root_split[-1] = f"-I{last_cnt}".join(
                root_split[-1].rsplit(last_postfix, 1)
            )
        else:
            root_split[-1] += f"-I{last_cnt}"
        s = os.path.join(*root_split) + ext
        root, ext = os.path.splitext(s)
    return s


def safe_mkdir(path: os.PathLike):
    """
    Make a previously not existing directory safely resolving conflicts. When multiple tasks are executed
    simultaneously, this is extremely useful even when ``os.path.exist`` is used.

    Parameters
    ----------
    path
        The intended path

    Returns
    -------
    str
        The actual made path
    """
    while True:
        try:
            os.mkdir(path)
            break
        except FileExistsError:
            path = add_postfix(path)
        except Exception as e:
            raise e
    return path


class torch_with_grad(_DecoratorContextManager):
    """
    A context manager that enabled gradient calculation. This is an inverse version of torch.no_grad
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)


class PickleAbleGenerator:
    """
    Turn a generator (not pickle-able) into a pickle-able object by extracting all items in the generator to a list.
    """

    def __init__(self, generator, max_generate=10000, inf=False):
        self.ls = []
        self.state = 0
        for i in range(max_generate):
            try:
                self.ls.append(generator.__next__())
            except:
                break
        else:
            if not inf:
                raise Exception(
                    f"The generator {generator} generates more than {max_generate} values. Set inf=True if you "
                    f"accept that only {max_generate} can be pickled."
                )

    def __next__(self):
        if self.state >= len(self.ls):
            raise StopIteration
        else:
            val = self.ls[self.state]
            self.state += 1
            return val

    def __getstate__(self):
        return {"state": self.state, "ls": self.ls}

    def __setstate__(self, state):
        self.state = state["state"]
        self.ls = state["ls"]
