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
from functools import partialmethod
import itertools
from copy import deepcopy as cp
from torch.autograd.grad_mode import _DecoratorContextManager
from typing import Any
import tabensemb
from .collate import fix_collate_fn
from typing import Dict

clr = sns.color_palette("deep")
sns.reset_defaults()
# matplotlib.use("Agg")
if find_executable("latex"):
    matplotlib.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True

global_palette = [
    "#FF3E41",
    "#00B2CA",
    "#FEB95F",
    "#3B3561",
    "#679436",
    "#BF9ACA",
    "#1F487E",
    "#2B59C3",
    "#6E0E0A",
    "#395E66",
    "#726DA8",
    "#5438DC",
    "#791E94",
    "#CEF7A0",
]


def is_notebook() -> bool:
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
    set_torch(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch.utils.data" not in sys.modules:
        dl = import_module("torch.utils.data").DataLoader
    else:
        dl = reload(sys.modules.get("torch.utils.data")).DataLoader

    if not dl.__init__.__name__ == "_method":
        # Actually, setting generator improves reproducibility, but torch._C.Generator does not support pickling.
        # https://pytorch.org/docs/stable/notes/randomness.html
        # https://github.com/pytorch/pytorch/issues/43672
        dl.__init__ = partialmethod(dl.__init__, worker_init_fn=seed_worker)

    torch.utils.data._utils.collate.default_collate = fix_collate_fn


def metric_sklearn(y_true, y_pred, metric):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
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
    if metric == "mse":
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        from sklearn.metrics import mean_squared_error

        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mae":
        from sklearn.metrics import mean_absolute_error

        return mean_absolute_error(y_true, y_pred)
    elif metric == "mape":
        from sklearn.metrics import mean_absolute_percentage_error

        return mean_absolute_percentage_error(y_true, y_pred)
    elif metric == "r2":
        from sklearn.metrics import r2_score

        return r2_score(y_true, y_pred)
    elif metric == "rmse_conserv":
        from sklearn.metrics import mean_squared_error

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


def plot_importance(ax, features, attr, pal, clr_map, **kwargs):
    df = pd.DataFrame(columns=["feature", "attr", "clr"])
    df["feature"] = features
    df["attr"] = np.abs(attr) / np.sum(np.abs(attr))
    df["pal"] = pal
    df.sort_values(by="attr", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    ax.set_axisbelow(True)
    x = df["feature"].values
    y = df["attr"].values

    palette = df["pal"]

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(x=y, y=x, palette=palette, **kwargs)
    # ax.set_xlim([0, 1])
    ax.set_xlabel("Permutation feature importance")

    legend = ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, color=value, ec="k", label=key)
            for key, value in clr_map.items()
        ],
        loc="lower right",
        handleheight=2,
        fancybox=False,
        frameon=False,
    )

    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor([1, 1, 1, 0.4])


def get_figsize(n, max_col, width_per_item, height_per_item, max_width):
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


def plot_pdp(
    feature_names,
    cat_feature_names,
    cat_feature_mapping,
    x_values_list,
    mean_pdp_list,
    ci_left_list,
    ci_right_list,
    hist_data,
    log_trans=True,
    lower_lim=2,
    upper_lim=7,
):
    figsize, width, height = get_figsize(
        n=len(feature_names),
        max_col=4,
        width_per_item=3,
        height_per_item=3,
        max_width=14,
    )

    fig = plt.figure(figsize=figsize)

    def transform(value):
        if log_trans:
            return 10**value
        else:
            return value

    for idx, focus_feature in enumerate(feature_names):
        ax = plt.subplot(height, width, idx + 1)
        # ax.plot(x_values_list[idx], mean_pdp_list[idx], color = clr_map[focus_feature], linewidth = 0.5)
        if focus_feature not in cat_feature_names:
            ax.plot(
                x_values_list[idx],
                transform(mean_pdp_list[idx]),
                color="k",
                linewidth=0.7,
            )

            ax.fill_between(
                x_values_list[idx],
                transform(ci_left_list[idx]),
                transform(ci_right_list[idx]),
                alpha=0.4,
                color="k",
                edgecolor=None,
            )
        else:
            yerr = (
                np.abs(
                    np.vstack(
                        [transform(ci_left_list[idx]), transform(ci_right_list[idx])]
                    )
                    - transform(mean_pdp_list[idx])
                )
                if not np.isnan(ci_left_list[idx]).any()
                else None
            )
            ax.bar(
                x_values_list[idx],
                transform(mean_pdp_list[idx]),
                yerr=yerr,
                capsize=5,
                color=[0.5, 0.5, 0.5],
                edgecolor=None,
                error_kw={"elinewidth": 0.2, "capthick": 0.2},
                tick_label=[
                    cat_feature_mapping[focus_feature][x] for x in x_values_list[idx]
                ],
            )

        ax.set_title(focus_feature, {"fontsize": 12})
        # ax.set_xlim([0, 1])
        if log_trans:
            ax.set_yscale("log")
            ax.set_ylim([10**lower_lim, 10**upper_lim])
            locmin = matplotlib.ticker.LogLocator(
                base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
            )
            # ax.xaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_locator(locmin)
            # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if np.min(x_values_list[idx]) < np.max(x_values_list[idx]):
            if focus_feature not in cat_feature_names:
                ax2 = ax.twinx()

                ax2.hist(
                    hist_data[focus_feature],
                    bins=x_values_list[idx],
                    density=True,
                    color="k",
                    alpha=0.2,
                    rwidth=0.8,
                )
                # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
                # ax2.set_ylim([0,1])
                ax2.set_xlim([np.min(x_values_list[idx]), np.max(x_values_list[idx])])
                ax2.set_yticks([])
        else:
            ax2 = ax.twinx()
            ax2.text(0.5, 0.5, "Invalid interval", ha="center", va="center")
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.set_yticks([])

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("Predicted fatigue life")
    plt.xlabel("Value of predictors (standard scaled, $10\%$-$90\%$ percentile)")

    return fig


def plot_partial_err(
    feature_data, cat_feature_names, cat_feature_mapping, truth, pred, thres=0.8
):
    feature_names = list(feature_data.columns)
    figsize, width, height = get_figsize(
        n=len(feature_names),
        max_col=4,
        width_per_item=3,
        height_per_item=3,
        max_width=14,
    )

    fig = plt.figure(figsize=figsize)

    err = np.abs(truth - pred)
    high_err_data = feature_data.loc[np.where(err > thres)[0], :]
    high_err = err[np.where(err > thres)[0]]
    low_err_data = feature_data.loc[np.where(err <= thres)[0], :]
    low_err = err[np.where(err <= thres)[0]]
    for idx, focus_feature in enumerate(feature_names):
        ax = plt.subplot(height, width, idx + 1)
        # ax.plot(x_values_list[idx], mean_pdp_list[idx], color = clr_map[focus_feature], linewidth = 0.5)
        ax.scatter(
            high_err_data[focus_feature].values, high_err, s=1, color=clr[0], marker="s"
        )
        ax.scatter(
            low_err_data[focus_feature].values, low_err, s=1, color=clr[1], marker="^"
        )

        ax.set_title(focus_feature, {"fontsize": 12})

        ax.set_ylim([0, np.max(err) * 1.1])
        ax2 = ax.twinx()

        ax2.hist(
            [high_err_data[focus_feature].values, low_err_data[focus_feature].values],
            bins=np.linspace(
                np.min(feature_data[focus_feature].values),
                np.max(feature_data[focus_feature].values),
                20,
            ),
            density=True,
            color=clr[:2],
            alpha=0.2,
            rwidth=0.8,
        )
        if focus_feature in cat_feature_names:
            ticks = np.sort(np.unique(feature_data[focus_feature].values)).astype(int)
            tick_label = [cat_feature_mapping[focus_feature][x] for x in ticks]
            ax.set_xticks(ticks)
            ax.set_xlabel(tick_label)
            ax.set_xlim([-0.5, len(ticks) - 0.5])
            ax2.set_xlim([-0.5, len(ticks) - 0.5])

        # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
        # ax2.set_ylim([0,1])
        # ax2.set_xlim([np.min(x_values_list[idx]), np.max(x_values_list[idx])])
        ax2.set_yticks([])

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("Prediction absolute error")
    plt.xlabel("Value of predictors")

    return fig


def plot_truth_pred(
    predictions,
    ax,
    model_name,
    log_trans=True,
    verbose=True,
):
    def plot_one(name, color, marker):
        pred_y, y = predictions[model_name][name]
        r2 = metric_sklearn(y, pred_y, "r2")
        loss = metric_sklearn(y, pred_y, "mse")
        if verbose:
            print(f"{name} MSE Loss: {loss:.4f}, R2: {r2:.4f}")
        ax.scatter(
            10**y if log_trans else y,
            10**pred_y if log_trans else pred_y,
            s=20,
            color=color,
            marker=marker,
            label=f"{name} dataset ($R^2$={r2:.3f})",
            linewidth=0.4,
            edgecolors="k",
        )

    plot_one("Training", clr[0], "o")
    plot_one("Validation", clr[2], "o")
    plot_one("Testing", clr[1], "o")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")


def set_truth_pred(ax, log_trans=True, upper_lim=9):
    if log_trans:
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(
            np.linspace(0, 10**upper_lim, 100),
            np.linspace(0, 10**upper_lim, 100),
            "--",
            c="grey",
            alpha=0.2,
        )
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
        )

        # ax.set_aspect("equal", "box")

        ax.xaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_xlim(1, 10**upper_lim)
        ax.set_ylim(1, 10**upper_lim)
        ax.set_box_aspect(1)
    else:
        # ax.set_aspect("equal", "box")
        lx, rx = ax.get_xlim()
        ly, ry = ax.get_ylim()
        l = np.min([lx, ly])
        r = np.max([rx, ry])

        ax.plot(
            np.linspace(l, r, 100),
            np.linspace(l, r, 100),
            "--",
            c="grey",
            alpha=0.2,
        )

        ax.set_xlim(left=l, right=r)
        ax.set_ylim(bottom=l, top=r)
        ax.set_box_aspect(1)

    # ax.set(xlim=[10, 10 ** 6], ylim=[10, 10 ** 6])

    # data_range = [
    #     np.floor(np.min([np.min(ground_truth), np.min(prediction)])),
    #     np.ceil(np.max([np.max(ground_truth), np.max(prediction)]))
    # ]


class HiddenPrints:
    def __init__(self, disable_logging=True, disable_std=True):
        self.disable_logging = disable_logging
        self.disable_std = disable_std

    def __enter__(self):
        if self.disable_std:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.disable_logging:
            self.logging_state = logging.root.manager.disable
            logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_std:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.disable_logging:
            logging.disable(self.logging_state)


class global_setting:
    def __init__(self, setting: Dict):
        self.setting = setting
        self.original = None

    def __enter__(self):
        self.original = tabensemb.setting.copy()
        tabensemb.setting.update(self.setting)

    def __exit__(self, exc_type, exc_val, exc_tb):
        tabensemb.setting.update(self.original)


class HiddenPltShow:
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
    if name not in sys.modules:
        mod = import_module(name)
    else:
        mod = reload(sys.modules.get(name))
    return mod


class TqdmController:
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
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def gini(x, w=None):
    # https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
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
    # https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
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


# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
# capture all outputs to a log file while still printing it
class Logger:
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
    def enter(self, path):
        self.out_logger = Logger(path, sys.stdout)
        self.err_logger = Logger(path, sys.stderr)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self.out_logger
        sys.stderr = self.err_logger

    def exit(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def add_postfix(path):
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


class torch_with_grad(_DecoratorContextManager):
    """
    Context-manager that enabled gradient calculation. This is an inverse version of torch.no_grad
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
