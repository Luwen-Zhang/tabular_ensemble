import os
import numpy as np
import pytest
import torch
from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.trainer.utils import NoBayesOpt
from tabensemb.utils import *
from tabensemb.utils.ranking import *
from tabensemb.model import *
import shutil
import warnings
from logging import getLogger
import copy


def test_no_bayes_opt():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile, manual_config={"bayes_opt": True})

    assert trainer.args["bayes_opt"]

    with NoBayesOpt(trainer):
        assert not trainer.args["bayes_opt"]

    assert trainer.args["bayes_opt"]


def test_ranking():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True

    def run_once(splitter, seed):
        trainer = Trainer(device="cpu")
        trainer.load_config(configfile, manual_config={"data_splitter": splitter})
        trainer.load_data()
        models = [
            PytorchTabular(trainer, model_subset=["Category Embedding"]),
            WideDeep(trainer, model_subset=["TabMlp"]),
        ]
        trainer.add_modelbases(models)
        tabensemb.setting["random_seed"] = seed
        trainer.train()
        trainer.get_leaderboard()
        return os.path.join(trainer.project_root, "leaderboard.csv")

    path_random_1 = run_once("RandomSplitter", 1)
    path_random_2 = run_once("RandomSplitter", 2)
    path_random_3 = run_once("RandomSplitter", 3)

    dfs = read_lbs([path_random_1, path_random_2])
    df1 = merge_leaderboards(dfs)
    dfs = read_lbs([path_random_3])
    df2 = merge_leaderboards(dfs)
    df = avg_rank([df1, df2])
    merge_to_excel(
        path=os.path.join(tabensemb.setting["default_output_path"], "summary.xlsx"),
        dfs=[df1, df2],
        avg_df=df,
        sheet_names=[
            "Material-cycle extrap.",
            "Cycle extrap.",
            "Average Ranking",
        ],
        index=False,
        engine="openpyxl",
    )


def test_with_global_setting():
    tabensemb.setting["debug_mode"] = False
    with global_setting({"debug_mode": True}):
        assert tabensemb.setting["debug_mode"]
    assert not tabensemb.setting["debug_mode"]


def test_torch_with_grad():
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        with torch_with_grad():
            assert torch.is_grad_enabled()
        assert not torch.is_grad_enabled()


def test_logging():
    logger = Logging()
    path = os.path.join(tabensemb.setting["default_output_path"], "log.txt")
    if os.path.exists(tabensemb.setting["default_output_path"]):
        shutil.rmtree(tabensemb.setting["default_output_path"])
    os.makedirs(tabensemb.setting["default_output_path"], exist_ok=True)
    logger.enter(path)
    print(1)
    with PlainText(disable=False):
        print(2)
        print(3, file=sys.stderr)
    print(4, file=sys.stderr)
    with HiddenPrints(disable_std=True):
        with HiddenPrints(disable_std=True):
            print(5)
    print(6)
    logger.exit()
    print(7)
    with open(path, "r") as file:
        lines = file.readlines()
    assert len(lines) == 5
    assert all([x == f"{y}\n" for x, y in zip(lines, [1, 2, 3, 4, 6])])


def test_logging_after_stream_modification():
    tabensemb.stdout_stream = tabensemb.Stream("stdout")
    tabensemb.stderr_stream = tabensemb.Stream("stderr")
    sys.stdout = tabensemb.stdout_stream
    sys.stderr = tabensemb.stderr_stream
    logger = Logging()
    path = os.path.join(tabensemb.setting["default_output_path"], "log.txt")
    if os.path.exists(tabensemb.setting["default_output_path"]):
        shutil.rmtree(tabensemb.setting["default_output_path"])
    os.makedirs(tabensemb.setting["default_output_path"], exist_ok=True)
    logger.enter(path)
    print(1)
    with PlainText(disable=False):
        print(2)
        print(3, file=sys.stderr)
    print(4, file=sys.stderr)
    with HiddenPrints(disable_std=True):
        with HiddenPrints(disable_std=True):
            print(5)
    print(6)
    logger.exit()
    print(7)
    with open(path, "r") as file:
        lines = file.readlines()
    sys.stdout = tabensemb.stdout_stream._stdout
    sys.stderr = tabensemb.stderr_stream._stderr
    assert len(lines) == 5
    assert all([x == f"{y}\n" for x, y in zip(lines, [1, 2, 3, 4, 6])])


def test_metric_sklearn():
    y_true = np.random.randn(100, 1)
    y_pred = np.random.randn(100, 1)

    for metric in REGRESSION_METRICS:
        assert auto_metric_sklearn(
            y_true.flatten(), y_pred, metric, "regression"
        ) == auto_metric_sklearn(y_true, y_pred.flatten(), metric, "regression")
    assert (
        metric_sklearn(np.ones_like(y_true), np.zeros_like(y_true), "rmse_conserv")
        == 0.0
    )

    y_true = np.random.randn(100, 2)
    y_pred = np.random.randn(100, 2)
    for metric in REGRESSION_METRICS:
        _ = auto_metric_sklearn(y_true, y_pred, metric, "regression")

    y_true = np.ones((100, 2), dtype=np.float32)
    y_pred = np.ones((100, 2), dtype=np.float32) + np.random.randn(100, 1) / 10

    with pytest.raises(Exception) as err:
        metric_sklearn(y_true, y_pred, "UNKNOWN_METRIC")
    assert "not implemented" in err.value.args[0]

    y_pred_na = copy.deepcopy(y_pred)
    y_pred_na[10] = np.nan

    tabensemb.setting["warn_nan_metric"] = True
    with pytest.warns(UserWarning):
        res = metric_sklearn(y_true, y_pred_na, "mse")
        assert res == 100

    tabensemb.setting["warn_nan_metric"] = False
    with pytest.raises(Exception) as err:
        _ = metric_sklearn(y_true, y_pred_na, "mse")
    assert "NaNs exist in the tested prediction" in err.value.args[0]

    with pytest.raises(Exception) as err:
        convert_proba_to_target(y_pred, "regression")
    assert "Not supported for regressions tasks" in err.value.args[0]
    with pytest.raises(Exception) as err:
        convert_proba_to_target(y_pred, "TEST")
    assert "Unrecognized task" in err.value.args[0]

    y_true = np.random.randint(0, 4, (10,))
    y_true[:4] = np.arange(4)
    y_pred_prob = np.abs(np.random.randn(10, 4))
    y_pred_prob /= np.repeat(np.sum(y_pred_prob, axis=1).reshape(-1, 1), 4, axis=1)
    y_pred = convert_proba_to_target(y_pred_prob, "multiclass")
    assert not np.any(y_pred > 3)
    for metric in MULTICLASS_METRICS:
        assert auto_metric_sklearn(
            y_true.reshape(-1, 1), y_pred_prob, metric, "multiclass"
        ) == auto_metric_sklearn(y_true, y_pred_prob, metric, "multiclass")
    with pytest.raises(Exception) as err:
        auto_metric_sklearn(y_true, y_pred_prob, "TEST", "TEST")
    assert "does not support auto metrics" in err.value.args[0]
    with pytest.raises(NotImplementedError):
        auto_metric_sklearn(y_true, y_pred_prob, "TEST", "multiclass")
    with pytest.raises(Exception) as err:
        auto_metric_sklearn(
            y_true.reshape(-1, 1).repeat(2, axis=1), y_pred_prob, "TEST", "multiclass"
        )
    assert "Expecting a 1d or 2d" in err.value.args[0]

    y_true = np.random.randint(0, 2, (10,))
    y_pred_prob_1d = np.abs(np.random.randn(10))
    y_pred_prob_1d /= np.max(y_pred_prob_1d)
    y_pred = convert_proba_to_target(y_pred_prob_1d, "binary")
    assert not np.any(y_pred > 1)
    for metric in BINARY_METRICS:
        assert auto_metric_sklearn(
            y_true.reshape(-1, 1), y_pred_prob_1d, metric, "binary"
        ) == auto_metric_sklearn(
            y_true, y_pred_prob_1d.reshape(-1, 1), metric, "binary"
        )
    with pytest.raises(NotImplementedError):
        auto_metric_sklearn(y_true, y_pred_prob_1d, "TEST", "binary")
    with pytest.raises(Exception) as err:
        auto_metric_sklearn(
            y_true.reshape(-1, 1).repeat(2, axis=1), y_pred_prob_1d, "TEST", "binary"
        )
    assert "Expecting a 1d or 2d" in err.value.args[0]
    with pytest.raises(Exception) as err:
        auto_metric_sklearn(
            y_true, y_pred_prob_1d.reshape(-1, 1).repeat(2, axis=1), "TEST", "binary"
        )
    assert "Expecting the probability of the positive class" in err.value.args[0]


def test_get_figsize():
    figsize, width, height = get_figsize(
        12, max_col=3, width_per_item=1, height_per_item=1, max_width=10
    )
    assert width == 3 and height == 4

    figsize, width, height = get_figsize(
        13, max_col=3, width_per_item=1, height_per_item=1, max_width=10
    )
    assert width == 3 and height == 5

    figsize, width, height = get_figsize(
        3, max_col=3, width_per_item=1, height_per_item=1, max_width=10
    )
    assert width == 3 and height == 1

    figsize, width, height = get_figsize(
        2, max_col=3, width_per_item=1, height_per_item=1, max_width=10
    )
    assert width == 2 and height == 1


def test_gini():
    x = np.random.randn(100)
    w = np.ones_like(x)
    assert np.allclose(gini(x, w), gini(x))


def test_debugger():
    debugger_is_active()


def test_seed_worker():
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler

    tensor = torch.randn(10, 2)
    dataset = TensorDataset(tensor)
    loader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        num_workers=2,
    )
    set_torch(0)
    res1 = [x[0] for x in loader]
    set_torch(0)
    res2 = [x[0] for x in loader]
    assert all([torch.allclose(x, y) for x, y in zip(res1, res2)])
