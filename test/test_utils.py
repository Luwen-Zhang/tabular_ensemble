import os
import numpy as np
import pytest
import torch
from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.trainer.utils import NoBayesOpt
from tabensemb.utils import (
    global_setting,
    torch_with_grad,
    Logging,
    metric_sklearn,
    get_figsize,
    gini,
)
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
    logger.exit()
    print(2)
    with open(path, "r") as file:
        lines = file.readlines()
    assert len(lines) == 1
    assert lines[0] == "1\n"


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
    logger.exit()
    print(2)
    with open(path, "r") as file:
        lines = file.readlines()
    sys.stdout = tabensemb.stdout_stream._stdout
    sys.stderr = tabensemb.stderr_stream._stderr
    assert len(lines) == 1
    assert lines[0] == "1\n"


def test_metric_sklearn():
    y_true = np.ones((100, 1), dtype=np.float32)
    y_pred = np.ones((100, 1), dtype=np.float32) + np.random.randn(100, 1) / 10

    _ = metric_sklearn(y_true, y_pred, "mse")
    _ = metric_sklearn(y_true, y_pred, "rmse")
    _ = metric_sklearn(y_true, y_pred, "mae")
    _ = metric_sklearn(y_true, y_pred, "mape")
    _ = metric_sklearn(y_true, y_pred, "r2")
    _ = metric_sklearn(y_true, y_pred, "rmse_conserv")
    assert metric_sklearn(y_true, np.zeros_like(y_true), "rmse_conserv") == 0.0

    with pytest.raises(Exception):
        metric_sklearn(y_true, y_pred, "UNKNOWN_METRIC")

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
