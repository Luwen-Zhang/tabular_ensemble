import os

import pytest
import torch
from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.trainer.utils import NoBayesOpt
from tabensemb.utils import global_setting, torch_with_grad, Logging
from tabensemb.utils.ranking import *
from tabensemb.model import *
import shutil
import warnings
from logging import getLogger


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
    log = getLogger()
    log.log(1, "2")
    logger.exit()
    print(3)

    log.log(1, "4")
    with open(path, "r") as file:
        lines = file.readlines()
    assert len(lines) == 1
    assert lines[0] == "1\n"
