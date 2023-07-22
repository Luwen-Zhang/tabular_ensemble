import copy
from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.trainer.utils import NoBayesOpt
from tabensemb.utils.ranking import *
from tabensemb.model import *
import shutil


def test_no_bayes_opt():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile, manual_config={"bayes_opt": True})

    assert trainer.args["bayes_opt"]

    with NoBayesOpt(trainer):
        assert not trainer.args["bayes_opt"]

    assert trainer.args["bayes_opt"]

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))


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

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))
