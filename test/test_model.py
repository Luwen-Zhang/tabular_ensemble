from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.model import *
import shutil


def _get_metric_from_leaderboard(leaderboard, model_name, program=None):
    if program is None:
        return leaderboard.loc[
            leaderboard["Model"] == model_name, "Testing RMSE"
        ].values
    else:
        return leaderboard.loc[
            (leaderboard["Model"] == model_name) & (leaderboard["Program"] == program),
            "Testing RMSE",
        ].values


def test_embed():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile)
    trainer.load_data()

    models = [
        CatEmbed(
            trainer,
            model_subset=["Category Embedding", "Category Embedding Extend dim"],
        ),
    ]
    trainer.add_modelbases(models)

    trainer.train()
    l = trainer.get_leaderboard()
    no_extend_rmse = _get_metric_from_leaderboard(
        leaderboard=l, model_name="Category Embedding"
    )
    extend_rmse = _get_metric_from_leaderboard(
        leaderboard=l, model_name="Category Embedding Extend dim"
    )

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))
    assert no_extend_rmse != extend_rmse


def test_wrap():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile)
    trainer.load_data()

    models = [
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        CatEmbed(trainer, program="ExtCatEmbed", model_subset=["Category Embedding"]),
        CatEmbed(
            trainer,
            model_subset=[
                "Category Embedding",
                "Require Model Autogluon LR",
                "Require Model WideDeep TabMlp",
                "Require Model WideDeep TabMlp Wrap",
                "Require Model PyTabular CatEmbed",
                "Require Model PyTabular CatEmbed Wrap",
                "Require Model Self CatEmbed",
                "Require Model ExtCatEmbed CatEmbed",
                "Require Model ExtCatEmbed CatEmbed Wrap",
            ],
        ),
    ]
    trainer.add_modelbases(models)

    trainer.train()
    l = trainer.get_leaderboard()

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))

    assert _get_metric_from_leaderboard(
        l, "Require Model Autogluon LR"
    ) == _get_metric_from_leaderboard(l, "Linear Regression")
    assert _get_metric_from_leaderboard(
        l, "Require Model WideDeep TabMlp"
    ) == _get_metric_from_leaderboard(l, "TabMlp")
    assert _get_metric_from_leaderboard(
        l, "Require Model WideDeep TabMlp Wrap"
    ) != _get_metric_from_leaderboard(l, "TabMlp")

    assert _get_metric_from_leaderboard(
        l, "Require Model PyTabular CatEmbed"
    ) == _get_metric_from_leaderboard(l, "Category Embedding", program="PytorchTabular")
    assert _get_metric_from_leaderboard(
        l, "Require Model PyTabular CatEmbed Wrap"
    ) != _get_metric_from_leaderboard(l, "Category Embedding", program="PytorchTabular")

    assert _get_metric_from_leaderboard(
        l, "Require Model Self CatEmbed"
    ) != _get_metric_from_leaderboard(l, "Category Embedding", program="CatEmbed")

    assert _get_metric_from_leaderboard(
        l, "Require Model ExtCatEmbed CatEmbed"
    ) == _get_metric_from_leaderboard(l, "Category Embedding", program="ExtCatEmbed")
    assert _get_metric_from_leaderboard(
        l, "Require Model ExtCatEmbed CatEmbed Wrap"
    ) != _get_metric_from_leaderboard(l, "Category Embedding", program="ExtCatEmbed")


def test_rfe():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile)
    trainer.load_data()

    rfe = RFE(
        trainer,
        modelbase=CatEmbed(trainer),
        model_subset=["Category Embedding"],
        min_features=2,
        cross_validation=2,
    )
    trainer.add_modelbases([rfe])

    rfe.run("Category Embedding")

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))
