from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.model import *
import shutil
import pytest
import pandas as pd
import numpy as np


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


def pytest_configure_trainer():
    if getattr(pytest, "model_configure_excuted", False):
        pytest.test_model_trainer.clear_modelbase()
        return
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile)
    trainer.load_data()
    pytest.test_model_trainer = trainer
    pytest.model_configure_excuted = True


def test_embed():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
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

    assert no_extend_rmse != extend_rmse


def test_wrap():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
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
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer

    base_model = CatEmbed(trainer, model_subset=["Category Embedding"])
    rfe = RFE(
        trainer,
        modelbase=base_model,
        model_subset=["Category Embedding"],
        min_features=2,
        cross_validation=2,
    )
    trainer.add_modelbases([base_model, rfe])

    trainer.train()
    l = trainer.get_leaderboard()

    assert l.loc[0, "Testing RMSE"] != l.loc[1, "Testing RMSE"]


def test_exceptions(capfd):
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    trainer.args["bayes_opt"] = True
    models = [
        CatEmbed(
            trainer,
            model_subset=["Category Embedding", "Category Embedding Extend dim"],
        ),
    ]
    trainer.add_modelbases(models)

    with pytest.raises(Exception):
        models[0]._check_train_status()

    with pytest.raises(Exception):
        models[0].predict(
            trainer.df,
            model_name="Category Embedding",
            derived_data=trainer.derived_data,
        )

    models[0].fit(
        trainer.df,
        cont_feature_names=trainer.cont_feature_names,
        cat_feature_names=trainer.cat_feature_names,
        label_name=trainer.label_name,
        derived_data=trainer.derived_data,
        bayes_opt=False,
    )
    out, err = capfd.readouterr()
    assert "conflicts" in out
    assert trainer.args["bayes_opt"]

    with pytest.raises(Exception):
        models[0].predict(
            trainer.df,
            model_name="TEST",
            derived_data=trainer.derived_data,
        )


def test_check_batch_size():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    model = CatEmbed(
        trainer,
        model_subset=["Category Embedding", "Category Embedding Extend dim"],
    )
    l = len(trainer.train_indices)

    with pytest.raises(Exception):
        model.limit_batch_size = -1
        res = model._check_params("TEST", **{"batch_size": 2})

    with pytest.warns(UserWarning):
        model.limit_batch_size = -1
        res = model._check_params("TEST", **{"batch_size": 6})

    with pytest.warns(UserWarning):
        model.limit_batch_size = 1
        res = model._check_params("TEST", **{"batch_size": 2})
        assert res["batch_size"] == 3

    with pytest.warns(UserWarning):
        model.limit_batch_size = 90
        res = model._check_params("TEST", **{"batch_size": 80})
        assert res["batch_size"] == l

    model = PytorchTabular(
        trainer,
        model_subset=["TabNet"],
    )
    with pytest.warns(UserWarning):
        model.limit_batch_size = 5
        res = model._check_params("TabNet", **{"batch_size": 32})
        assert res["batch_size"] == 64


def test_get_model_names():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    with pytest.raises(Exception):
        model = CatEmbed(trainer, model_subset=["TEST"])
    model = CatEmbed(trainer, exclude_models=["Category Embedding"])
    got = model.get_model_names()
    got_all = model._get_model_names()
    assert len(got_all) == len(got) + 1
    assert "Category Embedding" not in got

    model = CatEmbed(trainer)
    got = model.get_model_names()
    got_all = model._get_model_names()
    assert all([name in got for name in got_all])


def test_abstract_model():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer

    class NotImplementedModel(AbstractModel):
        def _get_program_name(self) -> str:
            return "NotImplementedModel"

        @staticmethod
        def _get_model_names():
            return ["TEST"]

        def _initial_values(self, model_name: str):
            return {}

        def _space(self, model_name: str):
            return []

    abs_model = NotImplementedModel(trainer, program="TEST_PROGRAM")
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._get_model_names()
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._get_program_name()
    with pytest.raises(NotImplementedError):
        abs_model._new_model("TEST", verbose=True)
    with pytest.raises(NotImplementedError):
        abs_model._train_data_preprocess("TEST")
    with pytest.raises(NotImplementedError):
        abs_model._data_preprocess(
            df=pd.DataFrame(), derived_data={}, model_name="TEST"
        )
    with pytest.raises(NotImplementedError):
        abs_model._train_single_model(
            model=None,
            epoch=1,
            X_train=None,
            y_train=np.array([]),
            X_val=None,
            y_val=None,
            verbose=False,
            warm_start=False,
            in_bayes_opt=False,
        )
    with pytest.raises(NotImplementedError):
        abs_model._pred_single_model(model=None, X_test=None, verbose=False)
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._space("TEST")
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._initial_values("TEST")


def test_count_params():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    model = CatEmbed(
        trainer,
        model_subset=["Category Embedding"],
    )
    trainer.add_modelbases([model])
    cnt_1 = model.count_params("Category Embedding")
    trainer.train()
    cnt_2 = model.count_params("Category Embedding")
    cnt_3 = model.count_params("Category Embedding", trainable_only=True)
    assert cnt_1 == cnt_2
    assert cnt_1 != cnt_3
