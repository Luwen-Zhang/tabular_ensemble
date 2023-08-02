import torch
from import_utils import *
import tabensemb
from tabensemb.trainer import Trainer
from tabensemb.model import *
from tabensemb.model.base import (
    AbstractWrapper,
    get_sequential,
    KeepDropout,
    AdaptiveDropout,
    DictDataFrameDataset,
    DictNDArrayDataset,
    DictMixDataset,
)
from tabensemb.model.widedeep import WideDeepWrapper
import pytest
import pandas as pd
import numpy as np
from skopt.space import Real
from torch import nn


class NotImplementedModel(AbstractModel):
    def _get_program_name(self) -> str:
        return "NotImplementedModel"

    @staticmethod
    def _get_model_names():
        return ["TEST", "TEST_2"]

    def _initial_values(self, model_name: str):
        return {}

    def _space(self, model_name: str):
        return []


class NotImplementedTorchModel(TorchModel):
    def _get_program_name(self) -> str:
        return "NotImplementedModel"

    @staticmethod
    def _get_model_names():
        return ["TEST", "TEST_2"]

    def _initial_values(self, model_name: str):
        return {}

    def _space(self, model_name: str):
        return []

    def _new_model(self, model_name: str, verbose: bool, **kwargs):
        return None


class NotImplementedWrapper(AbstractWrapper):
    def wrap_forward(self):
        pass


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

    with pytest.raises(Exception) as err:
        models[-1].get_full_name_from_required_model(
            models[-1].model["Require Model Self CatEmbed"].required_model
        )
    assert "`model_name` should be provided" in err.value.args[0]
    with pytest.raises(Exception) as err:
        models[-1].get_full_name_from_required_model(None)
    assert "The required model should be" in err.value.args[0]


def test_abstract_wrap():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    single_model = WideDeep(trainer, model_subset=["TabMlp"])
    multi_model = WideDeep(trainer, model_subset=["TabMlp", "TabNet"])
    with pytest.raises(Exception) as err:
        WideDeepWrapper(multi_model)
    assert "More than one model is included" in err.value.args[0]

    single_model.train()
    wrapped = WideDeepWrapper(single_model)
    # Get attribute of the wrapper
    assert wrapped.model_name == "TabMlp"
    # Get attribute of the wrapped model
    assert wrapped.program == "WideDeep"

    wrapped = NotImplementedWrapper(single_model)
    with pytest.raises(NotImplementedError):
        super(NotImplementedWrapper, wrapped).wrap_forward()
    with pytest.raises(NotImplementedError):
        wrapped.reset_forward()
    with pytest.raises(NotImplementedError):
        _ = wrapped.hidden_representation
    with pytest.raises(NotImplementedError):
        _ = wrapped.hidden_rep_dim


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


def test_abstract_model_exceptions():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    abs_model = NotImplementedModel(trainer, program="TEST_PROGRAM")
    abs_torch_model = NotImplementedTorchModel(trainer, program="TEST_TORCH_PROGRAM")
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
        abs_model.cal_feature_importance(model_name="TEST", method="NOT_EXIST")
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._space("TEST")
    with pytest.raises(NotImplementedError):
        super(NotImplementedModel, abs_model)._initial_values("TEST")
    with pytest.raises(Exception):
        abs_model = NotImplementedModel(
            trainer,
            program="TEST_PROGRAM",
            model_subset=["TEST"],
            exclude_models=["TEST"],
        )

    abs_model.model = []
    with pytest.raises(Exception):
        abs_model.detach_model(model_name="TEST")

    abs_model.model = {"TEST": None}
    abs_model.model_params = {"TEST": {"test_param": 1.2}}
    detached_abs_model = abs_model.detach_model(model_name="TEST")
    assert detached_abs_model.model["TEST"] is None
    assert detached_abs_model.model_params["TEST"]["test_param"] == 1.2

    with pytest.raises(NotImplementedError):
        abs_torch_model.cal_feature_importance(model_name="TEST", method="NOT_EXIST")

    with pytest.raises(Exception) as err:
        abs_torch_model.train()
    assert "_new_model must return an AbstractNN" in err.value.args[0]


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

    new_model = model.detach_model("Category Embedding")
    cnt_4 = new_model.count_params("Category Embedding", trainable_only=True)
    assert cnt_3 == cnt_4
    state = new_model.model["Category Embedding"].set_requires_grad(
        new_model.model["Category Embedding"], requires_grad=False
    )
    cnt_5 = new_model.count_params("Category Embedding", trainable_only=True)
    assert cnt_5 == 0
    new_model.model["Category Embedding"].set_requires_grad(
        new_model.model["Category Embedding"], state=state
    )
    cnt_6 = new_model.count_params("Category Embedding", trainable_only=True)
    assert cnt_3 == cnt_6

    with pytest.raises(Exception):
        new_model.model["Category Embedding"].set_requires_grad(
            new_model.model["Category Embedding"], state=state, requires_grad=False
        )


def test_config_not_loaded():
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    with pytest.raises(Exception):
        model = CatEmbed(trainer)


def test_require_model_exceptions():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    abs_model = NotImplementedModel(trainer, program="TEST_PROGRAM")
    abs_model.model = {}

    abs_model.required_models = lambda x: [x]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "is required by itself" in err.value.args[0]

    abs_model.required_models = lambda x: ["TEST_2"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "but is not trained" in err.value.args[0]

    abs_model.required_models = lambda x: ["EXTERN_TESTMODELBAS"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "from external model bases" in err.value.args[0]

    abs_model.required_models = lambda x: ["EXTERN_TESTMODELBASE_TESTMODEL_TESTARG"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "from external model bases" in err.value.args[0]

    abs_model.required_models = lambda x: ["EXTERN_TESTMODELBASE_TESTMODEL"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "mainly caused by model detaching" in err.value.args[0]

    abs_model.required_models = lambda x: ["EXTERN_TESTMODELBASE_TESTMODEL"]
    trainer.set_status(True)
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert (
        "but does not exist." in err.value.args[0]
        and "mainly caused by model detaching" not in err.value.args[0]
    )
    trainer.set_status(False)

    model = AutoGluon(trainer, model_subset=["Linear Regression"])
    model.model = {"Linear Regression": None}
    trainer.add_modelbases([model])
    abs_model.required_models = lambda x: ["EXTERN_AutoGluon_TESTMODEL"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "can not be detached from model base" in err.value.args[0]

    abs_model.required_models = lambda x: ["EXTERN_AutoGluon_Linear Regression_WRAP"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "does not support wrapping" in err.value.args[0]

    abs_model.required_models = lambda x: ["CANNOT_BE_RECOGNIZED"]
    with pytest.raises(Exception) as err:
        abs_model._get_required_models("TEST")
    assert "Unrecognized model name" in err.value.args[0]


def test_check_space():
    pytest_configure_trainer()
    trainer = pytest.test_model_trainer
    abs_model = NotImplementedModel(trainer, program="TEST_PROGRAM")
    abs_model._space = lambda model_name: [
        Real(low=0.1, high=1.5, name="not_exist_param")
    ]
    abs_model._initial_params = lambda model_name: {"exist_param": 1.1}
    with pytest.raises(Exception) as err:
        abs_model._check_space()
    assert "Defined spaces and initial values do not match" in err.value.args[0]


def test_get_sequential():
    layers = get_sequential(
        [32, 32],
        n_inputs=4,
        n_outputs=2,
        act_func=nn.ReLU,
        dropout=0,
        use_norm=True,
        norm_type="batch",
        out_activate=False,
        out_norm_dropout=False,
        adaptive_dropout=False,
    )
    # norm->linear->act->norm->linear->act->linear
    assert len(layers) == 7

    layers = get_sequential(
        [32, 32],
        n_inputs=4,
        n_outputs=2,
        act_func=nn.ReLU,
        dropout=0.1,
        use_norm=True,
        norm_type="batch",
        out_activate=False,
        out_norm_dropout=False,
        adaptive_dropout=False,
    )
    # norm->linear->act->dp->norm->linear->act->dp->linear
    assert len(layers) == 9

    layers = get_sequential(
        [32, 32],
        n_inputs=4,
        n_outputs=2,
        act_func=nn.ReLU,
        dropout=0.1,
        use_norm=True,
        norm_type="batch",
        out_activate=True,
        out_norm_dropout=True,
        adaptive_dropout=True,
    )
    # norm->linear->act->dp->norm->linear->act->dp->norm->linear->act->dp
    assert len(layers) == 12

    layers = get_sequential(
        [32, 32],
        n_inputs=4,
        n_outputs=2,
        act_func=nn.ReLU,
        dropout=0.1,
        use_norm=False,
        norm_type="layer",
        out_activate=True,
        out_norm_dropout=True,
        adaptive_dropout=True,
    )
    # linear->act->dp->linear->act->dp->linear->act->dp
    assert len(layers) == 9

    layers = get_sequential(
        [],
        n_inputs=4,
        n_outputs=2,
        act_func=nn.LeakyReLU,
        dropout=0.1,
        use_norm=True,
        norm_type="batch",
        out_activate=True,
        out_norm_dropout=True,
        adaptive_dropout=True,
    )
    # norm->linear->act->dp
    assert len(layers) == 4


def test_adaptive_dropout():
    seq = nn.Sequential()
    seq.add_module("id", nn.Identity())
    seq.add_module("dp", AdaptiveDropout(p=0.0))
    x = torch.ones((10, 1))
    seq.train()
    out = seq(x)
    assert torch.sum(out) == 10

    with KeepDropout(p=1.0):
        seq.train()
        out = seq(x)
        assert torch.sum(out) == 0
        seq.eval()
        out = seq(x)
        assert torch.sum(out) == 0

    seq.train()
    out = seq(x)
    assert torch.sum(out) == 10


def test_custom_dataset():
    array = np.arange(10).reshape(-1, 1)
    tensor = torch.arange(10).view(-1, 1)
    df = pd.DataFrame(columns=["test"], data=array, index=np.arange(10))
    inv_df = df.copy()
    inv_df.loc[:, "test"] = array[::-1]

    dataset = DictNDArrayDataset({"first": array, "second": array[::-1]})
    assert isinstance(dataset[0], dict)
    assert dataset[3]["first"] == 3 and dataset[3]["second"] == 6

    dataset = DictDataFrameDataset({"first": df, "second": inv_df})
    assert isinstance(dataset[0], dict)
    assert (
        dataset[3]["first"].loc["test"] == 3 and dataset[3]["second"].loc["test"] == 6
    )

    dataset = DictMixDataset({"first": df, "second": array[::-1], "third": tensor})
    assert isinstance(dataset[0], dict)
    assert (
        dataset[3]["first"].loc["test"] == 3
        and dataset[3]["second"] == 6
        and dataset[3]["third"][0] == 3
    )


def test_get_loss_fn():
    with pytest.raises(Exception):
        _ = AbstractNN.get_loss_fn(loss="TEST", task="binary")
    with pytest.raises(Exception):
        _ = AbstractNN.get_loss_fn(loss="TEST", task="multiclass")
    with pytest.raises(Exception):
        _ = AbstractNN.get_loss_fn(loss="mse", task="TEST")

    a = torch.tensor([-0.1, 1.1, 0.3, 0.2], dtype=torch.float32)
    b = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    nm = AbstractNN.get_output_norm(task="binary")
    fn = AbstractNN.get_loss_fn(loss="cross_entropy", task="binary")
    assert torch.allclose(
        torch.nn.functional.binary_cross_entropy_with_logits(a, b), fn(a, b)
    )
    assert torch.all(nm(a) < 1) and torch.all(nm(a) > 0)

    fn = AbstractNN.get_loss_fn(loss="mse", task="regression")
    nm = AbstractNN.get_output_norm(task="regression")
    assert torch.allclose(torch.nn.functional.mse_loss(a, b), fn(a, b))
    assert torch.equal(a, nm(a))
    fn = AbstractNN.get_loss_fn(loss="mae", task="regression")
    nm = AbstractNN.get_output_norm(task="regression")
    assert torch.allclose(torch.nn.functional.l1_loss(a, b), fn(a, b))
    assert torch.equal(a, nm(a))

    a = torch.tensor(
        [[-0.1, 1.1, 0.3, 0.2], [-0.2, 0.8, 0.1, 0.2], [10, -10, 0, 0.2]],
        dtype=torch.float32,
    ).T
    b = torch.tensor([0, 2, 0, 1], dtype=torch.float32).long()
    fn = AbstractNN.get_loss_fn(loss="cross_entropy", task="multiclass")
    nm = AbstractNN.get_output_norm(task="multiclass")
    assert torch.allclose(torch.nn.functional.cross_entropy(a, b), fn(a, b))
    assert torch.all(nm(a) < 1) and torch.all(nm(a) > 0)
