import os
from import_utils import *
import tabensemb
import numpy as np
from tabensemb.trainer import Trainer, load_trainer, save_trainer
from tabensemb.model import *
from tabensemb.data import AbstractDeriver
from tabensemb.data.dataderiver import deriver_mapping
from tabensemb.utils import HiddenPltShow
from tabensemb.config import UserConfig
from tabensemb.data.datasplitter import RandomSplitter
import torch
import pytest
import matplotlib
import shutil
from torch import nn


class TrainerCategoricalDeriver(AbstractDeriver):
    """
    This is an example of deriving categorical features.
    """

    def _required_cols(self):
        return []

    def _required_kwargs(self):
        return []

    def _defaults(self):
        return dict(stacked=True, intermediate=False, is_continuous=False)

    def _derived_names(self):
        return ["derived_cat_0", "derived_cat_1"]

    def _derive(self, df, datamodule):
        col_0 = df[datamodule.cont_feature_names[0]].values.flatten().reshape(-1, 1)
        if datamodule.training:
            self.mean = np.nanmean(col_0)
        derived_cat_0 = (col_0 > self.mean).astype(int)
        derived_cat_1 = np.array(
            [f"category_{i}" for i in derived_cat_0.flatten()]
        ).reshape(-1, 1)
        return np.concatenate([derived_cat_0, derived_cat_1], axis=-1)


deriver_mapping["TrainerCategoricalDeriver"] = TrainerCategoricalDeriver


def pytest_configure_trainer():
    if getattr(pytest, "trainer_configure_excuted", False):
        return
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={
            "data_splitter": "RandomSplitter",
            "data_derivers": [
                (
                    "RelativeDeriver",
                    {
                        "stacked": True,
                        "absolute_col": "cont_0",
                        "relative2_col": "cont_1",
                        "intermediate": False,
                        "derived_name": "derived_cont",
                    },
                ),
                (
                    "RelativeDeriver",
                    {
                        "stacked": False,
                        "absolute_col": "cont_0",
                        "relative2_col": "cont_1",
                        "intermediate": False,
                        "derived_name": "derived_cont_unstacked",
                    },
                ),
                (
                    "SampleWeightDeriver",
                    {
                        "stacked": True,
                        "intermediate": False,
                        "derived_name": "sample_weight",
                    },
                ),
                (
                    "TrainerCategoricalDeriver",
                    {
                        "stacked": True,
                        "intermediate": False,
                        "derived_name": "derived_cat",
                    },
                ),
            ],
            "data_processors": [
                ["CategoricalOrdinalEncoder", {}],
                ["NaNFeatureRemover", {}],
                ["VarianceFeatureSelector", {"thres": 1}],
                ["IQRRemover", {}],  # Note the remover here
                ["SampleDataAugmenter", {}],  # Note the augmenter here
                ["StandardScaler", {}],
            ],
        },
    )
    trainer.load_data()
    pytest.test_trainer_trainer = trainer
    pytest.test_trainer_trainer_configure_excuted = True


@pytest.mark.order(2)
def test_init_trainer():
    pytest_configure_trainer()
    pytest.test_trainer_trainer.summarize_setting()


@pytest.mark.order(3)
def test_init_models():
    trainer = pytest.test_trainer_trainer
    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    pytest.models = models


def test_init_trainer_without_feature():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={"feature_names_type": {}},
    )
    trainer.load_data()


def test_save_trainer():
    save_trainer(pytest.test_trainer_trainer)


def test_train_without_bayes():
    pytest.test_trainer_trainer.train()


def test_train_binary():
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config("sample", manual_config={"label_name": ["target_binary"]})
    # trainer.load_config("sample")
    trainer.load_data()
    assert trainer.datamodule.task == "binary"

    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    trainer.train()

    def test_one_modelbase(modelbase, model_name):
        res = modelbase.predict(
            trainer.df, derived_data=trainer.derived_data, model_name=model_name
        )
        assert np.all(np.mod(res, 1) == 0)
        res_prob = modelbase.predict_proba(
            trainer.df,
            derived_data=trainer.derived_data,
            model_name=model_name,
            proba=False,  # This will be ignored
        )
        assert np.all(np.mod(res_prob, 1) != 0)
        assert np.all(res_prob > 0) and np.all(res_prob < 1)

        assert res_prob.shape[0] == len(trainer.df) and res_prob.shape[1] == 1

    test_one_modelbase(models[0], "Category Embedding")
    test_one_modelbase(models[1], "TabMlp")
    test_one_modelbase(models[2], "Linear Regression")
    test_one_modelbase(models[3], "Category Embedding")
    l = trainer.get_leaderboard()

    df = trainer.df.copy()
    df.loc[:, "target_binary"] = np.array([f"test_{x}" for x in df["target_binary"]])
    trainer.clear_modelbase()
    trainer.datamodule.set_data(
        trainer.datamodule.categories_inverse_transform(df),
        cont_feature_names=trainer.cont_feature_names,
        cat_feature_names=trainer.cat_feature_names,
        label_name=trainer.label_name,
    )
    assert trainer.datamodule.task == "binary"
    model = PytorchTabular(trainer, model_subset=["Category Embedding"])
    trainer.add_modelbases([model])
    trainer.train()
    res = model.predict(
        trainer.df, derived_data=trainer.derived_data, model_name="Category Embedding"
    )
    assert res.dtype == object


def test_train_multiclass():
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config("sample", manual_config={"label_name": ["target_multi_class"]})
    # trainer.load_config("sample")
    trainer.load_data()
    assert trainer.datamodule.task == "multiclass"

    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    trainer.train()

    def test_one_modelbase(modelbase, model_name):
        res = modelbase.predict(
            trainer.df, derived_data=trainer.derived_data, model_name=model_name
        )
        assert np.all(np.mod(res, 1) == 0)
        res_prob = modelbase.predict_proba(
            trainer.df, derived_data=trainer.derived_data, model_name=model_name
        )
        assert np.all(np.mod(res_prob, 1) != 0)
        assert np.all(res_prob > 0) and np.all(res_prob < 1)

        assert (
            res_prob.shape[0] == len(trainer.df)
            and res_prob.shape[1] == trainer.datamodule.n_classes[0]
        )

    test_one_modelbase(models[0], "Category Embedding")
    test_one_modelbase(models[1], "TabMlp")
    test_one_modelbase(models[2], "Linear Regression")
    test_one_modelbase(models[3], "Category Embedding")
    l = trainer.get_leaderboard()

    df = trainer.df.copy()
    df.loc[:, "target_multi_class"] = np.array(
        [f"test_{x}" for x in df["target_multi_class"]]
    )
    trainer.clear_modelbase()
    trainer.datamodule.set_data(
        trainer.datamodule.categories_inverse_transform(df),
        cont_feature_names=trainer.cont_feature_names,
        cat_feature_names=trainer.cat_feature_names,
        label_name=trainer.label_name,
    )
    assert trainer.datamodule.task == "multiclass"
    model = PytorchTabular(trainer, model_subset=["Category Embedding"])
    trainer.add_modelbases([model])
    trainer.train()
    res = model.predict(
        trainer.df, derived_data=trainer.derived_data, model_name="Category Embedding"
    )
    assert res.dtype == object


@pytest.mark.order(after="test_train_without_bayes")
def test_get_leaderboard():
    no_model_trainer = Trainer(device="cpu")
    no_model_trainer.load_config("sample")
    with pytest.raises(Exception) as err:
        no_model_trainer.get_leaderboard()
    assert "No modelbase available" in err.value.args[0]

    trainer = pytest.test_trainer_trainer
    l0 = trainer.get_leaderboard(test_data_only=True)
    assert (
        "Training" not in l0.columns
        and "Validation" not in l0.columns
        and "Testing" not in l0.columns
        and "RMSE" in l0.columns
    )
    l = trainer.get_leaderboard()
    pytest.leaderboard_init = l

    with pytest.warns(UserWarning):
        # No cv exists
        trainer.get_approx_cv_leaderboard(leaderboard=l, save=False)
    with pytest.raises(Exception) as err:
        trainer._read_cv_leaderboards()
    assert "not found" in err.value.args[0]

    with pytest.raises(Exception) as err:
        _ = trainer.get_leaderboard(cross_validation=2, load_from_previous=True)
    assert "No previous state to load from" in err.value.args[0]

    best_model, best_metric = trainer.get_best_model()


@pytest.mark.order(after="test_get_leaderboard")
def test_load_trainer():
    trainer = pytest.test_trainer_trainer
    root = trainer.project_root + "_rename_test"
    old_root = trainer.project_root
    shutil.copytree(trainer.project_root, root)
    shutil.rmtree(trainer.project_root)
    trainer = load_trainer(os.path.join(root, "trainer.pkl"))
    l2 = trainer.get_leaderboard()
    cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
    assert np.allclose(
        pytest.leaderboard_init[cols].values.astype(float),
        l2[cols].values.astype(float),
    ), f"Reloaded local trainer does not get consistent results."
    shutil.copytree(root, old_root)
    shutil.rmtree(root)


@pytest.mark.order(after="test_train_without_bayes")
def test_predict_function():
    trainer = pytest.test_trainer_trainer
    models = trainer.modelbases
    x_test = trainer.datamodule.X_test
    d_test = trainer.datamodule.D_test
    assert len(models) > 0
    for model in models:
        model_name = model.model_subset[0]
        x_test_wo_label = x_test.copy()
        for label in trainer.label_name:
            del x_test_wo_label[label]
        pred = model.predict(x_test_wo_label, model_name=model_name)
        direct_pred = model._predict(
            x_test_wo_label, derived_data=d_test, model_name=model_name
        )
        pred_w_label = model.predict(x_test, model_name=model_name)
        assert np.allclose(
            pred, direct_pred
        ), f"{model.__class__.__name__} does not get consistent inference results."
        assert np.allclose(pred, pred_w_label)
        with pytest.raises(Exception) as err:
            model.predict_proba(x_test, model_name=model_name)
        assert "Calling predict_proba on regression models" in err.value.args[0]


@pytest.mark.order(after="test_train_without_bayes")
def test_detach_model():
    trainer = pytest.test_trainer_trainer
    model_trainer = trainer.detach_model(
        program="CatEmbed", model_name="Category Embedding"
    )
    model_trainer.train()
    direct_pred = trainer.get_modelbase("CatEmbed")._predict(
        trainer.datamodule.X_test,
        derived_data=trainer.datamodule.D_test,
        model_name="Category Embedding",
    )
    detached_pred = model_trainer.get_modelbase("CatEmbed_Category Embedding")._predict(
        model_trainer.datamodule.X_test,
        derived_data=model_trainer.datamodule.D_test,
        model_name="Category Embedding",
    )
    assert np.allclose(
        detached_pred, direct_pred
    ), f"The detached model does not get consistent results."
    pytest.model_trainer = model_trainer


@pytest.mark.order(after="test_detach_model")
def test_cuda():
    model_trainer = pytest.model_trainer
    if torch.cuda.is_available():
        model_trainer.set_device("cuda")
        model_trainer.train()
    else:
        print(f"Skipping cuda tests since torch.cuda.is_available() is False.")


@pytest.mark.order(after="test_detach_model")
def test_train_after_set_feature_names():
    model_trainer = pytest.model_trainer
    model_trainer.datamodule.set_feature_names(
        model_trainer.datamodule.cont_feature_names[:3]
    )
    model_trainer.train()


@pytest.mark.order(after="test_detach_model")
def test_bayes_opt():
    model_trainer = pytest.model_trainer
    model_trainer.args["bayes_opt"] = True
    model_trainer.get_leaderboard(cross_validation=2)
    model_trainer.args["bayes_opt"] = False


@pytest.mark.order(after="test_detach_model")
def test_continue_previous():
    model_trainer = pytest.model_trainer
    l0 = model_trainer.get_leaderboard(cross_validation=1, split_type="random")
    l1 = model_trainer.get_leaderboard(cross_validation=2, load_from_previous=True)
    l2 = model_trainer.get_leaderboard(cross_validation=2, split_type="random")
    with pytest.raises(Exception) as err:
        _ = model_trainer.get_leaderboard(cross_validation=1, load_from_previous=True)
    assert "The loaded state is incompatible" in err.value.args[0]

    cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
    assert np.allclose(
        l1[cols].values.astype(float), l2[cols].values.astype(float)
    ), f"load_from_previous does not get consistent results."


@pytest.mark.order(after="test_detach_model")
def test_inspect():
    trainer = pytest.model_trainer
    model = trainer.get_modelbase("CatEmbed_Category Embedding")

    print(f"\n-- Inspect model --\n")
    direct_inspect = model.inspect_attr(
        "Category Embedding", ["hidden_representation", "head"]
    )
    train_inspect = model.inspect_attr(
        "Category Embedding",
        ["hidden_representation"],
        trainer.df.loc[trainer.train_indices, :],
    )
    train_inspect_with_derived = model.inspect_attr(
        "Category Embedding",
        ["hidden_representation"],
        df=trainer.df.loc[trainer.train_indices, :],
        derived_data=trainer.datamodule.get_derived_data_slice(
            trainer.derived_data, trainer.train_indices
        ),
    )
    assert np.allclose(
        direct_inspect["train"]["prediction"],
        train_inspect["USER_INPUT"]["prediction"],
    )
    assert np.allclose(
        direct_inspect["train"]["prediction"],
        train_inspect_with_derived["USER_INPUT"]["prediction"],
    )
    assert np.allclose(
        direct_inspect["train"]["hidden_representation"],
        train_inspect["USER_INPUT"]["hidden_representation"],
    )
    assert np.allclose(
        direct_inspect["train"]["hidden_representation"],
        train_inspect_with_derived["USER_INPUT"]["hidden_representation"],
    )
    assert not direct_inspect["train"]["hidden_representation"].shape[
        0
    ] == direct_inspect["val"]["hidden_representation"].shape[0] or np.allclose(
        direct_inspect["train"]["hidden_representation"],
        direct_inspect["val"]["hidden_representation"],
    )
    assert isinstance(direct_inspect["train"]["head"], nn.Module)
    assert direct_inspect["train"]["head"].bias.device.type == "cpu"


def test_trainer_label_missing():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={
            "data_splitter": "RandomSplitter",
            "label_name": ["cont_1"],
        },
    )
    with pytest.raises(Exception) as err:
        trainer.load_data()
    assert "Label missing" in err.value.args[0]


def test_trainer_multitarget():
    with pytest.warns(
        UserWarning, match=r"Multi-target task is currently experimental.*?"
    ):
        print(f"\n-- Loading trainer --\n")
        configfile = "sample"
        tabensemb.setting["debug_mode"] = True
        trainer = Trainer(device="cpu")
        trainer.load_config(
            configfile,
            manual_config={
                "data_splitter": "RandomSplitter",
                "label_name": ["target", "cont_0"],
            },
        )
        with pytest.warns(UserWarning):
            trainer.load_data()
        trainer.summarize_setting()

        print(f"\n-- Initialize models --\n")

        with pytest.raises(Exception) as err:
            WideDeep(trainer, model_subset=["TabMlp"])
        assert "does not support multi-target tasks" in err.value.args[0]

        models = [
            PytorchTabular(trainer, model_subset=["Category Embedding"]),
            AutoGluon(trainer, model_subset=["Linear Regression"]),
            CatEmbed(trainer, model_subset=["Category Embedding"]),
        ]
        trainer.add_modelbases(models)

        print(f"\n-- Pickling --\n")
        save_trainer(trainer)

        print(f"\n-- Training without bayes --\n")
        trainer.train()

        print(f"\n-- Leaderboard --\n")
        l = trainer.get_leaderboard()

        print(f"\n-- Prediction consistency --\n")
        x_test = trainer.datamodule.X_test
        d_test = trainer.datamodule.D_test
        for model in models:
            model_name = model.model_subset[0]
            pred = model.predict(x_test, model_name=model_name)
            direct_pred = model._predict(
                x_test, derived_data=d_test, model_name=model_name
            )
            assert np.allclose(
                pred, direct_pred
            ), f"{model.__class__.__name__} does not get consistent inference results."

        print(f"\n-- Detach modelbase --\n")
        model_trainer = trainer.detach_model(
            program="CatEmbed", model_name="Category Embedding"
        )
        model_trainer.train()
        direct_pred = trainer.get_modelbase("CatEmbed")._predict(
            trainer.datamodule.X_test,
            derived_data=trainer.datamodule.D_test,
            model_name="Category Embedding",
        )
        detached_pred = model_trainer.get_modelbase(
            "CatEmbed_Category Embedding"
        )._predict(
            model_trainer.datamodule.X_test,
            derived_data=model_trainer.datamodule.D_test,
            model_name="Category Embedding",
        )
        assert np.allclose(
            detached_pred, direct_pred
        ), f"The detached model does not get consistent results."

        print(f"\n-- pytorch cuda functionality --\n")
        if torch.cuda.is_available():
            model_trainer.set_device("cuda")
            model_trainer.train()
        else:
            print(f"Skipping cuda tests since torch.cuda.is_available() is False.")

        print(
            f"\n-- Training after set_feature_names and without categorical features --\n"
        )
        with pytest.warns(UserWarning):
            model_trainer.datamodule.set_feature_names(
                model_trainer.datamodule.cont_feature_names[:10]
            )
        model_trainer.train()

        print(f"\n-- Bayes optimization --\n")
        model_trainer.args["bayes_opt"] = True
        model_trainer.train()

        print(f"\n-- Load local trainer --\n")
        root = trainer.project_root + "_rename_test"
        shutil.copytree(trainer.project_root, root)
        shutil.rmtree(trainer.project_root)
        trainer = load_trainer(os.path.join(root, "trainer.pkl"))
        l2 = trainer.get_leaderboard()
        cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
        assert np.allclose(
            l[cols].values.astype(float), l2[cols].values.astype(float)
        ), f"Reloaded local trainer does not get consistent results."


def test_feature_importance():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={
            "data_splitter": "RandomSplitter",
            "data_derivers": [
                (
                    "RelativeDeriver",
                    {
                        "stacked": True,
                        "absolute_col": "cont_0",
                        "relative2_col": "cont_1",
                        "intermediate": False,
                        "derived_name": "derived_cont",
                    },
                ),
                (
                    "RelativeDeriver",
                    {
                        "stacked": False,
                        "absolute_col": "cont_0",
                        "relative2_col": "cont_1",
                        "intermediate": False,
                        "derived_name": "unstacked_derived_cont",
                    },
                ),
            ],
        },
    )
    trainer.load_data()

    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        CatEmbed(
            trainer,
            model_subset=["Category Embedding", "Require Model PyTabular CatEmbed"],
        ),
    ]
    trainer.add_modelbases(models)

    trainer.train()

    absmodel_perm = trainer.cal_feature_importance(
        program="PytorchTabular", model_name="Category Embedding", method="permutation"
    )
    absmodel_shap = trainer.cal_feature_importance(
        program="PytorchTabular", model_name="Category Embedding", method="shap"
    )

    torchmodel_perm = trainer.cal_feature_importance(
        program="CatEmbed", model_name="Category Embedding", method="permutation"
    )
    np.random.seed(0)
    with pytest.warns(
        UserWarning, match=r"shap.DeepExplainer cannot handle categorical features"
    ):
        torchmodel_shap = trainer.cal_feature_importance(
            program="CatEmbed", model_name="Category Embedding", method="shap"
        )
    np.random.seed(0)
    with pytest.warns(
        UserWarning, match=r"shap.DeepExplainer cannot handle categorical features"
    ):
        torchmodel_shap_direct = trainer.cal_shap(
            program="CatEmbed", model_name="Category Embedding"
        )

    with pytest.raises(Exception) as err:
        with pytest.warns(
            UserWarning, match=r"shap.DeepExplainer cannot handle categorical features"
        ):
            trainer.cal_feature_importance(
                program="CatEmbed",
                model_name="Require Model PyTabular CatEmbed",
                method="shap",
            )
    assert "models that require other models is not supported" in err.value.args[0]

    assert len(absmodel_perm[0]) == len(absmodel_perm[1])
    assert np.all(np.abs(absmodel_perm[0]) > 1e-8)

    assert len(absmodel_shap[0]) == len(absmodel_shap[1])
    assert np.all(np.abs(absmodel_shap[0]) > 1e-8)

    assert len(torchmodel_perm[0]) == len(torchmodel_perm[1])
    cont_index = np.array(
        [torchmodel_perm[1].index(i) for i in trainer.cont_feature_names]
    )
    cat_index = np.array(
        [torchmodel_perm[1].index(i) for i in trainer.cat_feature_names]
    )
    other_index = np.setdiff1d(
        np.setdiff1d(np.arange(len(torchmodel_perm[1])), cont_index), cat_index
    )
    assert (
        "unstacked_derived_cont" in torchmodel_perm[1]
        and torchmodel_shap[1].index("unstacked_derived_cont") in other_index
    )
    assert np.all(np.abs(torchmodel_perm[0][cont_index]) > 1e-8)
    assert np.all(np.abs(torchmodel_perm[0][cat_index]) > 1e-8)
    # Unused data does not have feature importance.
    assert np.all(np.abs(torchmodel_perm[0][other_index]) < 1e-8)

    assert len(torchmodel_shap[0]) == len(torchmodel_shap[1])
    cont_index = np.array(
        [torchmodel_shap[1].index(i) for i in trainer.cont_feature_names]
    )
    cat_index = np.array(
        [torchmodel_shap[1].index(i) for i in trainer.cat_feature_names]
    )
    other_index = np.setdiff1d(
        np.setdiff1d(np.arange(len(torchmodel_shap[1])), cont_index), cat_index
    )
    assert (
        "unstacked_derived_cont" in torchmodel_perm[1]
        and torchmodel_shap[1].index("unstacked_derived_cont") in other_index
    )
    assert np.all(np.abs(torchmodel_shap[0][cont_index]) > 1e-8)
    # Categorical features does not have gradients, therefore does not have shap values using DeepExplainer.
    assert np.all(np.abs(torchmodel_shap[0][cat_index]) < 1e-8)
    # Unused data does not have feature importance.
    assert np.all(np.abs(torchmodel_shap[0][other_index]) < 1e-8)

    assert np.allclose(torchmodel_shap[0], torchmodel_shap_direct)


@pytest.mark.order(after="test_train_without_bayes")
def test_copy():
    trainer = pytest.test_trainer_trainer
    copied_trainer = trainer.copy()
    copied_leaderboard = copied_trainer.get_leaderboard()
    cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
    assert np.allclose(
        pytest.leaderboard_init[cols].values.astype(float),
        copied_leaderboard[cols].values.astype(float),
    )


@pytest.mark.order(after="test_train_without_bayes")
def test_finetune():
    trainer = pytest.test_trainer_trainer
    models = trainer.modelbases
    with pytest.warns(UserWarning, match=r"AutoGluon does not support warm_start.*?"):
        for model in models:
            model.fit(
                trainer.datamodule.categories_inverse_transform(trainer.df),
                trainer.cont_feature_names,
                trainer.cat_feature_names,
                trainer.label_name,
                derived_data=trainer.derived_data,
                warm_start=True,
            )


@pytest.mark.order(after="test_train_without_bayes")
def test_plots():
    matplotlib.rc("text", usetex=False)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    trainer = pytest.test_trainer_trainer

    with HiddenPltShow():
        print(f"\n-- Scatter --\n")
        trainer.plot_scatter(x_col="cont_1", y_col="cont_2")
        trainer.plot_scatter(
            x_col="cont_1",
            y_col="cont_2",
            select_by_value_kwargs={"selection": {"cat_1": [1, 2, 3]}},
        )

        print(f"\n-- multiple scatter --\n")
        trainer.plot_on_one_axes(
            meth_name="plot_scatter",
            meth_kwargs_ls=[
                {},
                dict(select_by_value_kwargs={"selection": {"cat_1": [1, 2, 3]}}),
            ],
            meth_fix_kwargs=dict(x_col="cont_1", y_col="cont_2"),
        )
        trainer.plot_on_one_axes(
            meth_name="plot_scatter",
            meth_kwargs_ls=[
                dict(x_col="cont_1", y_col="cont_2", scatter_kwargs={"label": "all"}),
                dict(
                    x_col="cont_1",
                    y_col="cont_2",
                    select_by_value_kwargs={"selection": {"cat_1": [1, 2, 3]}},
                    scatter_kwargs={"label": "cat_1=1,2,3"},
                ),
            ],
            legend=True,
        )

        trainer.plot_subplots(
            ls=[
                [
                    dict(
                        x_col="cont_1", y_col="cont_2", scatter_kwargs={"label": "all"}
                    ),
                    dict(
                        x_col="cont_1",
                        y_col="cont_2",
                        select_by_value_kwargs={"selection": {"cat_1": [1, 2, 3]}},
                        scatter_kwargs={"label": "cat_1=1,2,3"},
                    ),
                ]
            ]
            * 2,
            ls_kwarg_name="meth_kwargs_ls",
            meth_name="plot_on_one_axes",
            with_title=False,
            meth_fix_kwargs=dict(meth_name="plot_scatter", legend=True, twin=True),
            ylabel="TEST 1",
            twin_ylabel="TEST 2",
        )

        print(f"\n-- PDF --\n")
        trainer.plot_pdf(
            feature="cont_1", select_by_value_kwargs={"selection": {"cat_1": [1, 2]}}
        )

        print(f"\n-- KDE --\n")
        trainer.plot_kde_all()
        trainer.plot_kde(
            feature="cont_2", select_by_value_kwargs={"selection": {"cat_1": [1, 2]}}
        )

        print(f"\n-- Presence ratio --\n")
        trainer.plot_presence_ratio(order="type", barplot_kwargs={"orient": "h"})
        trainer.plot_presence_ratio(order="ratio", barplot_kwargs={"orient": "v"})

        print(f"\n-- fill rating --\n")
        trainer.plot_fill_rating()
        trainer.plot_fill_rating(category=trainer.cat_feature_names[0])

        print(f"\n-- PCA 2d --\n")
        trainer.plot_pca_2d_visual()
        trainer.plot_pca_2d_visual(category="cat_4")

        print(f"\n-- Correlation --\n")
        trainer.plot_corr(imputed=True)
        trainer.plot_corr(imputed=False)
        trainer.plot_corr_with_label(order="descending")

        print(f"\n-- hist --\n")
        trainer.plot_hist_all(imputed=True, kde=True)
        trainer.plot_hist_all(imputed=False)
        trainer.plot_hist(feature="cont_0")
        trainer.plot_hist(feature="cont_0", category=trainer.cat_feature_names[0])

        print(f"\n-- Pair --\n")
        trainer.plot_pairplot()

        print(f"\n-- Truth pred --\n")
        trainer.plot_truth_pred_all(program="CatEmbed", log_trans=True)
        trainer.plot_truth_pred_all(program="CatEmbed", log_trans=False)
        trainer.plot_truth_pred(
            program="CatEmbed", model_name="Category Embedding", log_trans=True
        )

        print(f"\n-- Feature box --\n")
        trainer.plot_feature_box(imputed=False)
        trainer.plot_feature_box(imputed=True)

        print(f"\n-- Partial Err --\n")
        trainer.plot_partial_err_all(
            program="CatEmbed", model_name="Category Embedding"
        )
        trainer.plot_partial_err(
            program="CatEmbed", model_name="Category Embedding", feature="cont_0"
        )

        print(f"\n-- Err histogram --\n")
        trainer.plot_err_hist(
            program="CatEmbed",
            model_name="Category Embedding",
            category=trainer.cat_feature_names[0],
        )

        print(f"\n-- Importance --\n")
        trainer.plot_feature_importance(program="WideDeep", model_name="TabMlp")
        with pytest.warns(
            UserWarning, match=r"shap.DeepExplainer cannot handle categorical features"
        ):
            trainer.plot_feature_importance(
                program="CatEmbed", model_name="Category Embedding", method="shap"
            )

        print(f"\n-- PDP --\n")
        trainer.plot_partial_dependence_all(
            program="WideDeep",
            model_name="TabMlp",
            n_bootstrap=1,
            grid_size=2,
            log_trans=True,
        )
        trainer.plot_partial_dependence_all(
            program="WideDeep",
            model_name="TabMlp",
            n_bootstrap=2,
            grid_size=2,
            log_trans=False,
            refit=False,
        )
        trainer.plot_partial_dependence(
            program="WideDeep",
            model_name="TabMlp",
            feature="cont_0",
            n_bootstrap=2,
            grid_size=2,
            log_trans=False,
            refit=False,
        )


def test_exception_during_bayes_opt(capfd):
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(configfile)
    trainer.load_data()
    models = [PytorchTabular(trainer, model_subset=["TabNet"])]
    trainer.add_modelbases(models)

    def _train_just_raise(*args, **kwargs):
        raise Exception("CUDA error: device-side assert triggered")

    models[0]._train_single_model = _train_just_raise
    trainer.args["bayes_opt"] = True
    with pytest.raises(ValueError) as err:
        trainer.train()
    assert "Unfortunately" in err.value.args[0]

    def _train_just_raise(*args, **kwargs):
        raise RuntimeError("Normal error")

    models[0]._train_single_model = _train_just_raise
    with pytest.raises(RuntimeError):
        trainer.train()
    out, err = capfd.readouterr()
    assert "Returning a large value instead" in out


@pytest.mark.order(after="test_train_without_bayes")
def test_exceptions():
    trainer = pytest.test_trainer_trainer
    with pytest.raises(Exception) as err:
        trainer.set_device("UNKNOWN_DEVICE")
    assert "is an invalid selection" in err.value.args[0]

    with pytest.raises(Exception) as err:
        trainer.add_modelbases(
            [PytorchTabular(trainer, model_subset=["Category Embedding"])]
        )
    assert "Conflicted model base names" in err.value.args[0]


def test_user_input_config():
    trainer = Trainer(device="cpu")
    cfg = UserConfig("sample")
    with pytest.warns(UserWarning):
        trainer.load_config(
            cfg, manual_config={"epoch": 2}, project_root_subfolder="test"
        )
    assert trainer.args["epoch"] != 2
    assert "test" in trainer.project_root

    trainer.load_config("sample", manual_config={"epoch": 2})
    assert trainer.args["epoch"] == 2

    trainer.load_config(
        "sample",
        manual_config={
            "SPACEs": {
                "lr": {
                    "type": "Real",
                    "low": 1e-4,
                    "high": 0.05,
                    "prior": "log-uniform",
                },
                "hidden_dim": {
                    "type": "Integer",
                    "low": 16,
                    "high": 64,
                    "prior": "uniform",
                    "dtype": int,
                },
                "batch_size": {
                    "type": "Categorical",
                    "categories": [64, 128, 256, 512, 1024, 2048],
                },
            }
        },
    )
    _ = trainer.SPACE

    trainer.load_config(
        "sample",
        manual_config={"SPACEs": {"lr": {"type": "UNKNOWN"}}},
    )
    with pytest.raises(Exception) as err:
        _ = trainer.SPACE
    assert "Invalid type of skopt space" in err.value.args[0]


def test_cmd_arguments_with_manual_config(mocker):
    mocker.patch(
        "sys.argv",
        [
            "NOT_USED",  # The first arg is the positional name of the script
            "--base",
            "sample",
            "--epoch",
            "2",
            "--bayes_opt",
            "--data_imputer",
            "GainImputer",
            "--split_ratio",
            "0.3",
            "0.1",
            "0.6",
        ],
    )
    trainer = Trainer(device="cpu")
    trainer.load_config(manual_config={"patience": 2})
    assert trainer.args["epoch"] == 2
    assert trainer.args["bayes_opt"]
    assert trainer.args["data_imputer"] == "GainImputer"
    assert trainer.args["patience"] == 2
    assert all([x == y for x, y in zip(trainer.args["split_ratio"], [0.3, 0.1, 0.6])])


def test_train_part_of_modelbases():
    trainer = pytest.test_trainer_trainer
    trainer.train(programs=["PytorchTabular"])
    with pytest.warns(UserWarning):
        trainer.train(programs=[])


def test_uci_iris_multiclass():
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    cfg = UserConfig.from_uci("Iris", column_names=iris_columns, datafile_name="iris")
    trainer.load_config(cfg)
    trainer.load_data()
    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    l = trainer.get_leaderboard(cross_validation=2, split_type="random")
    os.remove(os.path.join(tabensemb.setting["default_data_path"], "iris.csv"))


def test_uci_autompg_regression():
    tabensemb.setting["debug_mode"] = True
    cfg = UserConfig.from_uci("Auto MPG", column_names=mpg_columns, sep=r"\s+")
    trainer = Trainer(device="cpu")
    trainer.load_config(cfg)
    trainer.load_data()
    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    trainer.train()
    l = trainer.get_leaderboard()
    os.remove(os.path.join(tabensemb.setting["default_data_path"], "auto-mpg.csv"))


def test_uci_adult_binary():
    tabensemb.setting["debug_mode"] = True
    with pytest.warns(UserWarning):
        cfg = UserConfig.from_uci("Adult", column_names=adult_columns, sep=", ")
    trainer = Trainer(device="cpu")
    trainer.load_config(cfg)
    trainer.load_data()
    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    trainer.train()
    l = trainer.get_leaderboard()
    os.remove(os.path.join(tabensemb.setting["default_data_path"], "adult.csv"))
    os.remove(os.path.join(tabensemb.setting["default_data_path"], "Adult.zip"))
