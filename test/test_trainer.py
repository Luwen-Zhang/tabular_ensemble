import unittest
from import_utils import *
import tabensemb
from tabensemb.config import UserConfig
from tabensemb.data import DataModule, AbstractSplitter
import numpy as np
from tabensemb.trainer import Trainer, load_trainer, save_trainer
from tabensemb.model import *
import torch
from copy import deepcopy as cp
import shutil
import pytest


def test_trainer():
    print(f"\n-- Loading trainer --\n")
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
                    "SampleWeightDeriver",
                    {
                        "stacked": True,
                        "intermediate": False,
                        "derived_name": "sample_weight",
                    },
                ),
            ],
        },
    )
    trainer.load_data()
    trainer.summarize_setting()

    print(f"\n-- Initialize models --\n")
    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        WideDeep(trainer, model_subset=["TabMlp"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(
            trainer,
            model_subset=[
                "Category Embedding",
            ],
        ),
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
        direct_pred = model._predict(x_test, derived_data=d_test, model_name=model_name)
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
    detached_pred = model_trainer.get_modelbase("CatEmbed_Category Embedding")._predict(
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
    model_trainer.datamodule.set_feature_names(
        model_trainer.datamodule.cont_feature_names[:5]
    )
    model_trainer.train()

    print(f"\n-- Bayes optimization --\n")
    model_trainer.args["bayes_opt"] = True
    model_trainer.get_leaderboard(cross_validation=2)

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

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))


def test_continue_previous():
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={
            "data_splitter": "RandomSplitter",
        },
    )
    trainer.load_data()

    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
    ]
    trainer.add_modelbases(models)
    l0 = trainer.get_leaderboard(cross_validation=1, split_type="random")
    l1 = trainer.get_leaderboard(cross_validation=2, load_from_previous=True)

    l2 = trainer.get_leaderboard(cross_validation=2, split_type="random")
    cols = ["Training RMSE", "Testing RMSE", "Validation RMSE"]
    assert np.allclose(
        l1[cols].values.astype(float), l2[cols].values.astype(float)
    ), f"load_from_previous does not get consistent results."

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))


def test_inspect():
    print(f"\n-- Loading trainer --\n")
    configfile = "sample"
    tabensemb.setting["debug_mode"] = True
    trainer = Trainer(device="cpu")
    trainer.load_config(
        configfile,
        manual_config={
            "data_splitter": "RandomSplitter",
        },
    )
    trainer.load_data()

    print(f"\n-- Initialize model --\n")
    model = CatEmbed(
        trainer,
        model_subset=[
            "Category Embedding",
        ],
    )
    trainer.add_modelbases([model])

    print(f"\n-- Train model --\n")
    trainer.train()

    print(f"\n-- Inspect model --\n")
    direct_inspect = model.inspect_attr("Category Embedding", ["hidden_representation"])
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
    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))
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
    with pytest.raises(Exception):
        trainer.load_data()

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))


def test_trainer_multitarget():
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
    trainer.load_data()
    trainer.summarize_setting()

    print(f"\n-- Initialize models --\n")

    with pytest.raises(Exception):
        WideDeep(trainer, model_subset=["TabMlp"])

    models = [
        PytorchTabular(trainer, model_subset=["Category Embedding"]),
        AutoGluon(trainer, model_subset=["Linear Regression"]),
        CatEmbed(
            trainer,
            model_subset=[
                "Category Embedding",
            ],
        ),
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
        direct_pred = model._predict(x_test, derived_data=d_test, model_name=model_name)
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
    detached_pred = model_trainer.get_modelbase("CatEmbed_Category Embedding")._predict(
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

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))


def test_permutation_importance():
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
    torchmodel_shap = trainer.cal_feature_importance(
        program="CatEmbed", model_name="Category Embedding", method="shap"
    )

    with pytest.raises(Exception):
        trainer.cal_feature_importance(
            program="CatEmbed",
            model_name="Require Model PyTabular CatEmbed",
            method="shap",
        )

    assert len(absmodel_perm[0]) == len(absmodel_perm[1])
    assert np.all(np.abs(absmodel_perm[0]) > 1e-8)

    assert len(absmodel_shap[0]) == len(absmodel_shap[1])
    assert np.all(np.abs(absmodel_shap[0]) > 1e-8)

    assert len(torchmodel_perm[0]) == len(torchmodel_perm[1])
    assert np.all(np.abs(torchmodel_perm[0][: len(trainer.all_feature_names)]) > 1e-8)
    # Unused data does not have feature importance.
    assert np.all(np.abs(torchmodel_perm[0][len(trainer.all_feature_names) :]) < 1e-8)

    assert len(torchmodel_shap[0]) == len(torchmodel_shap[1])
    # Categorical features does not have gradients, therefore does not have shap values using DeepExplainer.
    assert np.all(np.abs(torchmodel_shap[0][: len(trainer.cont_feature_names)]) > 1e-8)
    # Unused data does not have feature importance.
    assert np.all(np.abs(torchmodel_shap[0][len(trainer.cont_feature_names) :]) < 1e-8)

    shutil.rmtree(os.path.join(tabensemb.setting["default_output_path"]))
