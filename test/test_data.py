import unittest
import pytest
from import_utils import *
import tabensemb
from tabensemb.config import UserConfig
from tabensemb.data import DataModule, AbstractSplitter
from tabensemb.data.dataimputer import get_data_imputer
import numpy as np
import pandas as pd

relative_deriver_kwargs = {
    "stacked": True,
    "absolute_col": "cont_0",
    "relative2_col": "cont_1",
    "intermediate": False,
    "derived_name": "derived_cont",
}
sample_weight_deriver_kwargs = {
    "stacked": True,
    "intermediate": True,
    "derived_name": "sample_weight",
}


def test_datamodule():
    print("\n-- Loading config --\n")
    config = UserConfig("sample")
    config.merge(
        {
            "data_processors": [
                ["CategoricalOrdinalEncoder", {}],
                ["NaNFeatureRemover", {}],
                ["VarianceFeatureSelector", {"thres": 0.1}],
                ["FeatureValueSelector", {"feature": "cat_1", "value": 0}],
                ["CorrFeatureSelector", {"thres": 0.1}],
                ["IQRRemover", {}],
                ["StdRemover", {}],
                ["SampleDataAugmentor", {}],
                ["StandardScaler", {}],
            ],
        }
    )

    print("\n-- Loading datamodule --\n")
    datamodule = DataModule(config=config)

    print("\n-- Loading data --\n")
    np.random.seed(1)
    datamodule.load_data()

    print(f"\n-- Check splitting --\n")
    AbstractSplitter._check_split(
        datamodule.train_indices,
        datamodule.val_indices,
        datamodule.test_indices,
    )

    print(f"\n-- Check augmentation --\n")
    aug_desc = datamodule.df.loc[
        datamodule.augmented_indices - len(datamodule.dropped_indices),
        datamodule.all_feature_names + datamodule.label_name,
    ].describe()
    original_desc = datamodule.df.loc[
        datamodule.val_indices[-2:],
        datamodule.all_feature_names + datamodule.label_name,
    ].describe()
    assert np.allclose(
        aug_desc.values.astype(float), original_desc.values.astype(float)
    )

    print(f"\n-- Prepare new data when indices are randpermed --\n")
    df = datamodule.df.copy()
    indices = np.array(df.index)
    np.random.shuffle(indices)
    df.index = indices
    df, derived_data = datamodule.prepare_new_data(df)
    assert np.allclose(
        df[datamodule.all_feature_names + datamodule.label_name].values,
        datamodule.df[datamodule.all_feature_names + datamodule.label_name].values,
    ), "Stacked features from prepare_new_data for the set dataframe does not get consistent results"
    assert len(derived_data) == len(datamodule.derived_data), (
        "The number of unstacked features from " "prepare_new_data is not consistent"
    )
    for key, value in datamodule.derived_data.items():
        if key != "augmented":
            assert np.allclose(value, derived_data[key]), (
                f"Unstacked feature `{key}` from prepare_new_data for the set "
                "dataframe does not get consistent results"
            )

    print(f"\n-- Set feature names --\n")
    datamodule.set_feature_names(datamodule.cont_feature_names[:2])
    assert (
        len(datamodule.cont_feature_names) == 2
        and len(datamodule.cat_feature_names) == 0
        and len(datamodule.label_name) == 1
    ), "set_feature_names is not functional."

    print(f"\n-- Prepare new data after set feature names --\n")
    df, derived_data = datamodule.prepare_new_data(datamodule.df)
    assert (
        len(datamodule.cont_feature_names) == 2
        and len(datamodule.cat_feature_names) == 0
        and len(datamodule.label_name) == 1
    ), "set_feature_names is not functional when prepare_new_data."
    assert np.allclose(
        df[datamodule.all_feature_names + datamodule.label_name].values,
        datamodule.df[datamodule.all_feature_names + datamodule.label_name].values,
    ), (
        "Stacked features from prepare_new_data after set_feature_names for the set dataframe does not get "
        "consistent results"
    )
    assert len(derived_data) == len(datamodule.derived_data), (
        "The number of unstacked features after set_feature_names from "
        "prepare_new_data is not consistent"
    )
    for key, value in datamodule.derived_data.items():
        if key != "augmented":
            assert np.allclose(value, derived_data[key]), (
                f"Unstacked feature `{key}` after set_feature_names from prepare_new_data for the set "
                "dataframe does not get consistent results"
            )

    print(f"\n-- Describe --\n")
    datamodule.describe()
    datamodule.cal_corr()

    print(f"\n-- Get not imputed dataframe --\n")
    datamodule.get_not_imputed_df()

    print(f"\n-- Remove feature --\n")
    df = datamodule.df.copy()
    original_cont_names = datamodule.cont_feature_names.copy()
    df[datamodule.cont_feature_names[0]] = 1.0
    datamodule.set_data(
        df,
        cont_feature_names=datamodule.cont_feature_names,
        cat_feature_names=datamodule.cat_feature_names,
        label_name=datamodule.label_name,
    )
    new_cont_names = datamodule.cont_feature_names.copy()

    assert len(original_cont_names) == len(new_cont_names) + 1
    assert original_cont_names[0] not in new_cont_names


def test_rfe():
    config = UserConfig("sample")
    min_features_to_select = 3
    config.merge(
        {
            "data_processors": [
                ["CategoricalOrdinalEncoder", {}],
                ["NaNFeatureRemover", {}],
                ["VarianceFeatureSelector", {"thres": 1}],
                [
                    "RFEFeatureSelector",
                    {
                        "n_estimators": 2,
                        "min_features_to_select": min_features_to_select,
                        "verbose": 1,
                    },
                ],
                ["StandardScaler", {}],
            ],
        }
    )

    print("\n-- Loading datamodule --\n")
    datamodule = DataModule(config=config)

    print("\n-- Loading data --\n")
    np.random.seed(1)
    datamodule.load_data()

    assert len(datamodule.cont_feature_names) >= min_features_to_select


def test_illegal_cont_feature():
    config = UserConfig("sample")
    # "cat_1" is not object
    config.merge(
        {
            "feature_names_type": {"cont_0": 0, "cat_1": 1},
            "categorical_feature_names": [],
        }
    )
    datamodule = DataModule(config=config)
    datamodule.load_data()

    # "cat_0" is object and cannot be converted
    with pytest.raises(Exception):
        datamodule.set_data(
            datamodule.df,
            cont_feature_names=["cont_0", "cat_0"],
            cat_feature_names=[],
            label_name=datamodule.label_name,
        )

    df = datamodule.df.copy()
    df["cat_1"] = df["cat_1"].values.astype(object)
    datamodule.set_data(
        df,
        cont_feature_names=["cont_0", "cat_1"],
        cat_feature_names=[],
        label_name=datamodule.label_name,
    )


def test_data_deriver():
    print("\n-- Loading config --\n")
    config = UserConfig("sample")
    config.merge(
        {
            "data_processors": [
                ["CategoricalOrdinalEncoder", {}],
                ["NaNFeatureRemover", {}],
                ["VarianceFeatureSelector", {"thres": 1}],
                ["StandardScaler", {}],
            ],
        }
    )
    relative_deriver_unstacked_kwargs = relative_deriver_kwargs.copy()
    relative_deriver_unstacked_kwargs["stacked"] = False
    relative_deriver_unstacked_kwargs["derived_name"] = "derived_cont_unstacked"
    config.merge(
        {
            "data_derivers": [
                ("RelativeDeriver", relative_deriver_kwargs),
                ("RelativeDeriver", relative_deriver_unstacked_kwargs),
                ("SampleWeightDeriver", sample_weight_deriver_kwargs),
            ]
        }
    )
    print("\n-- Loading datamodule --\n")
    datamodule = DataModule(config=config)

    print("\n-- Loading data --\n")
    np.random.seed(1)
    datamodule.load_data()

    assert "derived_cont" in datamodule.cont_feature_names
    assert "derived_cont_unstacked" not in datamodule.cont_feature_names
    assert "derived_cont_unstacked" in datamodule.derived_data.keys()
    assert "sample_weight" in datamodule.df.columns
    assert "sample_weight" not in datamodule.cont_feature_names


def test_data_splitter():
    import numpy as np
    import pandas as pd
    from tabensemb.data.datasplitter import RandomSplitter

    df = pd.DataFrame({"test_feature": np.random.randint(0, 20, (100,))})
    print("\n-- k-fold RandomSplitter --\n")
    spl = RandomSplitter()
    res_random = [spl.split(df, [], [], [], cv=5) for i in range(5)]
    assert np.allclose(
        np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
    ), "RandomSplitter is not getting correct k-fold results."

    print("\n-- k-fold RandomSplitter in a new iteration --\n")
    res_random = [spl.split(df, [], [], [], cv=5) for i in range(5)]
    assert np.allclose(
        np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
    ), "RandomSplitter is not getting correct k-fold results in a new iteration."

    print("\n-- k-fold RandomSplitter change k --\n")
    spl.split(df, [], [], [], cv=5)
    res_random = [spl.split(df, [], [], [], cv=3) for i in range(3)]
    assert np.allclose(
        np.sort(np.hstack([i[2] for i in res_random])), np.arange(100)
    ), "RandomSplitter is not getting correct k-fold results after changing the number of k-fold."

    print("\n-- Non-cv RandomSplitter --\n")
    spl = RandomSplitter()
    res_random = [spl.split(df, [], [], []) for i in range(5)]
    res = np.sort(np.hstack([i[2] for i in res_random]))
    assert len(res) != 100 or (
        len(res) == 100 and not np.allclose(res, np.arange(100))
    ), "RandomSplitter is getting k-fold results without cv arguments."


def test_data_imputer():
    config = UserConfig("sample")
    config.merge({"data_derivers": [("RelativeDeriver", relative_deriver_kwargs)]})
    datamodule = DataModule(config=config)

    print("\n-- MeanImputer --\n")
    datamodule.set_data_imputer("MeanImputer")
    datamodule.load_data()
    assert not np.any(pd.isna(datamodule.df[datamodule.all_feature_names]).values)

    original = datamodule.get_not_imputed_df()

    print("\n-- MedianImputer --\n")
    imputer = get_data_imputer("MedianImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)

    print("\n-- ModeImputer --\n")
    imputer = get_data_imputer("ModeImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)

    print("\n-- MiceLightgbmImputer --\n")
    imputer = get_data_imputer("MiceLightgbmImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)

    print("\n-- MiceImputer --\n")
    imputer = get_data_imputer("MiceImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)

    print("\n-- MissForestImputer --\n")
    imputer = get_data_imputer("MissForestImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)

    print("\n-- GainImputer --\n")
    imputer = get_data_imputer("GainImputer")()
    imputed = imputer.fit_transform(original.copy(), datamodule)
    assert not np.any(pd.isna(imputed[datamodule.all_feature_names]).values)
