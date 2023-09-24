import shutil
from import_utils import *
import tabensemb
from tabensemb.config import UserConfig
import pytest
import collections.abc
import pandas as pd


def test_config_basic():
    config = UserConfig()
    config.merge(UserConfig.from_file("sample"))
    config = UserConfig("sample")
    new_config = dict(epoch=100, bayes_epoch=None)
    config.merge(new_config)
    _ = config.defaults()

    assert config["bayes_epoch"] is not None
    assert config["epoch"] == 100


def test_config_json():
    config = UserConfig()
    config.to_file("sample.json")
    config.to_file("sample")

    with pytest.raises(Exception) as err:
        _ = UserConfig("./sample")
    assert "Both" in err.value.args[0]

    config_json = UserConfig("./sample.json")

    for key in config.keys():
        if not isinstance(config[key], collections.abc.Iterable):
            assert config[key] == config_json[key]

    os.remove("./sample.py")
    os.remove("./sample.json")


def test_exceptions():
    print("\n--- not exist ---\n")
    with pytest.raises(Exception) as err:
        _ = UserConfig("NOT_EXIST")
    assert "does not exist." in err.value.args[0]
    with pytest.raises(Exception) as err:
        _ = UserConfig("NOT_EXIST.py")
    assert "does not exist." in err.value.args[0]
    with pytest.raises(Exception) as err:
        _ = UserConfig("NOT_EXIST.json")
    assert "does not exist." in err.value.args[0]


def test_cmd_arguments(mocker):
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
    cfg = UserConfig.from_parser()
    assert cfg["epoch"] == 2
    assert cfg["bayes_opt"]
    assert cfg["data_imputer"] == "GainImputer"
    assert all([x == y for x, y in zip(cfg["split_ratio"], [0.3, 0.1, 0.6])])


def test_getitem():
    config = UserConfig()
    config.merge(
        {
            "continuous_feature_names": ["test_cont"],
            "categorical_feature_names": ["test_cat"],
        }
    )
    types = config["feature_types"]
    assert types["test_cont"] == "Continuous" and types["test_cat"] == "Categorical"
    unique = config["unique_feature_types"]
    assert unique[0] == "Categorical" and unique[1] == "Continuous"

    config.merge({"feature_types": {"test_cont": "TEST"}})
    types = config["feature_types"]
    assert types["test_cont"] == "TEST" and types["test_cat"] == "Categorical"
    unique = config["unique_feature_types"]
    assert unique[0] == "Categorical" and unique[1] == "TEST"

    config.merge({"feature_types": {"test_cont": "TEST_CONT", "test_cat": "TEST_CAT"}})
    types = config["feature_types"]
    assert types["test_cont"] == "TEST_CONT" and types["test_cat"] == "TEST_CAT"
    unique = config["unique_feature_types"]
    assert unique[0] == "TEST_CAT" and unique[1] == "TEST_CONT"


def test_from_uci():
    os.makedirs("temp_data", exist_ok=True)
    with tabensemb.utils.global_setting({"default_data_path": "temp_data"}):
        with pytest.warns(UserWarning, match=r"is not given"):
            cfg_iris = UserConfig.from_uci(
                "Iris",
                datafile_name="iris",
                max_retries=10,
            )
        df = pd.read_csv(
            os.path.join(
                tabensemb.setting["default_data_path"], f"{cfg_iris['database']}.csv"
            )
        )
        assert all([x == y for x, y in zip(df.columns, iris_columns)])
        assert cfg_iris is not None
        assert cfg_iris["task"] == "multiclass"

        with pytest.warns(UserWarning, match=r"is not given"):
            cfg_autompg = UserConfig.from_uci("Auto MPG", sep=r"\s+", max_retries=10)
        df = pd.read_csv(
            os.path.join(
                tabensemb.setting["default_data_path"], f"{cfg_autompg['database']}.csv"
            )
        )
        assert all(
            [
                x == y
                for x, y in zip(df.columns[:3], ["displacement", "mpg", "cylinders"])
            ]
        )
        assert all([x == y for x, y in zip(df.columns[3:], mpg_columns[3:])])
        assert cfg_autompg is not None
        assert cfg_autompg["task"] == "regression"

        with pytest.warns(UserWarning) as warninfo:
            # Exists a test file.
            cfg_adult = UserConfig.from_uci("Adult", sep=", ", max_retries=10)
        warns = [warn.message.args[0] for warn in warninfo]
        expected = ["There exists", "is not given"]
        assert all([x in y for x, y in zip(expected, warns)])
        assert cfg_adult is not None
        assert cfg_adult["task"] == "binary"
        df = pd.read_csv(
            os.path.join(
                tabensemb.setting["default_data_path"], f"{cfg_adult['database']}.csv"
            )
        )
        assert all([x == y for x, y in zip(df.columns, adult_columns)])

        with pytest.raises(Exception) as err:
            _ = UserConfig.from_uci(
                "Iris",
                column_names=iris_columns + ["TEST"],
                datafile_name="iris",
                max_retries=10,
            )
        assert "Available column names are" in err.value.args[0]
        with pytest.raises(Exception) as err:
            with pytest.warns(UserWarning, match=r"Available column names are"):
                _ = UserConfig.from_uci(
                    "Iris",
                    column_names=iris_columns[:-1],
                    datafile_name="iris",
                    max_retries=10,
                )
        assert "No label is found." in err.value.args[0]

        # sep
        assert (
            UserConfig.from_uci("Auto MPG", column_names=mpg_columns, max_retries=10)
            is None
        )
        # Task Not supported
        assert UserConfig.from_uci("Wine Quality", max_retries=10) is None
        # Found multiple data files
        assert UserConfig.from_uci("Iris", max_retries=10) is None
        # Task not supported
        assert UserConfig.from_uci("Kinship", max_retries=10) is None
        # Not Tabular
        assert UserConfig.from_uci("CMU Face Images", max_retries=10) is None
        # No file with suffix `.data`
        assert UserConfig.from_uci("Sundanese Twitter Dataset", max_retries=10) is None

        with pytest.raises(Exception) as err:
            UserConfig.from_uci("TESTTEST", max_retries=10)
        assert "not found" in err.value.args[0]

        with pytest.raises(Exception) as err:
            UserConfig.from_uci("TESTTEST", timeout=1e-9)
        assert "max_retries reached" in err.value.args[0]

        with pytest.raises(Exception) as err:
            UserConfig.from_uci("Iriss", max_retries=10)
        assert "Do you mean" in err.value.args[0]

        shutil.rmtree("temp_data")
