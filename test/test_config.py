import shutil
from import_utils import *
import tabensemb
from tabensemb.config import UserConfig
import pytest
import collections.abc


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

    with pytest.raises(Exception):
        _ = UserConfig("./sample")

    config_json = UserConfig("./sample.json")

    for key in config.keys():
        if not isinstance(config[key], collections.abc.Iterable):
            assert config[key] == config_json[key]

    os.remove("./sample.py")
    os.remove("./sample.json")


def test_exceptions():
    print("\n--- not exist ---\n")
    with pytest.raises(Exception):
        _ = UserConfig("NOT_EXIST")
    with pytest.raises(Exception):
        _ = UserConfig("NOT_EXIST.py")
    with pytest.raises(Exception):
        _ = UserConfig("NOT_EXIST.json")


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


def test_from_uci():
    _default_data_path = tabensemb.setting["default_data_path"]
    os.makedirs("temp_data", exist_ok=True)
    tabensemb.setting["default_data_path"] = "temp_data"
    cfg_iris = UserConfig.from_uci("Iris", datafile_name="iris", max_retries=10)
    assert cfg_iris is not None
    assert cfg_iris["task"] == "multiclass"
    cfg_autompg = UserConfig.from_uci("Auto MPG", sep="\s+", max_retries=10)
    assert cfg_autompg is not None
    assert cfg_autompg["task"] == "regression"
    with pytest.warns(UserWarning):
        # Exists a test file.
        cfg_adult = UserConfig.from_uci("Adult", sep=", ", max_retries=10)
    assert cfg_adult is not None
    assert cfg_adult["task"] == "binary"

    # sep
    assert UserConfig.from_uci("Auto MPG", max_retries=10) is None
    # Not supported
    assert UserConfig.from_uci("Wine Quality", max_retries=10) is None
    # Found multiple data files
    assert UserConfig.from_uci("Iris", max_retries=10) is None
    # Not tabular
    assert UserConfig.from_uci("Kinship", max_retries=10) is None
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
    tabensemb.setting["default_data_path"] = _default_data_path
