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
