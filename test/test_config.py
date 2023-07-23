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
