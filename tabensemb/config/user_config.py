from typing import Dict, Union
import json
import os.path
import importlib.machinery
import types
import tabensemb
from tabensemb.utils import pretty, str_to_dataframe
from .default import cfg as default_cfg
import argparse
import urllib.request
import ssl
import re
import zipfile
import numpy as np
import warnings


class UserConfig(dict):
    """
    The configuration holder for :class:`~tabensemb.data.datamodule.DataModule` and :class:`~tabensemb.trainer.Trainer`.
    """

    def __init__(self, path: str = None):
        """
        Parameters
        ----------
        path
            Path to the configuration file. See :meth:`from_file`.
        """
        super(UserConfig, self).__init__()
        self.update(default_cfg)
        self._defaults = default_cfg.copy()
        if path is not None:
            self.merge(self.from_file(path))

    def defaults(self):
        """
        The default values in ``tabensemb.config.default.py``

        Returns
        -------
        dict
            A dictionary of default values.
        """
        return self._defaults.copy()

    def merge(self, d: Dict):
        """
        Similar to :meth:`dict.update`, but will ignore values that are None.

        Parameters
        ----------
        d
            The dictionary used to update the configuration.
        """
        d_cp = d.copy()
        for key, val in d_cp.items():
            if val is None:
                d.__delitem__(key)
        super(UserConfig, self).update(d)

    @staticmethod
    def from_parser() -> Dict:
        """
        Try to parse the configuration using ``argparse`` and merge it into defaults.

        Returns
        -------
        dict
            The parsed configuration dictionary.
        """
        base_config = UserConfig()
        parser = argparse.ArgumentParser()
        parser.add_argument("--base", required=True)
        for key in base_config.keys():
            if type(base_config[key]) in [str, int, float]:
                parser.add_argument(
                    f"--{key}", type=type(base_config[key]), required=False
                )
            elif type(base_config[key]) == list:
                parser.add_argument(
                    f"--{key}",
                    nargs="+",
                    type=type(base_config[key][0])
                    if len(base_config[key]) > 0
                    else None,
                    required=False,
                )
            elif type(base_config[key]) == bool:
                parser.add_argument(f"--{key}", dest=key, action="store_true")
                parser.add_argument(f"--no-{key}", dest=key, action="store_false")
                parser.set_defaults(**{key: base_config[key]})
        parse_res = parser.parse_known_args()[0].__dict__
        return parse_res

    @staticmethod
    def from_dict(cfg: Dict) -> "UserConfig":
        """
        Merge the input dictionary into defaults.

        Parameters
        ----------
        cfg
            The dictionary used to update the default configuration.

        Returns
        -------
        UserConfig
            The combined configuration.
        """
        tmp_cfg = UserConfig()
        tmp_cfg.merge(cfg)
        return tmp_cfg

    @staticmethod
    def from_file(path: str) -> "UserConfig":
        """
        Merge the .py or .json file into defaults. If no suffix is given, it will search the current directory and
        ``tabensemb.setting["default_config_path"]`` for a matched file. In a legal .py file, there should be a
        dictionary named "cfg".

        Parameters
        ----------
        path
            The path to the configuration file to update the default configuration with or without a suffix
            (.py or .json).

        Returns
        -------
        UserConfig
            The combined configuration.
        """
        file_path = (
            path
            if "/" in path or os.path.isfile(path)
            else os.path.join(tabensemb.setting["default_config_path"], path)
        )
        ty = UserConfig.file_type(file_path)
        if ty is None:
            json_path = file_path + ".json"
            py_path = file_path + ".py"
            is_json = os.path.isfile(json_path)
            is_py = os.path.isfile(py_path)
            if is_json and is_py:
                raise Exception(
                    f"Both {json_path} and {py_path} exist. Specify the full name of the file."
                )
            elif not is_json and not is_py:
                raise Exception(f"{file_path} does not exist.")
            else:
                file_path = json_path if is_json else py_path
                ty = UserConfig.file_type(file_path)
        else:
            if not os.path.isfile(file_path):
                raise Exception(f"{file_path} does not exist.")

        if ty == "json":
            with open(file_path, "r") as file:
                cfg = json.load(file)
        else:
            loader = importlib.machinery.SourceFileLoader("cfg", file_path)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            cfg = mod.cfg
        return UserConfig.from_dict(cfg)

    @staticmethod
    def from_uci(
        name: str,
        datafile_name: str = None,
        save_zip: bool = False,
        max_retries=3,
        timeout=5,
        sep=",",
    ) -> Union["UserConfig", None]:
        """
        Search, download, and configure a dataset from https://archive.ics.uci.edu/. The dataset will be extracted and
        saved into a .csv file, and a corresponding UserConfig is returned. This function supports tabular datasets for
        "Classification" and "Regression". Integer features are treated as continuous features.

        Parameters
        ----------
        name
            The name of the dataset like "Heart Disease", "Iris", etc. The name will be searched on the website and be
            configured if there is a matched dataset.
        datafile_name
            The name of ".data" file in the downloaded .zip file. If is None and there exists more than one file with
            the suffix ".data" in a single dataset, the function will print available names.
        save_zip
            Whether the downloaded .zip file should be stored.
        max_retries
            The maximum number of tries of ``urllib.request.urlopen``.
        timeout
            Waiting time of ``urllib.request.urlopen``.
        sep
            The delimiter of ``pd.read_csv``.

        Returns
        -------
        UserConfig
            The configuration of the dataset. If the dataset can not be automatically configured, None will be returned
            and the reason will be printed.
        """
        # Extract information
        url = (
            f"https://archive.ics.uci.edu/datasets?skip=0&take=1&sort=desc&orderBy=Relevance&"
            f"search={name.replace(' ', '+')}"
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        reties = 0
        while True:
            try:
                uh = urllib.request.urlopen(url, context=ctx, timeout=timeout)
                break
            except:
                reties += 1
                if reties == max_retries:
                    raise Exception(
                        f"max_retries reached. Check whether {url} is accessible."
                    )
        html = uh.read()
        datasets: Dict = json.loads(
            re.findall(r"\"body\":\"\[(.*?)\]\"\}</script>", html.decode())[0]
            .encode()
            .decode("unicode-escape")
        )["result"]["data"]["json"]["datasets"]
        if len(datasets) == 0:
            raise Exception(f"Dataset {name} not found.")
        dataset = datasets[0]
        if dataset["Name"] != name:
            raise Exception(f"Dataset {name} not found. Do you mean {dataset['Name']}?")

        # Download the dataset
        id = dataset["ID"]
        slug = dataset["slug"]
        link = f"https://archive.ics.uci.edu/static/public/{id}/{slug}.zip"
        zip_save_to = os.path.join(
            tabensemb.setting["default_data_path"], f"{name}.zip"
        )
        print(f"Downloading {link} to {zip_save_to}")
        os.makedirs(tabensemb.setting["default_data_path"], exist_ok=True)
        urllib.request.urlretrieve(link, filename=zip_save_to)

        # Check task and tabular
        _saved_to_suffix = f" The downloaded file is saved to {zip_save_to}."
        task = dataset["Task"]
        is_tabular = dataset["isTabular"]
        if task not in ["Regression", "Classification"]:
            print(f"Task {task} is not supported.{_saved_to_suffix}")
            return None
        if not is_tabular:
            print(f"The dataset {name} is not tabular.{_saved_to_suffix}")
            return None

        # Check contents
        zipf = zipfile.ZipFile(zip_save_to, "r")
        files = zipf.namelist()
        datafiles = [name.split(".data")[0] for name in files if name.endswith(".data")]
        if len(datafiles) == 0:
            print(f"No file with suffix `.data` is found.{_saved_to_suffix}")
            return None
        if len(datafiles) > 1 and (
            datafile_name is None or datafile_name not in datafiles
        ):
            print(
                f"Found multiple data files {datafiles}, but `datafile_name` is {datafile_name}.{_saved_to_suffix}"
            )
            return None
        test_datafiles = [name for name in files if "test" in name]
        if len(test_datafiles) > 0:
            warnings.warn(
                f"There exists .test file(s) {test_datafiles} which should be used for final metrics. The .zip file is "
                f"left for the user to process."
            )
            save_zip = True
        if datafile_name is None:
            datafile_name = datafiles[0]

        # Extract feature information.
        all_features = []
        cont_feature_names = []
        cat_feature_names = []
        label_name = []
        for attr in dataset["attributes"]:
            if attr["role"] == "Feature":
                if attr["type"] == "Continuous":
                    cont_feature_names.append(attr["name"])
                if attr["type"] == "Integer":
                    cont_feature_names.append(attr["name"])
                    print(
                        f"{attr['name']} is Integer and will be treated as a continuous feature."
                    )
                elif attr["type"] in ["Categorical", "Binary"]:
                    cat_feature_names.append(attr["name"])
            elif attr["role"] == "Target":
                label_name.append(attr["name"])
            all_features.append(attr["name"])

        # Load and save as .csv
        datafile = zipf.read(datafile_name + ".data")
        try:
            df = str_to_dataframe(
                datafile.decode(),
                sep=sep,
                names=all_features,
                check_nan_on=cont_feature_names,
            )
        except Exception as e:
            print(e)
            print(_saved_to_suffix)
            return None

        # Save csv
        csv_name = name if datafile_name is None else datafile_name
        df.to_csv(
            os.path.join(tabensemb.setting["default_data_path"], f"{csv_name}.csv"),
            index=False,
        )
        zipf.close()
        if not save_zip:
            os.remove(zip_save_to)

        # Configurations
        if task == "Regression":
            inferred_task = "regression"
        else:
            if len(np.unique(df[label_name].values)) <= 2:
                inferred_task = "binary"
            else:
                inferred_task = "multiclass"
        feature_names_type = {
            name: 0 if name in cont_feature_names else 1
            for name in cont_feature_names + cat_feature_names
        }
        cfg = UserConfig()
        cfg.merge(
            {
                "database": csv_name,
                "task": inferred_task,
                "feature_types": ["Continuous", "Categorical", "Derived"],
                "feature_names_type": feature_names_type,
                "categorical_feature_names": cat_feature_names,
                "label_name": label_name,
            }
        )
        return cfg

    def to_file(self, path: str):
        """
        Save the configuration to a ``.py`` or ``.json`` file.

        Parameters
        ----------
        path
            The path to save the configuration. If no suffix is given, ``.py`` is added as the suffix.
        """
        if path.endswith(".json"):
            with open(os.path.join(path), "w") as f:
                json.dump(self, f, indent=4)
        else:
            if not path.endswith(".py"):
                path += ".py"
            s = "cfg = " + pretty(self, htchar=" " * 4, indent=0)
            try:
                import black

                s = black.format_str(s, mode=black.Mode())
            except:
                pass
            with open(path, "w") as f:
                f.write(s)

    @staticmethod
    def file_type(path: str) -> Union[str, None]:
        """
        Check the suffix of the path (json, py, or None).
        """
        if path.endswith(".json"):
            return "json"
        elif path.endswith(".py"):
            return "py"
        else:
            return None
