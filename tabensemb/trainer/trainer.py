import os.path
import matplotlib.figure
import matplotlib.axes
import numpy as np
import tabensemb
from tabensemb.utils import *
from tabensemb.config import UserConfig
from tabensemb.data import DataModule
from copy import deepcopy as cp
from skopt.space import Real, Integer, Categorical
import time
from typing import *
import torch.nn as nn
import torch.cuda
import torch.utils.data as Data
import scipy.stats as st
from sklearn.utils import resample as skresample
import platform, psutil, subprocess
import shutil
import pickle

set_random_seed(tabensemb.setting["random_seed"])


class Trainer:
    """
    The model manager that provides saving, loading, ranking, and analyzing utilities.

    Attributes
    ----------
    args
        A :class:`tabensemb.config.UserConfig` instance.
    configfile
        The source of the configuration. If the ``config`` argument of :meth:`load_config` is a
        :class:`tabensemb.config.UserConfig`, it is "UserInputConfig". If the ``config`` argument is a path, it is the
        path. If the ``config`` argument is not given, it is the "base" argument passed to python when executing the
        script.
    datamodule
        A :class:`tabensemb.data.datamodule.DataModule` instance.
    device
        The device on which models are trained. "cpu", "cuda", or "cuda:X".
    leaderboard
        The ranking of all models in all model bases. Only valid after :meth:`get_leaderboard` is called.
    modelbases
        A list of :class:`tabensemb.model.AbstractModel`.
    modelbases_names
        Corresponding names (:attr:`tabensemb.model.AbstractModel.program`) of :attr:`modelbases`.
    project
        The name of the :class:`Trainer`.
    project_root
        The place where all files are stored.
        ``tabensemb.setting["default_output_path"]`` ``/{project}/{project_root_subfolder}/{TIME}-{config}`` where ``project`` is :attr:`project`,
        ``project_root_subfolder`` and ``config`` are arguments of :meth:`load_config`.
    sys_summary
        Summary of the system when :meth:`summarize_device` is called.
    SPACE
    all_feature_names
    cat_feature_mapping
    cat_feature_names
    chosen_params
    cont_feature_names
    derived_data
    derived_stacked_features
    df
    feature_data
    label_data
    label_name
    static_params
    tensors
    test_indices
    train_indices
    training
    unscaled_feature_data
    unscaled_label_data
    val_indices
    """

    def __init__(self, device: str = "cpu", project: str = None):
        """
        The bridge of all modules. It contains all configurations and data. It can train model bases and evaluate
        results (including feature importance, partial dependency, etc.).

        Parameters
        ----------
        device:
            The device on which models are trained. Choose from "cpu", "cuda", or "cuda:X" (if available).
        project:
            The name of the :class:`Trainer`.
        """
        self.device = "cpu"
        self.project = project
        self.modelbases = []
        self.modelbases_names = []
        self.set_device(device)

    def set_device(self, device: str):
        """
        Set the device on which models are trained.

        Parameters
        ----------
        device
            "cpu", "cuda", or "cuda:X" (if available)

        Notes
        -----
        Multi-GPU training and training on a machine with multiple GPUs are not tested.
        """
        if device not in ["cpu", "cuda"] and "cuda" not in device:
            raise Exception(
                f"Device {device} is an invalid selection. Choose among {['cpu', 'cuda']}."
                f"Note: Multi-GPU training and training on a machine with multiple GPUs are not tested."
            )
        self.device = device

    def add_modelbases(self, models: List):
        """
        Add a list of model bases and check whether their names conflict.

        Parameters
        ----------
        models:
            A list of :class:`tabensemb.model.AbstractModel`.
        """
        new_modelbases_names = self.modelbases_names + [x.program for x in models]
        if len(new_modelbases_names) != len(list(set(new_modelbases_names))):
            raise Exception(f"Conflicted model base names: {self.modelbases_names}")
        self.modelbases += models
        self.modelbases_names = new_modelbases_names

    def get_modelbase(self, program: str):
        """
        Get the selected model base by its name.

        Parameters
        ----------
        program
            The name of the model base.

        Returns
        -------
        AbstractModel
            A model base.
        """
        if program not in self.modelbases_names:
            raise Exception(f"Model base {program} not added to the trainer.")
        return self.modelbases[self.modelbases_names.index(program)]

    def clear_modelbase(self):
        """
        Delete all model bases in the :class:`Trainer`.
        """
        self.modelbases = []
        self.modelbases_names = []

    def detach_modelbase(self, program: str, verbose: bool = True) -> "Trainer":
        """
        Detach the selected model base to a separate :class:`Trainer` and save it to another directory. It is much cheaper than
        :meth:`copy` if only one model base is needed.

        Parameters
        ----------
        program
            The selected model base.
        verbose
            Verbosity

        Returns
        -------
        Trainer
            A :class:`Trainer` with the selected model base.

        See Also
        --------
        :meth:`copy`, :meth:`detach_model`, :meth:`tabensemb.model.AbstractModel.detach_model`
        """
        modelbase = cp(self.get_modelbase(program=program))
        tmp_trainer = modelbase.trainer
        tmp_trainer.clear_modelbase()
        new_path = add_postfix(self.project_root)
        tmp_trainer.set_path(new_path, verbose=False)
        modelbase.set_path(os.path.join(new_path, modelbase.program))
        tmp_trainer.add_modelbases([modelbase])
        shutil.copytree(self.get_modelbase(program=program).root, modelbase.root)
        save_trainer(tmp_trainer, verbose=verbose)
        return tmp_trainer

    def detach_model(
        self, program: str, model_name: str, verbose: bool = True
    ) -> "Trainer":
        """
        Detach the selected model of the selected model base to a separate :class:`Trainer` and save it to another directory.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model.
        verbose
            Verbosity.

        Returns
        -------
        Trainer
            A :class:`Trainer` with the selected model in its model base.
        """
        tmp_trainer = self.detach_modelbase(program=program, verbose=False)
        tmp_modelbase = tmp_trainer.get_modelbase(program=program)
        detached_model = tmp_modelbase.detach_model(
            model_name=model_name, program=f"{program}_{model_name}"
        )
        tmp_trainer.clear_modelbase()
        tmp_trainer.add_modelbases([detached_model])
        shutil.rmtree(tmp_modelbase.root)
        save_trainer(tmp_trainer, verbose=verbose)
        return tmp_trainer

    def copy(self) -> "Trainer":
        """
        Copy the :class:`Trainer` and save it to another directory. It might be time and space-consuming because all
        model bases are copied once.

        Returns
        -------
        trainer
            A :class:`Trainer` instance.

        See Also
        --------
        :meth:`detach_modelbase`, :meth:`detach_model`, :meth:`tabensemb.model.AbstractModel.detach_model`
        """
        tmp_trainer = cp(self)
        new_path = add_postfix(self.project_root)
        tmp_trainer.set_path(new_path, verbose=True)
        for modelbase in tmp_trainer.modelbases:
            modelbase.set_path(os.path.join(new_path, modelbase.program))
        shutil.copytree(self.project_root, tmp_trainer.project_root, dirs_exist_ok=True)
        save_trainer(tmp_trainer)
        return tmp_trainer

    def load_config(
        self,
        config: Union[str, UserConfig] = None,
        manual_config: Dict = None,
        project_root_subfolder: str = None,
    ) -> None:
        """
        Load the configuration using a :class:`tabensemb.config.UserConfig` or a file in .py or .json format.
        Arguments passed to python when executing the script are parsed using ``argparse`` if ``config`` is
        left None. All keys in :meth:`tabensemb.config.UserConfig.defaults` can be parsed, for example:
        For the loss function: ``--loss mse``,
        For the total epoch: ``--epoch 200``,
        For the option of bayes opt: ``--bayes_opt`` to turn on Bayesian hyperparameter optimization,
        ``--no-bayes_opt`` to turn it off.
        The loaded configuration will be saved as a .py file in the project folder.

        Parameters
        ----------
        config
            It can be the path to the configuration file in json or python format, or a
            :class:`tabensemb.config.UserConfig` instance. If it is None, arguments passed to python will be parsed.
            If it is a path, it will be passed to :meth:`tabensemb.config.UserConfig.from_file`.
        manual_config
            Update the configuration with a dict. For example: ``manual_config={"bayes_opt": True}``.
        project_root_subfolder
            The subfolder that the project will be locate in. The folder name will be
            ``tabensemb.setting["default_output_path"]`` ``/{project}/{project_root_subfolder}/{TIME}-{config}``
        """
        input_config = config is not None
        if isinstance(config, str) or not input_config:
            # The base config is loaded using the --base argument
            if is_notebook() and not input_config:
                raise Exception(
                    "A config file must be assigned in notebook environment."
                )
            elif is_notebook() or input_config:
                parse_res = {"base": config}
            else:  # not notebook and config is None
                parse_res = UserConfig.from_parser()
            self.configfile = parse_res["base"]
            config = UserConfig(path=self.configfile)
            # Then, several args can be modified using other arguments like --lr, --weight_decay
            # only when a config file is not given so that configs depend on input arguments.
            if not is_notebook() and not input_config:
                config.merge(parse_res)
            if manual_config is not None:
                config.merge(manual_config)
            self.args = config
        else:
            self.configfile = "UserInputConfig"
            if manual_config is not None:
                warnings.warn(f"manual_config is ignored when config is an UserConfig.")
            self.args = config

        self.datamodule = DataModule(self.args)

        self.project = self.args["database"] if self.project is None else self.project
        self._create_dir(project_root_subfolder=project_root_subfolder)
        config.to_file(os.path.join(self.project_root, "args.py"))

    @property
    def static_params(self) -> Dict:
        """
        The "patience" and "epoch" parameters in the configuration.
        """
        return {
            "patience": self.args["patience"],
            "epoch": self.args["epoch"],
        }

    @property
    def chosen_params(self):
        """
        The "lr", "weight_decay", and "batch_size" parameters in the configuration.
        """
        return {
            "lr": self.args["lr"],
            "weight_decay": self.args["weight_decay"],
            "batch_size": self.args["batch_size"],
        }

    @property
    def SPACE(self):
        """
        Search spaces for "lr", "weight_decay", and "batch_size" defined in the configuration.
        """
        SPACE = []
        for var in self.args["SPACEs"].keys():
            setting = cp(self.args["SPACEs"][var])
            ty = setting["type"]
            setting.pop("type")
            if ty == "Real":
                SPACE.append(Real(name=var, **setting))
            elif ty == "Categorical":
                SPACE.append(Categorical(name=var, **setting))
            elif ty == "Integer":
                SPACE.append(Integer(name=var, **setting))
            else:
                raise Exception("Invalid type of skopt space.")
        return SPACE

    @property
    def feature_data(self) -> pd.DataFrame:
        """
        :meth:`tabensemb.data.datamodule.DataModule.feature_data`
        """
        return self.datamodule.feature_data if hasattr(self, "datamodule") else None

    @property
    def unscaled_feature_data(self):
        """
        :meth:`tabensemb.data.datamodule.DataModule.unscaled_feature_data`
        """
        return (
            self.datamodule.unscaled_feature_data
            if hasattr(self, "datamodule")
            else None
        )

    @property
    def unscaled_label_data(self):
        """
        :meth:`tabensemb.data.datamodule.DataModule.unscaled_label_data`
        """
        return (
            self.datamodule.unscaled_label_data if hasattr(self, "datamodule") else None
        )

    @property
    def label_data(self) -> pd.DataFrame:
        """
        :meth:`tabensemb.data.datamodule.DataModule.label_data`
        """
        return self.datamodule.label_data if hasattr(self, "datamodule") else None

    @property
    def derived_data(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.derived_data`
        """
        return self.datamodule.derived_data if hasattr(self, "datamodule") else None

    @property
    def cont_feature_names(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.cont_feature_names`
        """
        return (
            self.datamodule.cont_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def cat_feature_names(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.cat_feature_names`
        """
        return (
            self.datamodule.cat_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def all_feature_names(self):
        """
        :meth:`tabensemb.data.datamodule.DataModule.all_feature_names`
        """
        return (
            self.datamodule.all_feature_names if hasattr(self, "datamodule") else None
        )

    @property
    def label_name(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.label_name`
        """
        return self.datamodule.label_name if hasattr(self, "datamodule") else None

    @property
    def train_indices(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.train_indices`
        """
        return self.datamodule.train_indices if hasattr(self, "datamodule") else None

    @property
    def val_indices(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.val_indices`
        """
        return self.datamodule.val_indices if hasattr(self, "datamodule") else None

    @property
    def test_indices(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.test_indices`
        """
        return self.datamodule.test_indices if hasattr(self, "datamodule") else None

    @property
    def df(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.df`
        """
        return self.datamodule.df if hasattr(self, "datamodule") else None

    @property
    def tensors(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.tensors`
        """
        return self.datamodule.tensors if hasattr(self, "datamodule") else None

    @property
    def cat_feature_mapping(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.cat_feature_mapping`
        """
        return (
            self.datamodule.cat_feature_mapping if hasattr(self, "datamodule") else None
        )

    @property
    def derived_stacked_features(self):
        """
        :meth:`tabensemb.data.datamodule.DataModule.derived_stacked_features`
        """
        return (
            self.datamodule.derived_stacked_features
            if hasattr(self, "datamodule")
            else None
        )

    @property
    def training(self):
        """
        :attr:`tabensemb.data.datamodule.DataModule.training`
        """
        return self.datamodule.training if hasattr(self, "datamodule") else None

    def set_status(self, training: bool):
        """
        A wrapper of :meth:`tabensemb.data.datamodule.DataModule.set_status`
        """
        self.datamodule.set_status(training)

    def load_data(self, *args, **kwargs):
        """
        A wrapper of :meth:`tabensemb.data.datamodule.DataModule.load_data`. The ``save_path`` argument is set to
        :attr:`project_root`.
        """
        if "save_path" in kwargs.keys():
            kwargs.__delitem__("save_path")
        self.datamodule.load_data(save_path=self.project_root, *args, **kwargs)

    def set_path(self, path: Union[os.PathLike, str], verbose=False):
        """
        Set the work directory of the :class:`Trainer`.

        Parameters
        ----------
        path
            The work directory.
        """
        self.project_root = path
        if not os.path.exists(self.project_root):
            os.mkdir(self.project_root)
        if verbose:
            print(f"The project will be saved to {self.project_root}")

    def _create_dir(self, verbose: bool = True, project_root_subfolder: str = None):
        """
        Create the folder for the :class:`Trainer`.

        Parameters
        ----------
        verbose
            Whether to print the path of the :class:`Trainer`.
        project_root_subfolder
            See :meth:`load_config`.
        """
        default_path = tabensemb.setting["default_output_path"]
        if not os.path.exists(default_path):
            os.makedirs(default_path, exist_ok=True)
        if project_root_subfolder is not None:
            if not os.path.exists(os.path.join(default_path, project_root_subfolder)):
                os.makedirs(
                    os.path.join(default_path, project_root_subfolder), exist_ok=True
                )
        subfolder = (
            self.project
            if project_root_subfolder is None
            else os.path.join(project_root_subfolder, self.project)
        )
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        folder_name = t + "-0" + "_" + os.path.split(self.configfile)[-1]
        if not os.path.exists(os.path.join(default_path, subfolder)):
            os.makedirs(os.path.join(default_path, subfolder), exist_ok=True)
        self.set_path(
            add_postfix(os.path.join(default_path, subfolder, folder_name)),
            verbose=verbose,
        )

    def summarize_setting(self):
        """
        Print the summary of the device, the configuration, and the global setting of the package
        (``tabensemb.setting``).
        """
        print("Device:")
        print(pretty(self.summarize_device()))
        print("Configurations:")
        print(pretty(self.args))
        print(f"Global settings:")
        print(pretty(tabensemb.setting))

    def summarize_device(self):
        """
        Print a summary of the environment.
        https://www.thepythoncode.com/article/get-hardware-system-information-python
        """

        def get_size(bytes, suffix="B"):
            """
            Scale bytes to its proper format
            e.g:
                1253656 => '1.20MB'
                1253656678 => '1.17GB'
            """
            factor = 1024
            for unit in ["", "K", "M", "G", "T", "P"]:
                if bytes < factor:
                    return f"{bytes:.2f}{unit}{suffix}"
                bytes /= factor

        def get_processor_info():
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":
                return (
                    subprocess.check_output(
                        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
                    )
                    .strip()
                    .decode("utf-8")
                )
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo"
                all_info = (
                    subprocess.check_output(command, shell=True).strip().decode("utf-8")
                )

                for string in all_info.split("\n"):
                    if "model name\t: " in string:
                        return string.split("\t: ")[1]
            return ""

        uname = platform.uname()
        cpufreq = psutil.cpu_freq()
        svmem = psutil.virtual_memory()
        self.sys_summary = {
            "System": uname.system,
            "Node name": uname.node,
            "System release": uname.release,
            "System version": uname.version,
            "Machine architecture": uname.machine,
            "Processor architecture": uname.processor,
            "Processor model": get_processor_info(),
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "Max core frequency": f"{cpufreq.max:.2f}Mhz",
            "Total memory": get_size(svmem.total),
            "Python version": platform.python_version(),
            "Python implementation": platform.python_implementation(),
            "Python compiler": platform.python_compiler(),
            "Cuda availability": torch.cuda.is_available(),
            "GPU devices": [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ],
        }
        return self.sys_summary

    def train(
        self,
        programs: List[str] = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        """
        Train all model bases (:attr:`modelbases`).

        Parameters
        ----------
        programs
            A selected subset of model bases.
        verbose
            Verbosity.
        *args
            Arguments passed to :meth:`tabensemb.model.AbstractModel.train`
        **kwargs
            Arguments passed to :meth:`tabensemb.model.AbstractModel.train`
        """
        if programs is None:
            modelbases_to_train = self.modelbases
        else:
            modelbases_to_train = [self.get_modelbase(x) for x in programs]

        if len(modelbases_to_train) == 0:
            warnings.warn(
                f"No modelbase is trained. Please confirm that trainer.add_modelbases is called."
            )

        for modelbase in modelbases_to_train:
            modelbase.train(*args, verbose=verbose, **kwargs)

    def cross_validation(
        self,
        programs: List[str],
        n_random: int,
        verbose: bool,
        test_data_only: bool,
        split_type: str = "cv",
        load_from_previous: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        Repeat :meth:`load_data`, train model bases, and evaluate all models for multiple times.

        Parameters
        ----------
        programs
            A selected subset of model bases.
        n_random
            The number of repeats.
        verbose
            Verbosity.
        test_data_only
            Whether to evaluate models only on testing datasets.
        split_type
            The type of data splitting. "random" and "cv" are supported. Ignored when ``load_from_previous`` is True.
        load_from_previous
            Load the state of a previous run (mostly because of an unexpected interruption).
        **kwargs
            Arguments for :meth:`tabensemb.model.AbstractModel.train`

        Notes
        -----
        The results of a continuous run and a continued run (``load_from_previous=True``) are consistent.

        Returns
        -------
        dict
            A dict in the following format:
            {keys: programs, values: {keys: model names, values: {keys: ["Training", "Testing", "Validation"], values:
            (Predicted values, true values)}}
        """
        programs_predictions = {}
        for program in programs:
            programs_predictions[program] = {}

        if load_from_previous:
            if not os.path.exists(
                os.path.join(self.project_root, "cv")
            ) or not os.path.isfile(
                os.path.join(self.project_root, "cv", "cv_state.pkl")
            ):
                raise Exception(f"No previous state to load from.")
            with open(
                os.path.join(self.project_root, "cv", "cv_state.pkl"), "rb"
            ) as file:
                current_state = pickle.load(file)
            start_i = current_state["i_random"]
            self.load_state(current_state["trainer"])
            programs_predictions = current_state["programs_predictions"]
            reloaded_once_predictions = current_state["once_predictions"]
            skip_program = reloaded_once_predictions is not None
            if start_i >= n_random:
                raise Exception(
                    f"The loaded state is incompatible with the current setting."
                )
            print(f"Previous cross validation state is loaded.")
            split_type = (
                "cv"
                if self.datamodule.datasplitter.cv_generator is not None
                else "random"
            )
        else:
            start_i = 0
            skip_program = False
            reloaded_once_predictions = None
            if split_type == "cv" and not self.datamodule.datasplitter.support_cv:
                warnings.warn(
                    f"{self.datamodule.datasplitter.__class__.__name__} does not support cross validation splitting. "
                    f"Use its original regime instead."
                )
                split_type = "random"
            self.datamodule.datasplitter.reset_cv(
                cv=n_random if split_type == "cv" else -1
            )
            if n_random > 0 and not os.path.exists(
                os.path.join(self.project_root, "cv")
            ):
                os.mkdir(os.path.join(self.project_root, "cv"))

        def func_save_state(state):
            with open(
                os.path.join(self.project_root, "cv", "cv_state.pkl"), "wb"
            ) as file:
                pickle.dump(state, file)

        for i in range(start_i, n_random):
            if verbose:
                print(
                    f"----------------------------{i + 1}/{n_random} {split_type}----------------------------"
                )
            trainer_state = cp(self)
            if not skip_program:
                current_state = {
                    "trainer": trainer_state,
                    "i_random": i,
                    "programs_predictions": programs_predictions,
                    "once_predictions": None,
                }
                func_save_state(current_state)
            with HiddenPrints(disable_std=not verbose):
                set_random_seed(tabensemb.setting["random_seed"] + i)
                self.load_data()
            once_predictions = {} if not skip_program else reloaded_once_predictions
            for program in programs:
                if skip_program:
                    if program in once_predictions.keys():
                        print(f"Skipping finished model base {program}")
                        continue
                    else:
                        skip_program = False
                modelbase = self.get_modelbase(program)
                modelbase.train(dump_trainer=True, verbose=verbose, **kwargs)
                predictions = modelbase._predict_all(
                    verbose=verbose, test_data_only=test_data_only
                )
                once_predictions[program] = predictions
                for model_name, value in predictions.items():
                    if model_name in programs_predictions[program].keys():
                        # current_predictions is a reference, so modifications are directly applied to it.
                        current_predictions = programs_predictions[program][model_name]

                        def append_once(key):
                            current_predictions[key] = (
                                np.append(
                                    current_predictions[key][0], value[key][0], axis=0
                                ),
                                np.append(
                                    current_predictions[key][1], value[key][1], axis=0
                                ),
                            )

                        append_once("Testing")
                        if not test_data_only:
                            append_once("Training")
                            append_once("Validation")
                    else:
                        programs_predictions[program][model_name] = value
                # It is expected that only model bases in self is changed. datamodule is not updated because the cross
                # validation status should remain before load_data() is called.
                trainer_state.modelbases = self.modelbases
                current_state = {
                    "trainer": trainer_state,
                    "i_random": i,
                    "programs_predictions": programs_predictions,
                    "once_predictions": once_predictions,
                }
                func_save_state(current_state)
            df_once = self._cal_leaderboard(
                once_predictions, test_data_only=test_data_only, save=False
            )
            df_once.to_csv(
                os.path.join(self.project_root, "cv", f"leaderboard_cv_{i}.csv")
            )
            trainer_state.modelbases = self.modelbases
            current_state = {
                "trainer": trainer_state,
                "i_random": i + 1,
                "programs_predictions": programs_predictions,
                "once_predictions": None,
            }
            func_save_state(current_state)
            if verbose:
                print(
                    f"--------------------------End {i + 1}/{n_random} {split_type}--------------------------"
                )
        return programs_predictions

    def get_leaderboard(
        self,
        test_data_only: bool = False,
        dump_trainer: bool = True,
        cross_validation: int = 0,
        verbose: bool = True,
        load_from_previous: bool = False,
        split_type: str = "cv",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run all model bases with/without cross validation for a leaderboard.

        Parameters
        ----------
        test_data_only
            Whether to evaluate models only on testing datasets.
        dump_trainer
            Whether to save the :class:`Trainer`.
        cross_validation
            The number of cross-validation. See :meth:`cross_validation`. 0 to evaluate current trained models on the
            current dataset.
        verbose
            Verbosity.
        load_from_previous
            Load the state of a previous run (mostly because of an unexpected interruption).
        split_type
            The type of data splitting. "random" and "cv" are supported. Ignored when ``load_from_previous`` is True.
        **kwargs
            Arguments for :meth:`tabensemb.model.AbstractModel.train`

        Returns
        -------
        pd.DataFrame
            The leaderboard.
        """
        if len(self.modelbases) == 0:
            raise Exception(
                f"No modelbase available. Run trainer.add_modelbases() first."
            )
        if cross_validation != 0:
            programs_predictions = self.cross_validation(
                programs=self.modelbases_names,
                n_random=cross_validation,
                verbose=verbose,
                test_data_only=test_data_only,
                load_from_previous=load_from_previous,
                split_type=split_type,
                **kwargs,
            )
        else:
            programs_predictions = {}
            for modelbase in self.modelbases:
                print(f"{modelbase.program} metrics")
                programs_predictions[modelbase.program] = modelbase._predict_all(
                    verbose=verbose, test_data_only=test_data_only
                )

        df_leaderboard = self._cal_leaderboard(
            programs_predictions, test_data_only=test_data_only
        )
        if dump_trainer:
            save_trainer(self)
        return df_leaderboard

    def get_approx_cv_leaderboard(
        self, leaderboard: pd.DataFrame, save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate approximated averages and standard errors based on :meth:`cross_validation` results in the folder
        ``self.project_root/cv``.

        Parameters
        ----------
        leaderboard
            A reference leaderboard to be filled by avg and std, and to sort the returned DataFrame.
        save
            Save returned results locally with names "leaderboard_approx_mean.csv" and "leaderboard_approx_std.csv"

        Returns
        -------
        pd.DataFrame
            Averages in the same format as the input ``leaderboard``. There is an additional column "Rank".
        pd.DataFrame
            Standard errors in the same format as the input ``leaderboard``. There is an additional column "Rank".

        Notes
        -----
        The returned results are approximations of the precise leaderboard from ``get_leaderboard``. Some metrics like
        RMSE may be different because data-point-wise and cross-validation-wise averaging are different.
        """
        leaderboard_mean = leaderboard.copy()
        leaderboard_std = leaderboard.copy()
        leaderboard_mean["Rank"] = np.nan
        leaderboard_std["Rank"] = np.nan
        if not os.path.exists(os.path.join(self.project_root, "cv")):
            warnings.warn(
                f"Cross validation folder {os.path.join(self.project_root, 'cv')} not found."
            )
            leaderboard_mean["Rank"] = leaderboard.index.values + 1
            leaderboard_std.loc[
                :, np.setdiff1d(leaderboard_std.columns, ["Program", "Model"])
            ] = 0
            return leaderboard_mean, leaderboard_std
        df_cvs, programs, models, metrics = self._read_cv_leaderboards()
        modelwise_cv = self.get_modelwise_cv_metrics()
        for program in programs:
            program_models = models[program]
            for model in program_models:
                res_cv = modelwise_cv[program][model]
                # If numeric_only=True, only "Rank" is calculated somehow.
                mean = res_cv[metrics].mean(0, numeric_only=False)
                std = res_cv[metrics].std(0, numeric_only=False)
                where_model = leaderboard_std.loc[
                    (leaderboard_std["Program"] == program)
                    & (leaderboard_std["Model"] == model)
                ].index[0]
                leaderboard_mean.loc[where_model, mean.index] = mean
                leaderboard_std.loc[where_model, std.index] = std
        if save:
            leaderboard_mean.to_csv(
                os.path.join(self.project_root, "leaderboard_approx_mean.csv"),
                index=False,
            )
            leaderboard_std.to_csv(
                os.path.join(self.project_root, "leaderboard_approx_std.csv"),
                index=False,
            )
        return leaderboard_mean, leaderboard_std

    def get_modelwise_cv_metrics(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Assemble cross-validation results in the folder ``self.project_root/cv`` for metrics of each model in each
        model base.

        Returns
        -------
        dict
            A dict of dicts where each of them contains metrics of cross-validation of one model.
        """
        df_cvs, programs, models, metrics = self._read_cv_leaderboards()
        res_cvs = {}
        for program in programs:
            res_cvs[program] = {}
            program_models = models[program]
            for model in program_models:
                res_cvs[program][model] = pd.DataFrame(
                    columns=df_cvs[0].columns, index=np.arange(len(df_cvs))
                )
                cv_metrics = np.zeros((len(df_cvs), len(metrics)))
                for cv_idx, df_cv in enumerate(df_cvs):
                    where_model = (df_cv["Program"] == program) & (
                        df_cv["Model"] == model
                    )
                    model_metrics = df_cv.loc[where_model][metrics].values.flatten()
                    cv_metrics[cv_idx, :] = model_metrics
                res_cvs[program][model][metrics] = cv_metrics
                res_cvs[program][model]["Program"] = program
                res_cvs[program][model]["Model"] = model
        return res_cvs

    def _read_cv_leaderboards(
        self,
    ) -> Tuple[List[pd.DataFrame], List[str], Dict[str, List[str]], List[str]]:
        """
        Read cross-validation leaderboards in the folder ``self.project_root/cv``.

        Returns
        -------
        list
            Cross validation leaderboards
        list
            Model base names
        dict
            Model names in each model base
        list
            Metric names.
        """
        if not os.path.exists(os.path.join(self.project_root, "cv")):
            raise Exception(
                f"Cross validation folder {os.path.join(self.project_root, 'cv')} not found."
            )
        cvs = sorted(
            [
                i
                for i in os.listdir(os.path.join(self.project_root, "cv"))
                if "leaderboard_cv" in i
            ]
        )
        df_cvs = [
            pd.read_csv(os.path.join(self.project_root, "cv", cv), index_col=0)
            for cv in cvs
        ]
        programs = list(np.unique(df_cvs[0]["Program"].values))
        models = {
            a: list(df_cvs[0].loc[np.where(df_cvs[0]["Program"] == a)[0], "Model"])
            for a in programs
        }
        for df_cv in df_cvs:
            df_cv["Rank"] = df_cv.index.values + 1
        metrics = list(np.setdiff1d(df_cvs[0].columns, ["Program", "Model"]))
        return df_cvs, programs, models, metrics

    def _cal_leaderboard(
        self,
        programs_predictions: Dict[
            str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        ],
        metrics: List[str] = None,
        test_data_only: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the leaderboard based on results from :meth:`cross_validation` or
        :meth:`tabensemb.model.AbstractModel._predict_all`.

        Parameters
        ----------
        programs_predictions
            Results from :meth:`cross_validation`, or assembled results from
            :meth:`tabensemb.model.AbstractModel._predict_all`. See the source code of
            :meth:`get_leaderboard` for details.
        metrics
            The metrics that have been implemented in :func:`tabensemb.utils.utils.metric_sklearn`.
        test_data_only
            Whether to evaluate models only on testing datasets.
        save
            Whether to save the leaderboard locally and as an attribute in the :class:`Trainer`.

        Returns
        -------
        pd.DataFrame
            The leaderboard dataframe.
        """
        if metrics is None:
            metrics = {
                "regression": REGRESSION_METRICS,
                "binary": BINARY_METRICS,
                "multiclass": MULTICLASS_METRICS,
            }[self.datamodule.task]
        dfs = []
        for modelbase_name in self.modelbases_names:
            df = self._metrics(
                programs_predictions[modelbase_name],
                metrics,
                test_data_only=test_data_only,
            )
            df["Program"] = modelbase_name
            dfs.append(df)

        df_leaderboard = pd.concat(dfs, axis=0, ignore_index=True)
        sorted_by = metrics[0].upper()
        df_leaderboard.sort_values(
            f"Testing {sorted_by}" if not test_data_only else sorted_by, inplace=True
        )
        df_leaderboard.reset_index(drop=True, inplace=True)
        df_leaderboard = df_leaderboard[["Program"] + list(df_leaderboard.columns)[:-1]]
        if save:
            df_leaderboard.to_csv(os.path.join(self.project_root, "leaderboard.csv"))
            self.leaderboard = df_leaderboard
            if os.path.exists(os.path.join(self.project_root, "cv")):
                self.get_approx_cv_leaderboard(df_leaderboard, save=True)
        return df_leaderboard

    def _plot_subplots(
        self,
        ls: List[str],
        ls_kwarg_name: str,
        meth_name: str,
        with_title: bool = False,
        fontsize: float = 12,
        xlabel: str = None,
        ylabel: str = None,
        get_figsize_kwargs: dict = None,
        figure_kwargs: dict = None,
        meth_fix_kwargs: dict = None,
    ):
        """
        Iterate over a list to plot subplots.

        Parameters
        ----------
        ls
            The list to be iterated.
        ls_kwarg_name
            The argument name of the components in ``ls`` when the component is passed to ``meth_name``.
        meth_name
            The method to plot on a subplot. It has an argument named ``ax`` which indicates the subplot.
        with_title
            Whether each subplot has a title, which is the components in ``ls``.
        fontsize
            ``plt.rcParams["font.size"]``
        xlabel
            The overall xlabel.
        ylabel
            The overall ylabel.
        get_figsize_kwargs
            Arguments for :func:`tabensemb.utils.utils.get_figsize`.
        figure_kwargs
            Arguments for ``plt.figure()``
        meth_fix_kwargs
            Fixed arguments of ``meth_name`` (except for ``ax`` and ``ls_kwarg_name``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure that has plotted subplots.
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        get_figsize_kwargs_ = update_defaults_by_kwargs(
            dict(max_col=4, width_per_item=3, height_per_item=3, max_width=14),
            get_figsize_kwargs,
        )
        figsize, width, height = get_figsize(n=len(ls), **get_figsize_kwargs_)

        fig = plt.figure(figsize=figsize, **figure_kwargs_)
        plt.rcParams["font.size"] = fontsize
        for idx, name in enumerate(ls):
            ax = plt.subplot(height, width, idx + 1)
            if with_title:
                ax.set_title(name, {"fontsize": fontsize})
            getattr(self, meth_name)(ax=ax, **{ls_kwarg_name: name}, **meth_fix_kwargs)

        ax = fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig

    def _after_plot(self, fig_name, tight_layout=False, savefig_kwargs: dict = None):
        """
        Set the layout, save the current figure, show the figure if in a notebook, and close the figure.

        Parameters
        ----------
        fig_name
            The path to save the figure.
        tight_layout
            If True, ``plt.tight_layout`` is called.
        savefig_kwargs
            Arguments for ``plt.savefig``.
        """
        savefig_kwargs_ = update_defaults_by_kwargs(dict(), savefig_kwargs)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(fig_name, **savefig_kwargs_)
        if is_notebook():
            plt.show()
        plt.close()

    def plot_truth_pred_all(
        self,
        program: str,
        fontsize=14,
        get_figsize_kwargs: dict = None,
        figure_kwargs: dict = None,
        save_show_close: bool = True,
        **kwargs,
    ) -> Union[None, matplotlib.figure.Figure]:
        """
        Compare ground truth and prediction for all models in a model base.

        Parameters
        ----------
        program
            The selected model base.
        fontsize
            ``plt.rcParams["font.size"]``
        get_figsize_kwargs
            Arguments for :func:`tabensemb.utils.utils.get_figsize`.
        figure_kwargs
            Arguments for ``plt.figure()``
        save_show_close
            Whether to save, show (in the notebook), and close the figure, or return the ``matplotlib.figure.Figure``
            instance.
        kwargs
            Arguments for :meth:`plot_truth_pred`

        Returns
        -------
        matplotlib.figure.Figure
            The figure if ``save_show_close`` is False.
        """
        modelbase = self.get_modelbase(program)
        model_names = modelbase.get_model_names()

        fig = self._plot_subplots(
            ls=model_names,
            ls_kwarg_name="model_name",
            meth_name="plot_truth_pred",
            meth_fix_kwargs=dict(program=program, **kwargs),
            fontsize=fontsize,
            with_title=True,
            xlabel="Ground truth",
            ylabel="Prediction",
            get_figsize_kwargs=get_figsize_kwargs,
            figure_kwargs=figure_kwargs,
        )

        return (
            self._after_plot(
                fig_name=os.path.join(self.project_root, program, f"truth_pred.pdf"),
                tight_layout=False,
            )
            if save_show_close
            else fig
        )

    def plot_truth_pred(
        self,
        program: str,
        model_name: str,
        log_trans: bool = True,
        upper_lim=9,
        ax=None,
        figure_kwargs: dict = None,
        scatter_kwargs: dict = None,
        legend_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Compare ground truth and prediction for one model.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base
        log_trans
            Whether the label data is in log scale.
        upper_lim
            The upper limit of x/y-axis.
        fontsize
            ``plt.rcParams["font.size"]``
        ax
            ``matplotlib.axes.Axes``
        figure_kwargs
            Arguments for ``plt.figure()``
        scatter_kwargs
            Arguments for ``plt.scatter()``
        legend_kwargs
            Arguments for ``plt.legend()``
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        legend_kwargs_ = update_defaults_by_kwargs(
            dict(loc="upper left", markerscale=1.5, handlelength=0.2, handleheight=0.9),
            legend_kwargs,
        )

        given_ax = ax is not None
        if not given_ax:
            plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        prediction = self.get_modelbase(program)._predict_model(
            model_name=model_name, test_data_only=False
        )

        def plot_one(name, color, marker):
            pred_y, y = prediction[name]
            r2 = metric_sklearn(y, pred_y, "r2")
            loss = metric_sklearn(y, pred_y, "mse")
            print(f"{name} MSE Loss: {loss:.4f}, R2: {r2:.4f}")
            scatter_kwargs_ = update_defaults_by_kwargs(
                dict(
                    s=20,
                    color=color,
                    marker=marker,
                    label=f"{name} dataset ($R^2$={r2:.3f})",
                    linewidth=0.4,
                    edgecolors="k",
                ),
                scatter_kwargs,
            )
            ax.scatter(
                10**y if log_trans else y,
                10**pred_y if log_trans else pred_y,
                **scatter_kwargs_,
            )

        plot_one("Training", clr[0], "o")
        plot_one("Validation", clr[2], "o")
        plot_one("Testing", clr[1], "o")

        if log_trans:
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.plot(
                np.linspace(0, 10**upper_lim, 100),
                np.linspace(0, 10**upper_lim, 100),
                "--",
                c="grey",
                alpha=0.2,
            )
            locmin = matplotlib.ticker.LogLocator(
                base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
            )

            # ax.set_aspect("equal", "box")

            ax.xaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

            ax.set_xlim(1, 10**upper_lim)
            ax.set_ylim(1, 10**upper_lim)
            ax.set_box_aspect(1)
        else:
            # ax.set_aspect("equal", "box")
            lx, rx = ax.get_xlim()
            ly, ry = ax.get_ylim()
            l = np.min([lx, ly])
            r = np.max([rx, ry])

            ax.plot(
                np.linspace(l, r, 100),
                np.linspace(l, r, 100),
                "--",
                c="grey",
                alpha=0.2,
            )

            ax.set_xlim(left=l, right=r)
            ax.set_ylim(bottom=l, top=r)
            ax.set_box_aspect(1)

        ax.legend(**legend_kwargs_)

        if not given_ax:
            ax.set_xlabel("Ground truth")
            ax.set_ylabel("Prediction")
            s = model_name.replace("/", "_")
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root, program, f"{s}_truth_pred.pdf"
                    ),
                    tight_layout=False,
                )
        return ax

    def cal_feature_importance(
        self, program: str, model_name: str, method: str = "permutation", **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate feature importance using a specified model. If the model base is a
        :class:`tabensemb.model.TorchModel`, ``captum`` or ``shap`` is called to make permutations. If the model base
        is only a :class:`tabensemb.model.AbstractModel`, the calculation will be much slower.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        method
            The method to calculate importance. "permutation" or "shap".
        kwargs
            kwargs for :meth:`tabensemb.model.AbstractModel.cal_feature_importance`

        Returns
        -------
        attr
            Values of feature importance.
        importance_names
            Corresponding feature names. If the model base is a ``TorchModel``, all features including derived unstacked
            features will be included. Otherwise, only :meth:`all_feature_names` will be considered.

        See Also
        --------
        :meth:`tabensemb.model.AbstractModel.cal_feature_importance`,
        :meth:`tabensemb.model.TorchModel.cal_feature_importance`
        """
        modelbase = self.get_modelbase(program)
        return modelbase.cal_feature_importance(
            model_name=model_name, method=method, **kwargs
        )

    def cal_shap(self, program: str, model_name: str, **kwargs) -> np.ndarray:
        """
        Calculate SHAP values using a specified model. If the model base is a :class:`tabensemb.model.TorchModel`, the
        ``shap.DeepExplainer`` is used. Otherwise, ``shap.KernelExplainer`` is called, which is much slower, and
        shap.kmeans is called to summarize the training data to 10 samples as the background data and 10 random samples
        in the testing set is explained, which will bias the results.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        kwargs
            kwargs for :meth:`tabensemb.model.AbstractModel.cal_shap`

        Returns
        -------
        attr
            The SHAP values. If the model base is a `TorchModel`, all features including derived unstacked features will
            be included. Otherwise, only :meth:`all_feature_names` will be considered.

        See Also
        --------
        :meth:`tabensemb.model.AbstractModel.cal_shap`,
        :meth:`tabensemb.model.TorchModel.cal_shap`

        """
        modelbase = self.get_modelbase(program)
        return modelbase.cal_shap(model_name=model_name, **kwargs)

    def plot_feature_importance(
        self,
        program: str,
        model_name: str,
        method: str = "permutation",
        clr=None,
        ax=None,
        figure_kwargs: dict = None,
        bar_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot feature importance of a model using :meth:`cal_feature_importance`.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        fig_size
            The figure size.
        method
            The method to calculate feature importance. "permutation" or "shap".
        clr
            A seaborn color palette. For example seaborn.color_palette("deep").
        ax
            ``matplotlib.axes.Axes``
        figure_kwargs
            Arguments for ``plt.figure``
        bar_kwargs
            Arguments for ``seaborn.barplot``.
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attr, names = self.cal_feature_importance(
            program=program, model_name=model_name, method=method
        )

        clr = sns.color_palette("deep") if clr is None else clr
        bar_kwargs_ = update_defaults_by_kwargs(
            dict(linewidth=1, edgecolor="k", orient="h"), bar_kwargs
        )
        figure_kwargs_ = update_defaults_by_kwargs(dict(figsize=(7, 4)), figure_kwargs)

        # if feature type is not assigned in config files, the feature is from dataderiver.
        pal = []
        for key in names:
            if key in self.cont_feature_names:
                c = (
                    clr[self.args["feature_names_type"][key]]
                    if key in self.args["feature_names_type"].keys()
                    else clr[self.args["feature_types"].index("Derived")]
                )
            elif key in self.cat_feature_names:
                c = clr[self.args["feature_types"].index("Categorical")]
            else:
                c = clr[self.args["feature_types"].index("Derived")]
            pal.append(c)

        clr_map = dict()
        for idx, feature_type in enumerate(self.args["feature_types"]):
            clr_map[feature_type] = clr[idx]

        where_effective = np.abs(attr) > 1e-5
        effective_names = np.array(names)[where_effective]
        not_effective = list(np.setdiff1d(names, effective_names))
        if len(not_effective) > 0:
            print(f"Feature importance less than 1e-5: {not_effective}")
        attr = attr[where_effective]
        pal = [pal[x] for x in np.where(where_effective)[0]]

        given_ax = ax is not None
        if not given_ax:
            plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        df = pd.DataFrame(columns=["feature", "attr", "clr"])
        df["feature"] = effective_names
        df["attr"] = np.abs(attr) / np.sum(np.abs(attr))
        df["pal"] = pal
        df.sort_values(by="attr", inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)

        ax.set_axisbelow(True)
        x = df["feature"].values
        y = df["attr"].values

        palette = df["pal"]

        # ax.set_facecolor((0.97,0.97,0.97))
        # plt.grid(axis='x')
        plt.grid(axis="x", linewidth=0.2)
        # plt.barh(x,y, color= [clr_map[name] for name in x])
        sns.barplot(x=y, y=x, palette=palette, **bar_kwargs_)
        # ax.set_xlim([0, 1])

        legend = ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, color=value, ec="k", label=key)
                for key, value in clr_map.items()
            ],
            loc="lower right",
            handleheight=2,
            fancybox=False,
            frameon=False,
        )

        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor([1, 1, 1, 0.4])

        if not given_ax:
            if method == "permutation":
                ax.set_xlabel("Permutation feature importance")
            elif method == "shap":
                ax.set_xlabel("SHAP feature importance")
            else:
                ax.set_xlabel("Feature importance")
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"feature_importance_{program}_{model_name}_{method}.png",
                    ),
                    tight_layout=True,
                )
        return ax

    def plot_partial_dependence_all(
        self,
        program: str,
        model_name: str,
        fontsize=12,
        figure_kwargs: dict = None,
        get_figsize_kwargs: dict = None,
        save_show_close: bool = True,
        **kwargs,
    ) -> Union[None, matplotlib.figure.Figure]:
        """
        Calculate and plot partial dependence plots with bootstrapping.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        fontsize
            ``plt.rcParams["font.size"]``
        figure_kwargs
            Arguments for ``plt.figure``.
        get_figsize_kwargs
            Arguments for :func:`tabensemb.utils.utils.get_figsize`.
        save_show_close
            Whether to save, show (in the notebook), and close the figure, or return the ``matplotlib.figure.Figure``
            instance.
        kwargs
            Arguments for :meth:`plot_partial_dependence`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure if ``save_show_close`` is False.
        """
        fig = self._plot_subplots(
            ls=self.all_feature_names,
            ls_kwarg_name="feature",
            meth_name="plot_partial_dependence",
            meth_fix_kwargs=dict(program=program, model_name=model_name, **kwargs),
            fontsize=fontsize,
            with_title=True,
            xlabel=r"Value of predictors ($10\%$-$90\%$ percentile)",
            ylabel="Predicted target",
            get_figsize_kwargs=get_figsize_kwargs,
            figure_kwargs=figure_kwargs,
        )
        return (
            self._after_plot(
                fig_name=os.path.join(
                    self.project_root, f"partial_dependence_{program}_{model_name}.pdf"
                ),
                tight_layout=False,
            )
            if save_show_close
            else fig
        )

    def plot_partial_dependence(
        self,
        program: str,
        model_name: str,
        feature: str,
        ax=None,
        refit: bool = True,
        log_trans: bool = True,
        lower_lim: float = 2,
        upper_lim: float = 7,
        n_bootstrap: int = 1,
        grid_size: int = 30,
        CI: float = 0.95,
        verbose: bool = True,
        figure_kwargs: dict = None,
        plot_kwargs: dict = None,
        fill_between_kwargs: dict = None,
        bar_kwargs: dict = None,
        hist_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Calculate and plot a partial dependence plot with bootstrapping for a feature.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        feature
            The selected feature to calculate partial dependence.
        ax
            ``matplotlib.axes.Axes``
        refit
            Whether to refit models on bootstrapped datasets. See :meth:`_bootstrap`.
        log_trans
            Whether the label data is in log scale.
        lower_lim
            Lower limit of all pdp plots.
        upper_lim
            Upper limit of all pdp plot.
        n_bootstrap
            The number of bootstrap evaluations. It should be greater than 0.
        grid_size
            The number of steps of all pdp plot.
        CI
            The confidence interval of pdp results calculated across multiple bootstrap runs.
        verbose
            Verbosity
        figure_kwargs
            Arguments for ``plt.figure``.
        plot_kwargs
            Arguments for ``ax.plot``.
        fill_between_kwargs
            Arguments for ``ax.fill_between``.
        bar_kwargs
            Arguments for ``ax.bar`` (used for frequencies of categorical features).
        hist_kwargs
            Arguments for ``ax.hist`` (used for histograms of continuous features).
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        (
            x_values_list,
            mean_pdp_list,
            ci_left_list,
            ci_right_list,
        ) = self.cal_partial_dependence(
            feature_subset=[feature],
            program=program,
            model_name=model_name,
            df=self.datamodule.X_train,
            derived_data=self.datamodule.D_train,
            n_bootstrap=n_bootstrap,
            refit=refit,
            grid_size=grid_size,
            verbose=verbose,
            rederive=True,
            percentile=80,
            CI=CI,
            average=True,
        )
        x_values = x_values_list[0]
        mean_pdp = mean_pdp_list[0]
        ci_left = ci_left_list[0]
        ci_right = ci_right_list[0]

        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        plot_kwargs_ = update_defaults_by_kwargs(
            dict(color="k", linewidth=0.7), plot_kwargs
        )
        fill_between_kwargs_ = update_defaults_by_kwargs(
            dict(alpha=0.4, color="k", edgecolor=None), fill_between_kwargs
        )

        given_ax = ax is not None
        if not given_ax:
            plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        def transform(value):
            if log_trans:
                return 10**value
            else:
                return value

        if feature not in self.cat_feature_names:
            ax.plot(x_values, transform(mean_pdp), **plot_kwargs_)

            ax.fill_between(
                x_values,
                transform(ci_left),
                transform(ci_right),
                **fill_between_kwargs_,
            )
        else:
            yerr = (
                np.abs(
                    np.vstack([transform(ci_left), transform(ci_right)])
                    - transform(mean_pdp)
                )
                if not np.isnan(ci_left).any()
                else None
            )
            ax.errorbar(x_values, transform(mean_pdp), yerr=yerr, **plot_kwargs_)

        # ax.set_xlim([0, 1])
        if log_trans:
            ax.set_yscale("log")
            ax.set_ylim([10**lower_lim, 10**upper_lim])
            locmin = matplotlib.ticker.LogLocator(
                base=10.0, subs=[0.1 * x for x in range(10)], numticks=20
            )
            # ax.xaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_locator(locmin)
            # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if np.min(x_values) < np.max(x_values):
            ax2 = ax.twinx()
            hist_kwargs_ = update_defaults_by_kwargs(dict(bins=x_values), hist_kwargs)
            self.plot_hist(
                feature=feature,
                ax=ax2,
                imputed=False,
                x_values=x_values,
                hist_kwargs=hist_kwargs_,
                bar_kwargs=bar_kwargs,
            )
            ax2.set_yticks([])
        else:
            ax2 = ax.twinx()
            ax2.text(0.5, 0.5, "Invalid interval", ha="center", va="center")
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.set_yticks([])

        if not given_ax:
            ax.set_ylabel("Predicted target")
            ax.set_xlabel(feature + r" ($10\%$-$90\%$ percentile)")
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"partial_dependence_{program}_{model_name}_{feature}.pdf",
                    ),
                    tight_layout=False,
                )
        return ax

    def cal_partial_dependence(
        self, feature_subset: List[str] = None, **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Calculate partial dependency. See the source code of :meth:`plot_partial_dependence` for its usage.

        Parameters
        ----------
        feature_subset
            A subset of :meth:`all_feature_names`.
        kwargs
            Arguments for :meth:`_bootstrap`.

        Returns
        -------
        list
            x values for each feature
        list
            pdp values for each feature
        list
            lower confidence limits for each feature
        list
            upper confidence limits for each feature
        """
        x_values_list = []
        mean_pdp_list = []
        ci_left_list = []
        ci_right_list = []

        for feature_idx, feature_name in enumerate(
            self.all_feature_names if feature_subset is None else feature_subset
        ):
            if kwargs["verbose"]:
                print("Calculate PDP: ", feature_name)

            x_value, model_predictions, ci_left, ci_right = self._bootstrap(
                focus_feature=feature_name, **kwargs
            )

            x_values_list.append(x_value)
            mean_pdp_list.append(model_predictions)
            ci_left_list.append(ci_left)
            ci_right_list.append(ci_right)

        return x_values_list, mean_pdp_list, ci_left_list, ci_right_list

    def plot_partial_err_all(
        self,
        program: str,
        model_name: str,
        fontsize=12,
        figure_kwargs: dict = None,
        get_figsize_kwargs: dict = None,
        save_show_close: bool = True,
        **kwargs,
    ) -> Union[None, matplotlib.figure.Figure]:
        """
        Calculate prediction absolute errors on the testing dataset, and plot histograms of high-error samples and
        low-error samples respectively.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        fontsize
            ``plt.rcParams["font.size"]``
        figure_kwargs
            Arguments for ``plt.figure``.
        get_figsize_kwargs
            Arguments for :func:`tabensemb.utils.utils.get_figsize`.
        save_show_close
            Whether to save, show (in the notebook), and close the figure, or return the ``matplotlib.figure.Figure``
            instance.
        kwargs
            Arguments for :meth:`plot_partial_err`

        Returns
        -------
        matplotlib.figure.Figure
            The figure if ``save_show_close`` is False.
        """
        fig = self._plot_subplots(
            ls=self.all_feature_names,
            ls_kwarg_name="feature",
            meth_name="plot_partial_err",
            meth_fix_kwargs=dict(program=program, model_name=model_name, **kwargs),
            fontsize=fontsize,
            with_title=True,
            xlabel="Value of predictors",
            ylabel="Prediction absolute error",
            get_figsize_kwargs=get_figsize_kwargs,
            figure_kwargs=figure_kwargs,
        )
        return (
            self._after_plot(
                fig_name=os.path.join(
                    self.project_root, f"partial_err_{program}_{model_name}.pdf"
                ),
                tight_layout=False,
            )
            if save_show_close
            else fig
        )

    def plot_partial_err(
        self,
        program: str,
        model_name: str,
        feature,
        thres=0.8,
        ax=None,
        figure_kwargs: dict = None,
        scatter_kwargs: dict = None,
        hist_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Calculate prediction absolute errors on the testing dataset, and plot histograms of high-error samples and
        low-error samples respectively for a single feature.

        Parameters
        ----------
        program
            The selected model base.
        model_name
            The selected model in the model base.
        feature
            The selected feature.
        thres
            The absolute error threshold to identify high-error samples and low-error samples.
        ax
            ``matplotlib.axes.Axes``
        figure_kwargs
            Arguments for ``plt.figure``.
        scatter_kwargs
            Arguments for ``ax.scatter()``
        hist_kwargs
            Arguments for ``ax.hist()``
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        scatter_kwargs_ = update_defaults_by_kwargs(dict(s=1), scatter_kwargs)
        hist_kwargs_ = update_defaults_by_kwargs(
            dict(density=True, alpha=0.2, rwidth=0.8), hist_kwargs
        )

        feature_data = self.df.loc[
            np.array(self.test_indices), self.all_feature_names
        ].reset_index(drop=True)

        truth = self.label_data.loc[self.test_indices, :].values.flatten()
        modelbase = self.get_modelbase(program)
        pred = modelbase.predict(
            df=self.datamodule.X_test,
            derived_data=self.datamodule.D_test,
            model_name=model_name,
        ).flatten()
        err = np.abs(truth - pred)
        high_err_data = feature_data.loc[np.where(err > thres)[0], :]
        high_err = err[np.where(err > thres)[0]]
        low_err_data = feature_data.loc[np.where(err <= thres)[0], :]
        low_err = err[np.where(err <= thres)[0]]

        given_ax = ax is not None
        if not given_ax:
            plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        ax.scatter(
            high_err_data[feature].values,
            high_err,
            color=clr[0],
            marker="s",
            **scatter_kwargs_,
        )
        ax.scatter(
            low_err_data[feature].values,
            low_err,
            color=clr[1],
            marker="^",
            **scatter_kwargs_,
        )

        ax.set_ylim([0, np.max(err) * 1.1])
        ax2 = ax.twinx()

        ax2.hist(
            [
                high_err_data[feature].values,
                low_err_data[feature].values,
            ],
            bins=np.linspace(
                np.min(feature_data[feature].values),
                np.max(feature_data[feature].values),
                20,
            ),
            color=clr[:2],
            **hist_kwargs_,
        )
        if feature in self.cat_feature_names:
            ticks = np.sort(np.unique(feature_data[feature].values)).astype(int)
            tick_label = [self.cat_feature_mapping[feature][x] for x in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_label)
            ax.set_xlim([-0.5, len(ticks) - 0.5])
            ax2.set_xlim([-0.5, len(ticks) - 0.5])

        # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
        # ax2.set_ylim([0,1])
        # ax2.set_xlim([np.min(x_values_list[idx]), np.max(x_values_list[idx])])
        ax2.set_yticks([])

        if not given_ax:
            ax.set_ylabel("Prediction absolute error")
            ax.set_xlabel(feature)
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"partial_err_{program}_{model_name}_{feature}.pdf",
                    ),
                    tight_layout=False,
                )
        return ax

    def plot_corr(
        self,
        fontsize: Any = 10,
        imputed=False,
        ax=None,
        figure_kwargs: dict = None,
        imshow_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot Pearson correlation coefficients among features and the target.

        Parameters
        ----------
        fontsize
            The ``fontsize`` argument for matplotlib.
        imputed
            Whether the imputed dataset should be considered. If False, some NaN coefficients may exist for features
            with missing values.
        ax
            ``matplotlib.axes.Axes``
        figure_kwargs
            Arguments for ``plt.figure``.
        imshow_kwargs
            Arguments for ``plt.imshow``.
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(
            dict(figsize=(10, 10)), figure_kwargs
        )
        imshow_kwargs_ = update_defaults_by_kwargs(dict(cmap="bwr"), imshow_kwargs)

        cont_feature_names = self.cont_feature_names + self.label_name
        # sns.reset_defaults()
        given_ax = ax is not None
        if not given_ax:
            fig = plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)
        plt.box(on=True)
        corr = self.datamodule.cal_corr(imputed=imputed).values
        im = ax.imshow(corr, **imshow_kwargs_)
        ax.set_xticks(np.arange(len(cont_feature_names)))
        ax.set_yticks(np.arange(len(cont_feature_names)))

        ax.set_xticklabels(cont_feature_names, fontsize=fontsize)
        ax.set_yticklabels(cont_feature_names, fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        norm_corr = corr - (np.max(corr) + np.min(corr)) / 2
        norm_corr /= np.max(norm_corr)

        for i in range(len(cont_feature_names)):
            for j in range(len(cont_feature_names)):
                text = ax.text(
                    j,
                    i,
                    round(corr[i, j], 2),
                    ha="center",
                    va="center",
                    color="w" if np.abs(norm_corr[i, j]) > 0.3 else "k",
                    fontsize=fontsize,
                )

        if not given_ax and save_show_close:
            self._after_plot(
                fig_name=os.path.join(
                    self.project_root, f"corr{'_imputed' if imputed else ''}.pdf"
                ),
                tight_layout=True,
            )
        return ax

    def plot_pairplot(
        self, pairplot_kwargs: dict = None, save_show_close: bool = True
    ) -> Union[None, sns.axisgrid.PairGrid]:
        """
        Plot ``seaborn.pairplot`` among features and label. Kernel Density Estimation plots are on the diagonal.

        Parameters
        ----------
        pairplot_kwargs
            Arguments for ``seaborn.pairplot``.
        save_show_close
            Whether to save, show (in the notebook), and close the figure, or return the ``seaborn.axisgrid.PairGrid``
            instance.
        """
        pairplot_kwargs_ = update_defaults_by_kwargs(
            dict(corner=True, diag_kind="kde"), pairplot_kwargs
        )

        df_all = pd.concat(
            [self.unscaled_feature_data, self.unscaled_label_data], axis=1
        )
        grid = sns.pairplot(df_all, **pairplot_kwargs_)

        if save_show_close:
            self._after_plot(
                fig_name=os.path.join(self.project_root, "pair.jpg"),
                tight_layout=True,
            )
        else:
            return grid

    def plot_feature_box(
        self,
        imputed: bool = False,
        ax=None,
        figure_kwargs: dict = None,
        boxplot_kwargs: dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot boxplot of the tabular data.

        Parameters
        ----------
        imputed
            Whether the imputed dataset should be considered.
        ax
            ``matplotlib.axes.Axes``
        figure_kwargs
            Arguments for ``plt.figure``
        boxplot_kwargs
            Arguments for ``seaborn.boxplot``
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(figsize=(6, 6)), figure_kwargs)
        boxplot_kwargs_ = update_defaults_by_kwargs(
            dict(orient="h", linewidth=1, fliersize=4, flierprops={"marker": "o"}),
            boxplot_kwargs,
        )

        # sns.reset_defaults()
        given_ax = ax is not None
        if not given_ax:
            plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)
        bp = sns.boxplot(
            data=self.feature_data
            if imputed
            else self.datamodule.data_transform(
                self.datamodule.get_not_imputed_df()[self.cont_feature_names],
                scaler_only=True,
            ),
            ax=ax,
            **boxplot_kwargs_,
        )

        boxes = []

        for x in ax.get_children():
            if isinstance(x, matplotlib.patches.PathPatch):
                boxes.append(x)

        color = "#639FFF"

        for patch in boxes:
            patch.set_facecolor(color)

        plt.grid(linewidth=0.4, axis="x")
        ax.set_axisbelow(True)
        # ax.tick_params(axis='x', rotation=90)
        if not given_ax:
            plt.ylabel("Values (Scaled)")
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"feature_box{'_imputed' if imputed else ''}.pdf",
                    ),
                    tight_layout=True,
                )
        return ax

    def plot_hist_all(
        self,
        imputed=False,
        fontsize=12,
        get_figsize_kwargs: Dict = None,
        figure_kwargs: Dict = None,
        save_show_close: bool = True,
        **kwargs,
    ) -> Union[None, matplotlib.figure.Figure]:
        """
        Plot histograms of the tabular data.

        Parameters
        ----------
        imputed
            Whether the imputed dataset should be considered.
        figure_kwargs
            Arguments for ``plt.figure``.
        fontsize
            ``plt.rcParams["font.size"]``
        get_figsize_kwargs
            Arguments for :func:`tabensemb.utils.utils.get_figsize`.
        save_show_close
            Whether to save, show (in the notebook), and close the figure, or return the ``matplotlib.figure.Figure``
            instance.
        **kwargs
            Arguments for :meth:`plot_hist`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure if ``save_show_close`` is False.
        """
        fig = self._plot_subplots(
            ls=self.all_feature_names + self.label_name,
            ls_kwarg_name="feature",
            meth_name="plot_hist",
            meth_fix_kwargs=dict(imputed=imputed, **kwargs),
            fontsize=fontsize,
            with_title=True,
            xlabel="Value of predictors",
            ylabel="Density",
            get_figsize_kwargs=get_figsize_kwargs,
            figure_kwargs=figure_kwargs,
        )
        return (
            self._after_plot(
                fig_name=os.path.join(
                    self.project_root, f"hist{'_imputed' if imputed else ''}.pdf"
                ),
                tight_layout=False,
            )
            if save_show_close
            else fig
        )

    def plot_hist(
        self,
        feature: str,
        ax=None,
        imputed=False,
        x_values=None,
        figure_kwargs: Dict = None,
        hist_kwargs: Dict = None,
        bar_kwargs: Dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot the histogram of a feature.

        Parameters
        ----------
        feature
            The selected feature.
        ax
            ``matplotlib.axes.Axes``
        imputed
            Whether the imputed dataset should be considered.
        x_values
            Unique values of the `feature`. If None, it will be inferred from the dataset.
        figure_kwargs
            Arguments for ``plt.figure``.
        bar_kwargs
            Arguments for ``ax.bar`` (used for frequencies of categorical features).
        hist_kwargs
            Arguments for ``ax.hist`` (used for histograms of continuous features).
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)

        given_ax = ax is not None
        if not given_ax:
            fig = plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        hist_data = (
            self.datamodule.categories_transform(self.datamodule.get_not_imputed_df())
            if not imputed
            else self.df
        )
        bar_kwargs_ = update_defaults_by_kwargs(
            dict(color="k", alpha=0.2, edgecolor=None), bar_kwargs
        )
        hist_kwargs_ = update_defaults_by_kwargs(
            dict(density=True, color="k", alpha=0.2, rwidth=0.95), hist_kwargs
        )
        x_values = (
            np.sort(np.unique(hist_data[feature].values.flatten()))
            if x_values is None
            else x_values
        )
        x_values = x_values[np.isfinite(x_values)]
        if len(x_values) > 0:
            if feature not in self.cat_feature_names:
                ax.hist(hist_data[feature], **hist_kwargs_)
                # sns.rugplot(data=chosen_data, height=0.05, ax=ax2, color='k')
                # ax2.set_ylim([0,1])
                ax.set_xlim([np.min(x_values), np.max(x_values)])
            else:
                counts = np.array(
                    [len(np.where(hist_data[feature].values == x)[0]) for x in x_values]
                )
                ax.bar(
                    x_values,
                    counts,
                    tick_label=[self.cat_feature_mapping[feature][x] for x in x_values],
                    **bar_kwargs_,
                )
                ax.set_xlim([np.min(x_values) - 0.5, np.max(x_values) + 0.5])
                count_range = np.max(counts) - np.min(counts)
                ax.set_ylim(
                    [
                        np.min(counts) - 0.2 * count_range,
                        np.max(counts) + 0.2 * count_range,
                    ]
                )
        else:
            ax.text(0.5, 0.5, "Invalid interval", ha="center", va="center")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_yticks([])

        if not given_ax:
            ax.set_ylabel("Density")
            ax.set_xlabel(feature)
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"hist{'_imputed' if imputed else ''}_{feature}.pdf",
                    ),
                    tight_layout=False,
                )
        return ax

    def plot_on_one_axes(
        self,
        meth_name: Union[str, List],
        meth_kwargs_ls: List[Dict],
        fontsize: float = 12,
        xlabel: str = None,
        ylabel: str = None,
        ax=None,
        meth_fix_kwargs: dict = None,
        figure_kwargs: Dict = None,
        legend_kwargs: Dict = None,
        save_show_close: bool = True,
        fig_name: str = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot multiple items on one ``matplotlib.axes.Axes``.

        Parameters
        ----------
        meth_name
            The method or a list of methods to plot multiple items. The method should have an argument named `ax` which
            indicates the subplot.
        meth_kwargs_ls
            A list of arguments of the corresponding ``meth_name`` (except for ``ax``).
        fontsize
            ``plt.rcParams["font.size"]``
        xlabel
            The overall xlabel.
        ylabel
            The overall ylabel.
        ax
            ``matplotlib.axes.Axes``
        meth_fix_kwargs
            Fixed arguments of ``meth_name`` (except for ``ax``, ``ls_kwarg_name``, and those given in
            ``meth_kwargs_ls``).
        figure_kwargs
            Arguments for ``plt.figure``.
        legend_kwargs
            Arguments for ``plt.legend()``
        save_show_close
            Whether to save, show (in the notebook), and close the figure if ``ax`` is not given.
        fig_name
            The path to save the figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        legend_kwargs_ = update_defaults_by_kwargs(dict(), legend_kwargs)
        meth_fix_kwargs_ = update_defaults_by_kwargs(dict(), meth_fix_kwargs)

        given_ax = ax is not None
        if not given_ax:
            fig = plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        plt.rcParams["font.size"] = fontsize
        if isinstance(meth_name, str):
            meth_name = [meth_name] * len(meth_kwargs_ls)
        for meth, meth_kwargs in zip(meth_name, meth_kwargs_ls):
            getattr(self, meth)(ax=ax, **meth_kwargs, **meth_fix_kwargs_)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(**legend_kwargs_)

        if not given_ax and save_show_close:
            self._after_plot(
                fig_name=os.path.join(
                    self.project_root,
                    "plot_on_one_axes.pdf",
                )
                if fig_name is None
                else fig_name,
                tight_layout=False,
            )
        return ax

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        ax=None,
        imputed: bool = False,
        figure_kwargs: Dict = None,
        scatter_kwargs: Dict = None,
        select_by_value_kwargs: Dict = None,
        save_show_close: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot one column versus the other.

        Parameters
        ----------
        x_col
            The column for the x-axis.
        y_col
            The column for the y-axis.
        ax
            ``matplotlib.axes.Axes``
        imputed
            Whether the imputed dataset should be considered.
        figure_kwargs
            Arguments for ``plt.figure``.
        scatter_kwargs
            Arguments for ``plt.scatter()``
        select_by_value_kwargs
            Arguments for :meth:`tabensemb.data.datamodule.DataModule.select_by_value`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        figure_kwargs_ = update_defaults_by_kwargs(dict(), figure_kwargs)
        scatter_kwargs_ = update_defaults_by_kwargs(dict(), scatter_kwargs)

        given_ax = ax is not None
        if not given_ax:
            fig = plt.figure(**figure_kwargs_)
            ax = plt.subplot(111)
        plt.sca(ax)

        df = self.df if imputed else self.datamodule.get_not_imputed_df()
        if select_by_value_kwargs is not None:
            indices = self.datamodule.select_by_value(**select_by_value_kwargs)
        else:
            indices = df.index

        x = df.loc[indices, x_col].values.flatten()
        y = df.loc[indices, y_col].values.flatten()
        isna = np.union1d(np.where(np.isnan(x))[0], np.where(np.isnan(y))[0])
        notna = np.setdiff1d(np.arange(len(x)), isna)

        ax.scatter(x[notna], y[notna], **scatter_kwargs_)

        if not given_ax:
            ax.set_ylabel(y_col)
            ax.set_xlabel(x_col)
            if save_show_close:
                self._after_plot(
                    fig_name=os.path.join(
                        self.project_root,
                        f"scatter_{x_col}_{y_col}.pdf",
                    ),
                    tight_layout=False,
                )
        return ax

    def _bootstrap(
        self,
        program: str,
        df: pd.DataFrame,
        derived_data: Dict[str, np.ndarray],
        focus_feature: str,
        n_bootstrap: int = 1,
        grid_size: int = 30,
        verbose: bool = True,
        rederive: bool = True,
        refit: bool = True,
        resample: bool = True,
        percentile: float = 100,
        x_min: float = None,
        x_max: float = None,
        CI: float = 0.95,
        average: bool = True,
        model_name: str = "ThisWork",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make bootstrap resampling, fit the selected model on the resampled data, and assign sequential values to the
        selected feature to see how the prediction changes with respect to the feature.

        Cook, Thomas R., et al. Explaining Machine Learning by Bootstrapping Partial Dependence Functions and Shapley
        Values. No. RWP 21-12. 2021.

        Parameters
        ----------
        program
            The selected model base.
        df
            The tabular dataset.
        derived_data
            The derived data calculated using :meth:`derive_unstacked`.
        focus_feature
            The feature to assign sequential values.
        n_bootstrap
            The number of bootstrapping, fitting, and assigning runs.
        grid_size
            The number of sequential values.
        verbose
            Ignored.
        rederive
            Ignored. If the focus_feature is a derived stacked feature, derivation will not be performed on the
            bootstrap dataset. Otherwise, stacked/unstacked features will be re-derived.
        refit
            Whether to fit the model on the bootstrap dataset (with warm_start=True).
        resample
            Whether to do bootstrap resampling. Only recommended to False when n_bootstrap=1.
        percentile
            The percentile of the feature used to generate sequential values.
        x_min
            The lower limit of the generated sequential values. It will override the left percentile.
        x_max
            The upper limit of the generated sequential values. It will override the right percentile.
        CI
            The confidence interval level to evaluate bootstrapped predictions.
        average
            If True, CI will be calculated on results ``(grid_size, n_bootstrap)``where predictions for all samples are
            averaged for each bootstrap run.
            If False, CI will be calculated on results ``(grid_size, n_bootstrap*len(df))``.
        model_name
            The selected model in the model base.

        Returns
        -------
        np.ndarray
            The generated sequential values for the feature.
        np.ndarray
            Averaged predictions on the sequential values across multiple bootstrap runs and all samples.
        np.ndarray
            The left confidence interval.
        np.ndarray
            The right confidence interval.
        """
        from .utils import NoBayesOpt

        modelbase = self.get_modelbase(program)
        derived_data = self.datamodule.sort_derived_data(derived_data)
        if focus_feature in self.cont_feature_names:
            x_value = np.linspace(
                np.nanpercentile(df[focus_feature].values, (100 - percentile) / 2)
                if x_min is None
                else x_min,
                np.nanpercentile(df[focus_feature].values, 100 - (100 - percentile) / 2)
                if x_max is None
                else x_max,
                grid_size,
            )
        elif focus_feature in self.cat_feature_names:
            x_value = np.unique(df[focus_feature].values)
        else:
            raise Exception(f"{focus_feature} not available.")
        df = df.reset_index(drop=True)
        expected_value_bootstrap_replications = []
        for i_bootstrap in range(n_bootstrap):
            if resample:
                df_bootstrap = skresample(df)
            else:
                df_bootstrap = df
            tmp_derived_data = self.datamodule.get_derived_data_slice(
                derived_data, list(df_bootstrap.index)
            )
            df_bootstrap = df_bootstrap.reset_index(drop=True)
            bootstrap_model = modelbase.detach_model(model_name=model_name)
            if refit:
                with NoBayesOpt(self):
                    bootstrap_model.fit(
                        df_bootstrap,
                        model_subset=[model_name],
                        cont_feature_names=self.datamodule.dataprocessors[
                            0
                        ].record_cont_features,
                        cat_feature_names=self.datamodule.dataprocessors[
                            0
                        ].record_cat_features,
                        label_name=self.label_name,
                        verbose=False,
                        warm_start=True,
                    )
            bootstrap_model_predictions = []
            for value in x_value:
                df_perm = df_bootstrap.copy()
                df_perm[focus_feature] = value
                bootstrap_model_predictions.append(
                    bootstrap_model.predict(
                        df_perm,
                        model_name=model_name,
                        derived_data=tmp_derived_data
                        if focus_feature in self.derived_stacked_features
                        else None,  # To avoid rederiving stacked data
                    )
                )
            if average:
                expected_value_bootstrap_replications.append(
                    np.mean(np.hstack(bootstrap_model_predictions), axis=0)
                )
            else:
                expected_value_bootstrap_replications.append(
                    np.hstack(bootstrap_model_predictions)
                )

        expected_value_bootstrap_replications = np.vstack(
            expected_value_bootstrap_replications
        )
        ci_left = []
        ci_right = []
        mean_pred = []
        for col_idx in range(expected_value_bootstrap_replications.shape[1]):
            y_pred = expected_value_bootstrap_replications[:, col_idx]
            if len(y_pred) != 1 and len(np.unique(y_pred)) != 1:
                ci_int = st.norm.interval(CI, loc=np.mean(y_pred), scale=np.std(y_pred))
            else:
                ci_int = (np.nan, np.nan)
            ci_left.append(ci_int[0])
            ci_right.append(ci_int[1])
            mean_pred.append(np.mean(y_pred))

        return x_value, np.array(mean_pred), np.array(ci_left), np.array(ci_right)

    def load_state(self, trainer: "Trainer"):
        """
        Restore a :class:`Trainer` from a deep-copied state.

        Parameters
        ----------
        trainer
            A deep-copied status of a :class:`Trainer`.
        """
        # https://stackoverflow.com/questions/1216356/is-it-safe-to-replace-a-self-object-by-another-object-of-the-same-type-in-a-meth
        current_root = cp(self.project_root)
        self.__dict__.update(trainer.__dict__)
        # The update operation does not change the location of self. However, model bases contains another trainer
        # that points to another location if the state is loaded from disk.
        for model in self.modelbases:
            model.trainer = self
        self.set_path(current_root, verbose=False)
        for modelbase in self.modelbases:
            modelbase.set_path(os.path.join(current_root, modelbase.program))

    def get_best_model(self) -> Tuple[str, str]:
        """
        Get the best model from :attr:`leaderboard`.

        Returns
        -------
        str
            The name of a model base where the best model is.
        model_name
            The name of the best model.
        """
        if not hasattr(self, "leaderboard"):
            self.get_leaderboard(test_data_only=True, dump_trainer=False)
        return (
            self.leaderboard["Program"].values[0],
            self.leaderboard["Model"].values[0],
        )

    def _metrics(
        self,
        predictions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        metrics: List[str],
        test_data_only: bool,
    ) -> pd.DataFrame:
        """
        Calculate metrics for predictions from :meth:`tabensemb.model.AbstractModel._predict_all`.

        Parameters
        ----------
        predictions
            Results from :meth:`tabensemb.model.AbstractModel._predict_all`.
        metrics
            The metrics that have been implemented in :func:`tabensemb.utils.utils.metric_sklearn`.
        test_data_only
            Whether to evaluate models only on testing datasets.

        Returns
        -------
        pd.DataFrame
            A dataframe of metrics.
        """
        df_metrics = pd.DataFrame()
        for model_name, model_predictions in predictions.items():
            df = pd.DataFrame(index=[0])
            df["Model"] = model_name
            for tvt, (y_pred, y_true) in model_predictions.items():
                if test_data_only and tvt != "Testing":
                    continue
                for metric in metrics:
                    metric_value = auto_metric_sklearn(
                        y_true, y_pred, metric, self.datamodule.task
                    )
                    df[
                        tvt + " " + metric.upper()
                        if not test_data_only
                        else metric.upper()
                    ] = metric_value
            df_metrics = pd.concat([df_metrics, df], axis=0, ignore_index=True)

        return df_metrics


def save_trainer(
    trainer: Trainer, path: Union[os.PathLike, str] = None, verbose: bool = True
):
    """
    Pickling the :class:`Trainer` instance.

    Parameters
    ----------
    trainer
        The :class:`Trainer` to be saved.
    path
        The folder path to save the :class:`Trainer`.
    verbose
        Verbosity.
    """
    import pickle

    path = os.path.join(trainer.project_root, "trainer.pkl") if path is None else path
    with open(path, "wb") as outp:
        pickle.dump(trainer, outp, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(
            f"Trainer saved. To load the trainer, run trainer = load_trainer(path='{path}')"
        )


def load_trainer(path: Union[os.PathLike, str]) -> Trainer:
    """
    Loading a pickled :class:`Trainer`. Paths of the :class:`Trainer` and its model bases (i.e. :attr:`project_root`,
    :attr:`tabensemb.model.AbstractModel.root`, :attr:`tabensemb.model.base.ModelDict.root`, and
    :meth:`tabensemb.model.base.ModelDict.model_path.keys`) will be changed.

    Parameters
    ----------
    path
        Path of the :class:`Trainer`.

    Returns
    -------
    trainer
        The loaded :class:`Trainer`.
    """
    import pickle

    with open(path, "rb") as inp:
        trainer = pickle.load(inp)
    root = os.path.join(*os.path.split(path)[:-1])
    trainer.set_path(root, verbose=False)
    for modelbase in trainer.modelbases:
        modelbase.set_path(os.path.join(root, modelbase.program))
    return trainer
