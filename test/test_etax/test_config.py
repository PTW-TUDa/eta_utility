import copy
import re
from pathlib import Path

import pytest
import toml

from eta_utility.eta_x.config import ConfigOpt, ConfigOptSettings, ConfigOptSetup
from test.resources.config.config_python import config as python_dict


@pytest.fixture
def base_path() -> Path:
    return Path("test/resources/config")


@pytest.fixture
def config_dict() -> dict:
    return copy.deepcopy(python_dict)


class TestConfigOpt:
    @pytest.fixture
    def _load_config_mp(self, monkeypatch, config_dict) -> None:
        monkeypatch.setattr(ConfigOpt, "_load_config_file", lambda x: config_dict)

    @pytest.mark.usefixtures("_load_config_mp")
    def test_from_config_file(self):
        ConfigOpt.from_config_file(file="", path_root="")

    @pytest.mark.usefixtures("_load_config_mp")
    def test_from_config_file_overwrite(self):
        overwrite = {
            "agent_specific": {"solver_name": "foobar"},
            "environment_specific": {"scenario_files": [{"path": "foobar.csv"}]},
        }
        config_opt = ConfigOpt.from_config_file(file="", path_root="", overwrite=overwrite)

        assert config_opt.settings.agent["solver_name"] == "foobar"
        assert config_opt.settings.environment["scenario_files"][0]["path"] == "foobar.csv"

    @pytest.mark.usefixtures("_load_config_mp")
    def test_deprecated_from_json(self, recwarn):
        ConfigOpt.from_json(file="", path_root="")
        warning = recwarn.pop(DeprecationWarning)
        assert warning.message.args[0] == "Use `ConfigOpt.from_config_file()` instead."

    def test_from_config_file_fail(self, config_dict, monkeypatch):
        config_dict.pop("settings")
        monkeypatch.setattr(ConfigOpt, "_load_config_file", lambda x: config_dict)
        error_msg = re.escape("The section 'settings' is not present in configuration file foobar.")
        with pytest.raises(ValueError, match=error_msg):
            ConfigOpt.from_config_file(file="foobar", path_root="")

    config_names = ["config1", "config2", "config3", "config1.json", "config2.toml", "config3.yaml"]

    @pytest.mark.parametrize("config_name", config_names)
    def test_load_config_file(self, config_name: str, base_path: Path):
        file_path = base_path / f"{config_name}"
        read_dict = ConfigOpt._load_config_file(file_path)
        assert read_dict == python_dict

    def test_load_config_file_fail(self, base_path: Path):
        file_path = base_path / "no_configfile"
        error_msg = re.escape(f"Config file not found: {file_path}")
        with pytest.raises(FileNotFoundError, match=error_msg):
            ConfigOpt._load_config_file(file_path)

    def test_load_config_file_fail2(self, base_path: Path, monkeypatch):
        file_path = base_path / "config2"
        monkeypatch.setattr(toml, "load", lambda x: ["settings"])
        error_msg = re.escape(f"Config file {file_path} must define a dictionary of options.")
        with pytest.raises(TypeError, match=error_msg):
            ConfigOpt._load_config_file(file_path)

    def test_build_config_opt_dictfail(self, config_dict: dict, caplog):
        caplog.set_level(10)  # Set log level to debug

        config_dict["setup"] = ["foobar"]
        config_dict.pop("environment_specific")
        error_msg = re.escape("'setup' section must be a dictionary of settings.")
        with pytest.raises(TypeError, match=error_msg):
            ConfigOpt.from_dict(config_dict, file="", path_root=Path())
        log_msg = "Section 'environment_specific' not present in configuration, assuming it is empty."
        assert log_msg in caplog.messages

    def test_build_config_opt_altname(self, config_dict: dict, caplog):
        config_dict["interaction_environment_specific"] = {"foo": "bar"}
        config_dict.pop("interaction_env_specific")
        config_dict["agentspecific"] = {"solver_name": "foobar"}
        config = ConfigOpt.from_dict(config_dict, file="", path_root=Path())
        assert config.settings.interaction_env["foo"] == "bar"
        log_msg = (
            "Specified configuration value 'agentspecific' in the setup section of the configuration "
            "was not recognized and is ignored."
        )
        assert log_msg in caplog.messages

    def test_build_config_opt_pathfail(self, config_dict: dict):
        config_dict["paths"].pop("relpath_results")
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). Could not load config file."
        )
        with pytest.raises(ValueError, match=error_msg):
            ConfigOpt.from_dict(config_dict, file="", path_root=Path())


class TestConfigOptSetup:
    missing_classes = [
        "interaction_env_class",
        "interaction_env_package",
        "vectorizer_class",
        "vectorizer_package",
        "policy_class",
        "policy_package",
    ]

    @pytest.mark.parametrize("missing_class", missing_classes)
    def test_from_dict_no_fail(self, config_dict: dict, missing_class: str):
        config_dict["setup"].pop(missing_class)
        ConfigOptSetup.from_dict(config_dict["setup"])

    missing_classes = [
        "environment_import",
        "agent_import",
    ]

    @pytest.mark.parametrize("missing_class", missing_classes)
    def test_from_dict_fail(self, config_dict: dict, missing_class: str, caplog):
        config_dict["setup"].pop(missing_class)
        missing = missing_class.rsplit("_", 1)[0]
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). "
            f"Could not load config file. Missing values: {missing}"
        )
        with pytest.raises(ValueError, match=error_msg):
            ConfigOptSetup.from_dict(config_dict["setup"])
        log_msg = (
            f"'{missing}_import' or both of '{missing}_package' and " f"'{missing}_class' parameters must be specified."
        )
        assert log_msg in caplog.messages

    def test_module_not_found(self, config_dict: dict):
        config_dict["setup"]["environment_import"] = "foobar.FooBar"
        error_msg = "Could not find module 'foobar'. " "While importing class 'FooBar' from 'environment_import' value."
        with pytest.raises(ModuleNotFoundError, match=error_msg):
            ConfigOptSetup.from_dict(config_dict["setup"])

    def test_class_not_found(self, config_dict: dict):
        config_dict["setup"]["environment_import"] = "eta_utility.eta_x.envs.FooBar"
        error_msg = (
            "Could not find class 'FooBar' in module 'eta_utility.eta_x.envs'. "
            "While importing class 'FooBar' from 'environment_import' value."
        )
        with pytest.raises(AttributeError, match=error_msg):
            ConfigOptSetup.from_dict(config_dict["setup"])

    def test_from_config_opt_fail(self, config_dict: dict):
        config_dict["setup"].pop("environment_import")
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). Could not load config file."
        )
        with pytest.raises(ValueError, match=error_msg):
            ConfigOpt.from_dict(config_dict, file="", path_root=Path())

    def test_unrecognized_keys(self, config_dict: dict, caplog):
        config_dict["setup"]["foobar"] = "barfoo"
        ConfigOptSetup.from_dict(config_dict["setup"])
        log_msg = "Following values were not recognized in the config setup section and are ignored: foobar"
        assert log_msg in caplog.messages


class TestConfigOptSettings:
    @pytest.mark.parametrize("missing_key", ["n_episodes_play", "episode_duration", "sampling_time"])
    def test_from_dict_fail(self, config_dict: dict, missing_key: str):
        config_dict["settings"].pop(missing_key)
        error_msg = re.escape("Not all required values were found in settings (see log). Could not load config file.")
        with pytest.raises(ValueError, match=error_msg):
            ConfigOptSettings.from_dict(config_dict)

    def test_fail_from_config_opt(self, config_dict: dict):
        config_dict["settings"].pop("n_episodes_play")
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). Could not load config file."
        )
        with pytest.raises(ValueError, match=error_msg):
            ConfigOpt.from_dict(config_dict, file="", path_root=Path())
