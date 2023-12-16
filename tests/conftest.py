"""This file prepares config fixtures for other tests."""

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_sample_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="sample.yaml", return_hydra_config=True, overrides=["ckpt_path=."]
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_sample` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_sample(cfg_sample_global, tmp_path) -> DictConfig:
    cfg = cfg_sample_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()