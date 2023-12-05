import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def cfg_sample_config(cfg_sample: DictConfig):
    assert cfg_sample
    assert cfg_sample.data
    assert cfg_sample.model
    assert cfg_sample.trainer

    HydraConfig().set_config(cfg_sample)

    hydra.utils.instantiate(cfg_sample.data)
    hydra.utils.instantiate(cfg_sample.model)
    hydra.utils.instantiate(cfg_sample.trainer)
