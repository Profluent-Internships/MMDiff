import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.sample import sample
from src.train import train


@pytest.mark.slow
def test_train_sample(tmp_path, cfg_train, cfg_sample):
    """Train for 1 epoch with `train.py` and run inference with `sample.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_sample.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_sample):
        cfg_sample.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_sample)
    test_metric_dict, _ = sample(cfg_sample)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001
