# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import threading

import hydra
import lightning as L
import pyrootutils
import torch
from beartype.typing import List, Optional, Tuple
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from omegaconf import DictConfig

from src.utils.utils import run_fn_periodically, set_up_mlflow

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import register_custom_omegaconf_resolvers, utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.model.diffusion_cfg.sampling.force_na_seq_type is not None:
        assert cfg.model.diffusion_cfg.sampling.force_na_seq_type in [
            "DNA",
            "RNA",
        ], "The argument `force_na_seq_type` must be one of ['DNA', 'RNA']."

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        data_cfg=cfg.data.data_cfg,
        path_cfg=cfg.paths,
    )
    if cfg.get("ckpt_path") is not None:
        if os.path.exists(cfg.ckpt_path):
            log.info("Loading checkpoint!")
            model = model.load_from_checkpoint(
                # allow one to resume training with an older model using custom hyperparameters
                cfg.ckpt_path,
                strict=False,
                diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
                data_cfg=cfg.data.data_cfg,
                path_cfg=cfg.paths,
            )
        else:
            log.warning("Requested ckpt not found! Using new weights for training...")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    os.environ["WANDB_RESUME"] = "allow"  # enable resuming of WandB runs
    logging_with_mlflow = cfg.get("logger") and "mlflow" in cfg.get("logger")
    if logging_with_mlflow:
        set_up_mlflow()
        log.info("Spawning asynchronous MLFlow configuration process...")
        event = threading.Event()
        run_fn_periodically(fn=set_up_mlflow, event=event, time_interval_in_seconds=1800)
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if logging_with_mlflow:
        log.info("Establishing MLFlow run ID...")
        for logger_ in logger:
            if isinstance(logger_, MLFlowLogger):
                trainer.logger.experiment.run_id = logger_.run_id
                trainer.logger.experiment.update_run(trainer.logger.experiment.run_id, "RUNNING")
                break

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = (
            cfg.get("ckpt_path")
            if cfg.get("ckpt_path") is not None and os.path.exists(cfg.get("ckpt_path"))
            else None
        )
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
