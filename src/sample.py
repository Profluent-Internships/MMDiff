# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import os
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import pyrootutils
from beartype.typing import Tuple
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

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
def sample(cfg: DictConfig) -> Tuple[dict, dict]:
    """Samples given checkpoint on a datamodule prediction set.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path
    assert os.path.exists(
        cfg.paths.usalign_exec_path
    ), "A valid path to one's US-align executable must be provided."
    assert os.path.exists(
        cfg.paths.qtmclust_exec_path
    ), "A valid path to one's qTMclust executable must be provided."

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.model.diffusion_cfg.sampling.force_na_seq_type is not None:
        assert (
            cfg.model.diffusion_cfg.sampling.force_na_seq_type
            in ["DNA", "RNA"]
        ), "The argument `force_na_seq_type` must be one of ['DNA', 'RNA']."
    if cfg.inference.diffusion.force_na_seq_type is not None:
        assert (
            cfg.inference.diffusion.force_na_seq_type in ["DNA", "RNA"]
        ), "The argument `force_na_seq_type` must be one of ['DNA', 'RNA']."

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, inference_cfg=cfg.inference
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        data_cfg=cfg.data.data_cfg,
        path_cfg=cfg.paths,
        inference_cfg=cfg.inference,
    )

    log.info("Loading checkpoint!")
    model = model.load_from_checkpoint(
        cfg.ckpt_path,
        strict=False,
        diffusion_cfg=cfg.model.diffusion_cfg,
        data_cfg=cfg.data.data_cfg,
        path_cfg=cfg.paths,
        inference_cfg=cfg.inference,
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }

    trainer.predict(model=model, datamodule=datamodule)

    # evaluate parallel-capable sample metrics outside of prediction loop to avoid CUDA multiprocessing errors
    run_self_consistency_eval = (
        hasattr(cfg.inference, "run_self_consistency_eval")
        and cfg.inference.run_self_consistency_eval
    )
    run_diversity_eval = (
        hasattr(cfg.inference, "run_diversity_eval") and cfg.inference.run_diversity_eval
    )
    run_novelty_eval = (
        hasattr(cfg.inference, "run_novelty_eval") and cfg.inference.run_novelty_eval
    )
    predictions_csv_root_dir = Path(model.predictions_csv_path).parent
    combined_predictions_csv_path = os.path.join(
        predictions_csv_root_dir, "all_rank_predictions.csv"
    )
    if run_self_consistency_eval and not os.path.exists(combined_predictions_csv_path):
        # self-consistency is computed as part of the prediction loop, so we now need
        # to load predictions produced by all ranks and combine them into a single output CSV
        predictions_csv_paths = [
            pd.read_csv(csv_file, index_col=None)
            for csv_file in sorted(
                glob.glob(os.path.join(predictions_csv_root_dir, "rank_*_predictions.csv"))
            )
        ]
        samples_df = pd.concat(predictions_csv_paths)
        samples_df.to_csv(combined_predictions_csv_path, index=False)
        log.info(
            f"After running self-consistency evaluation, recorded combined predictions as the output CSV: {combined_predictions_csv_path}"
        )
    if run_diversity_eval:
        if os.path.exists(combined_predictions_csv_path):
            samples_df = pd.read_csv(combined_predictions_csv_path, index_col=None)
        else:
            # load predictions produced by all ranks and combine them into a single output CSV
            predictions_csv_paths = [
                pd.read_csv(csv_file)
                for csv_file in sorted(
                    glob.glob(os.path.join(predictions_csv_root_dir, "rank_*_predictions.csv"))
                )
            ]
            samples_df = pd.concat(predictions_csv_paths)
        # evaluate the diversity of the model's generated samples
        samples_df = model.eval_diversity(samples_df, cfg.paths.qtmclust_exec_path)
        samples_df.to_csv(combined_predictions_csv_path, index=False)
        log.info(
            f"After running diversity evaluation, recorded combined predictions as the output CSV: {combined_predictions_csv_path}"
        )
    if run_novelty_eval:
        if os.path.exists(combined_predictions_csv_path):
            samples_df = pd.read_csv(combined_predictions_csv_path, index_col=None)
        else:
            # load predictions produced by all ranks and combine them into a single output CSV
            predictions_csv_paths = [
                pd.read_csv(csv_file)
                for csv_file in sorted(
                    glob.glob(os.path.join(predictions_csv_root_dir, "rank_*_predictions.csv"))
                )
            ]
            samples_df = pd.concat(predictions_csv_paths)
        # evaluate the novelty of the model's generated samples, using the model's training dataset as a structural reference point
        samples_df = model.eval_novelty(
            samples_df,
            datamodule.data_train.csv,
            cfg.paths.usalign_exec_path,
        )
        samples_df.to_csv(combined_predictions_csv_path, index=False)
        log.info(
            f"After running novelty evaluation, recorded combined predictions as the output CSV: {combined_predictions_csv_path}"
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    sample(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
