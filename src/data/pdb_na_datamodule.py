# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
from functools import partial

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from lightning import LightningDataModule
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.components.pdb.data_utils import concat_np_features, length_batching
from src.data.components.pdb.pdb_na_dataset import (
    PDBNADataset,
    SamplingDataset,
    TrainSampler,
)


class PDBNADataModule(LightningDataModule):
    """A LightningDataModule for downloaded PDB protein-nucleic acid complex data."""

    def __init__(
        self,
        data_cfg: DictConfig,
        inference_cfg: Optional[DictConfig] = None,
        predict_min_length: int = 90,
        predict_max_length: int = 100,
        predict_length_step: int = 10,
        predict_samples_per_length: int = 30,
        predict_residue_molecule_type_mappings: List[str] = ["R:90", "D:40,A:60"],
        predict_residue_chain_mappings: List[str] = ["a:30,b:30,c:30", "a:20,b:20,c:60"],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.sampler_train: Optional[Sampler] = None
        self.sampler_val: Optional[Sampler] = None
        self.sampler_test: Optional[Sampler] = None

        # note: will be overridden using arguments from `inference_cfg` upon executing the `predict` phase
        self.data_predict: Dataset = SamplingDataset(
            min_length=predict_min_length,
            max_length=predict_max_length,
            length_step=predict_length_step,
            samples_per_length=predict_samples_per_length,
            residue_molecule_type_mappings=predict_residue_molecule_type_mappings,
            residue_chain_mappings=predict_residue_chain_mappings,
        )

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        self.ddpm = getattr(self.trainer.model, "ddpm", None)
        self.sequence_ddpm = getattr(self.trainer.model, "sequence_ddpm", None)
        # load and split datasets, and create their samplers, only if not done so already
        if stage == "fit" and (not self.data_train or not self.data_val):
            self.data_train = PDBNADataset(
                self.hparams.data_cfg,
                self.ddpm,
                is_training=True,
                sequence_ddpm=self.sequence_ddpm,
            )
            self.sampler_train = TrainSampler(
                data_conf=self.hparams.data_cfg,
                dataset=self.data_train,
                batch_size=self.hparams.data_cfg.batch_size,
                sample_mode=self.hparams.data_cfg.sample_mode,
            )
            if not (
                hasattr(self.trainer.strategy, "strategy_name")
                and "single_device" in self.trainer.strategy.strategy_name
            ):
                # when training in a distributed manner, wrap the custom `Sampler` for proper distributed sharding
                # note: one needs to wrap the `Sampler` manually here since Lightning does not allow one to leave the training `Sampler` unshuffled
                self.sampler_train = DistributedSamplerWrapper(
                    sampler=self.sampler_train, shuffle=False
                )
            self.data_val = PDBNADataset(
                self.hparams.data_cfg,
                self.ddpm,
                is_training=False,
                sequence_ddpm=self.sequence_ddpm,
            )
        if stage == "test" and not self.data_test:
            # note: this assumes, via sampling, that `self.data_test` will be assigned at least some PDBs not in `self.data_val`
            self.data_test = PDBNADataset(
                self.hparams.data_cfg,
                self.ddpm,
                is_training=False,
                sequence_ddpm=self.sequence_ddpm,
            )
        if stage == "predict":
            assert (
                self.hparams.inference_cfg is not None
            ), "Inference config must be provided to perform sampling."
            eval_data_cfg = copy.deepcopy(self.hparams.data_cfg)
            eval_data_cfg.num_eval_lengths = self.hparams.inference_cfg.samples.num_length_steps
            eval_data_cfg.samples_per_eval_length = (
                eval_data_cfg.eval_batch_size
            ) = self.hparams.inference_cfg.samples.samples_per_length
            eval_dataset = (
                PDBNADataset(
                    eval_data_cfg,
                    self.ddpm,
                    is_training=False,
                    sequence_ddpm=self.sequence_ddpm,
                    filter_eval_split=self.hparams.inference_cfg.filter_eval_split,
                    inference_cfg=self.hparams.inference_cfg,
                )
                if self.hparams.inference_cfg.run_statified_eval
                else None
            )
            self.data_predict = SamplingDataset(
                min_length=self.hparams.inference_cfg.samples.min_length,
                max_length=self.hparams.inference_cfg.samples.max_length,
                length_step=self.hparams.inference_cfg.samples.length_step,
                samples_per_length=self.hparams.inference_cfg.samples.samples_per_length,
                residue_molecule_type_mappings=self.hparams.inference_cfg.samples.residue_molecule_type_mappings,
                residue_chain_mappings=self.hparams.inference_cfg.samples.residue_chain_mappings,
                eval_dataset=eval_dataset,
            )
            if not self.data_train and self.hparams.inference_cfg.run_novelty_eval:
                # ensure the training dataset is constructed to evaluate the novelty of each generated (i.e., predicted) sample
                self.data_train = PDBNADataset(
                    self.hparams.data_cfg,
                    self.ddpm,
                    is_training=True,
                    sequence_ddpm=self.sequence_ddpm,
                )

    @beartype
    def create_data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        sampler: Optional[Sampler] = None,
        max_squared_res: int = 1e6,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        np_collate: bool = False,
        length_batch: bool = False,
        drop_last: bool = False,
    ):
        """Create a DataLoader with JAX-compatible data structures."""
        if np_collate:
            collate_fn = partial(concat_np_features, add_batch_dim=True)
        elif length_batch:
            collate_fn = partial(length_batching, max_squared_res=max_squared_res)
        else:
            collate_fn = None
        persistent_workers = True if num_workers > 0 else False
        prefetch_factor = None if num_workers == 0 else prefetch_factor
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            # Need fork https://github.com/facebookresearch/hydra/issues/964
            multiprocessing_context="fork" if num_workers != 0 else None,
        )

    def train_dataloader(self):
        return self.create_data_loader(
            dataset=self.data_train,
            batch_size=self.hparams.data_cfg.batch_size // self.trainer.world_size,
            shuffle=False,
            sampler=self.sampler_train,
            max_squared_res=self.hparams.data_cfg.max_squared_res,
            num_workers=self.hparams.data_cfg.num_workers,
            np_collate=False,
            length_batch=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return self.create_data_loader(
            dataset=self.data_val,
            batch_size=self.hparams.data_cfg.eval_batch_size,
            shuffle=False,
            sampler=self.sampler_val,
            num_workers=0,
            np_collate=False,
            length_batch=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.create_data_loader(
            dataset=self.data_test,
            batch_size=self.hparams.data_cfg.eval_batch_size,
            shuffle=False,
            sampler=self.sampler_test,
            num_workers=0,
            np_collate=False,
            length_batch=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = PDBNADataModule()
