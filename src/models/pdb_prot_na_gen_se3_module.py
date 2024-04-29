# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import glob
import os
import random
import shutil
import subprocess  # nosec
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
import tree
import wandb
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from biotite.structure import base_pairs
from biotite.structure.io import load_structure
from jaxtyping import jaxtyped
from lightning import LightningModule
from omegaconf import DictConfig
from pandarallel import pandarallel
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import src.models.components.pdb.analysis_utils as au
from src.data.components.pdb import all_atom, complex_constants
from src.data.components.pdb import data_utils as du
from src.data.components.pdb import rigid_utils as ru
from src.data.components.pdb import vocabulary
from src.data.components.pdb.data_transforms import (
    convert_na_aatype6_to_aatype9,
    convert_na_aatype9_to_aatype6,
)
from src.data.components.pdb.nucleotide_constants import (
    NA_ATOM37_N1_ATOM_INDEX,
    NA_ATOM37_N9_ATOM_INDEX,
    NA_SUPERVISED_ATOM_N9_ATOM_INDEX,
)
from src.data.components.pdb.pdb_na_dataset import (
    BATCH_TYPE,
    NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
    NUM_PROTEIN_ONEHOT_AATYPE_CLASSES,
    SAMPLING_ARGS,
    diffuse_sequence,
)
from src.models import (
    SE3_DIFFUSION_NETWORKS,
    EMAModelCheckpoint,
    Queue,
    detach_tensor_to_np,
    flip_traj,
    get_grad_norm,
    log_grad_flow_lite_mlflow,
    log_grad_flow_lite_wandb,
)
from src.models.components.pdb import metrics
from src.models.components.pdb.framediff import FrameDiff
from src.models.components.pdb.se3_diffusion import (
    BATCH_MASK_TENSOR_TYPE,
    FLOAT_TIMESTEP_TYPE,
    TIMESTEP_TYPE,
    SE3Diffusion,
    SE3ScoreNetwork,
)
from src.models.components.pdb.sequence_diffusion import GaussianSequenceDiffusion
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PDBProtNAGenSE3LitModule(LightningModule):
    """LightningModule for PDB protein-nucleic acid complex generation using SE(3) diffusion."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_cfg: DictConfig,
        diffusion_cfg: DictConfig,
        data_cfg: DictConfig,
        path_cfg: Optional[DictConfig] = None,
        inference_cfg: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # hyperparameters #

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # DDPM
        ddpm_modes = {"se3_unconditional": SE3Diffusion}
        self.ddpm_mode = diffusion_cfg.ddpm_mode
        assert (
            self.ddpm_mode in ddpm_modes
        ), f"Selected DDPM mode {self.ddpm_mode} is currently not supported."

        diffusion_networks = {"se3_score": SE3ScoreNetwork}
        dynamics_networks = {"framediff": FrameDiff}
        diffusion_mode_networks = {"se3_unconditional": SE3_DIFFUSION_NETWORKS}
        assert (
            diffusion_cfg.diffusion_network in diffusion_networks
        ), f"Selected diffusion network {diffusion_cfg.diffusion_networks} is currently not supported."
        assert (
            diffusion_cfg.diffusion_network in diffusion_mode_networks[diffusion_cfg.ddpm_mode]
        ), f"Selected diffusion network {diffusion_cfg.diffusion_network} is currently not supported for diffusion mode {diffusion_cfg.ddpm_mode}."
        assert (
            diffusion_cfg.dynamics_network in dynamics_networks
        ), f"Selected dynamics network {diffusion_cfg.dynamics_network} is currently not supported."

        # PyTorch modules #
        # SE(3) DDPM
        self.ddpm = ddpm_modes[self.ddpm_mode](diffusion_cfg=diffusion_cfg)

        # optional sequence DDPM
        self.diffuse_sequence = (
            hasattr(diffusion_cfg, "diffuse_sequence") and diffusion_cfg.diffuse_sequence
        ) or (
            inference_cfg
            and hasattr(inference_cfg.diffusion, "diffuse_sequence")
            and inference_cfg.diffusion.diffuse_sequence
        )
        num_sequence_timesteps = (
            inference_cfg.diffusion.num_t
            if inference_cfg
            else diffusion_cfg.num_sequence_timesteps
        )
        sequence_noise_schedule = (
            inference_cfg.diffusion.sequence_noise_schedule
            if inference_cfg
            else diffusion_cfg.sequence_noise_schedule
        )
        sequence_sample_distribution = (
            inference_cfg.diffusion.sequence_sample_distribution
            if inference_cfg
            else diffusion_cfg.sequence_sample_distribution
        )
        sequence_sample_distribution_gmm_means = (
            inference_cfg.diffusion.sequence_sample_distribution_gmm_means
            if inference_cfg
            else diffusion_cfg.sequence_sample_distribution_gmm_means
        )
        sequence_sample_distribution_gmm_variances = (
            inference_cfg.diffusion.sequence_sample_distribution_gmm_variances
            if inference_cfg
            else diffusion_cfg.sequence_sample_distribution_gmm_variances
        )
        self.sequence_ddpm = (
            GaussianSequenceDiffusion(
                num_timesteps=num_sequence_timesteps,
                noise_schedule=sequence_noise_schedule,
                sample_distribution=sequence_sample_distribution,
                sample_distribution_gmm_means=sequence_sample_distribution_gmm_means,
                sample_distribution_gmm_variances=sequence_sample_distribution_gmm_variances,
            )
            if self.diffuse_sequence
            else None
        )

        # protein-nucleic acid molecule dynamics network encapsulating SE(3) diffusion
        self.model = diffusion_networks[diffusion_cfg.diffusion_network](
            model_cfg=model_cfg, ddpm=self.ddpm, sequence_ddpm=self.sequence_ddpm
        )

        # training #
        if self.hparams.model_cfg.clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # losses #
        self.loss_cfg = self.hparams.model_cfg.loss_cfg

        # metrics #
        self.train_phase, self.val_phase, self.test_phase, self.predict_phase = (
            "train",
            "val",
            "test",
            "predict",
        )
        self.phases = [self.train_phase, self.val_phase, self.test_phase]
        self.metrics_to_monitor = [
            "loss",
            "rot_loss",
            "trans_loss",
            "bb_atom_loss",
            "dist_mat_loss",
            "mean_seq_length",
            "examples_per_second",
        ]
        if getattr(self.loss_cfg, "supervise_interfaces", False):
            self.metrics_to_monitor.append("interface_dist_mat_loss")
        if getattr(self.loss_cfg, "supervise_torsion_angles", False):
            self.metrics_to_monitor.append("torsion_loss")
        if self.diffuse_sequence:
            self.metrics_to_monitor.extend(["cce_seq_loss", "kl_seq_loss"])
        self.eval_metrics_to_monitor = []  # note: will be filled in during validation/testing
        for phase in self.phases:
            metrics_to_monitor = (
                self.metrics_to_monitor
                if phase == self.train_phase
                else self.eval_metrics_to_monitor
            )
            for metric in metrics_to_monitor:
                # note: individual metrics e.g., for averaging loss across batches
                setattr(self, f"{phase}_{metric}", torchmetrics.MeanMetric())

        self.loss_info_history = deque(maxlen=100)

        self.run_self_consistency_eval = (
            hasattr(inference_cfg, "run_self_consistency_eval")
            and inference_cfg.run_self_consistency_eval
        )

        # hook outputs #
        setattr(self, f"{self.val_phase}_step_outputs", [])
        setattr(self, f"{self.test_phase}_step_outputs", [])
        setattr(self, f"{self.predict_phase}_step_outputs", [])

    @beartype
    def standardize_batch_features(self, batch: BATCH_TYPE) -> BATCH_TYPE:
        # select and rename molecule features as necessary
        batch["node_types"] = (
            batch["diffused_aatype"]
            if self.training and self.diffuse_sequence
            else batch["aatype"]
        )
        batch["node_deoxy"] = (
            batch["diffused_atom_deoxy"]
            if self.training and self.diffuse_sequence
            else batch["atom_deoxy"]
        )
        batch["node_indices"] = batch["residue_index"]
        batch["node_chain_indices"] = batch["chain_idx"]
        batch["node_mask"] = batch["res_mask"]
        batch["edge_mask"] = batch["node_mask"].unsqueeze(-1) * batch["node_mask"].unsqueeze(-2)
        batch["fixed_node_mask"] = batch["fixed_mask"]
        del (
            batch["residue_index"],
            batch["chain_idx"],
            batch["res_mask"],
            batch["fixed_mask"],
        )
        if self.training and self.diffuse_sequence:
            batch["onehot_node_types"] = batch["diffused_onehot_aatype"]
            del batch["diffused_aatype"], batch["diffused_onehot_aatype"]
        else:
            del batch["aatype"]
        return batch

    @beartype
    def self_condition(self, batch: BATCH_TYPE) -> BATCH_TYPE:
        if self.training and self.diffuse_sequence:
            orig_node_types = batch["node_types"].clone()
            orig_onehot_node_types = batch["onehot_node_types"].clone()
            # to self-condition on a sequence, inject the back-mapped sequence from the `t + 1` timestep
            batch["node_types"] = batch["diffused_aatype_t_plus_1"]
            batch["onehot_node_types"] = batch["diffused_onehot_aatype_t_plus_1"]
        sc_batch = self.model(batch)
        batch["sc_pos_t"] = sc_batch["rigids"][..., 4:]  # note: main backbone atom positions
        if self.diffuse_sequence:
            batch["sc_aatype_t"] = sc_batch["pred_node_types"]
            if self.training:
                # after making a forward pass for self-conditioning, restore the original (onehot) node types corresponding to timestep `t`
                batch["node_types"] = orig_node_types
                batch["onehot_node_types"] = orig_onehot_node_types
        return batch

    @jaxtyped
    @beartype
    def set_t_features_in_batch(
        self, batch: BATCH_TYPE, t: TIMESTEP_TYPE, t_placeholder: TIMESTEP_TYPE
    ) -> BATCH_TYPE:
        batch["t"] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.ddpm.score_scaling(t)
        batch["rot_score_scaling"] = rot_score_scaling * t_placeholder
        batch["trans_score_scaling"] = trans_score_scaling * t_placeholder
        return batch

    @jaxtyped
    @beartype
    def normalize_loss(
        self,
        x: FLOAT_TIMESTEP_TYPE,
        batch_loss_mask: BATCH_MASK_TENSOR_TYPE,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        return x.sum() / (batch_loss_mask.sum() + eps)

    @beartype
    def compute_loss(
        self,
        batch: BATCH_TYPE,
        pred_outputs: Dict[str, Any],
        eps: float = 1e-10,
        separate_rot_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        bb_mask = batch["node_mask"]
        diffuse_mask = 1 - batch["fixed_node_mask"]
        is_protein_residue_mask = batch["is_protein_residue_mask"]
        is_na_residue_mask = batch["is_na_residue_mask"]
        protein_inputs_present = is_protein_residue_mask.any().item()
        na_inputs_present = is_na_residue_mask.any().item()
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch["rot_score"]
        gt_trans_score = batch["trans_score"]
        rot_score_scaling = batch["rot_score_scaling"]
        trans_score_scaling = batch["trans_score_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = pred_outputs["rot_score"] * diffuse_mask[..., None]
        pred_trans_score = pred_outputs["trans_score"] * diffuse_mask[..., None]
        pred_seq = pred_outputs.get("pred_node_types", None)

        pred_torsions = (pred_outputs["torsions"] * diffuse_mask[..., None]).view(
            *pred_outputs["torsions"].shape[:-1], -1, 2
        )
        gt_torsions = (
            batch["torsion_angles_sin_cos"].flatten(start_dim=-2)[
                ..., : pred_outputs["torsions"].shape[-1]
            ]
        ).view(*pred_outputs["torsions"].shape[:-1], -1, 2)

        # calculate translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score) ** 2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None] ** 2, dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + eps)

        # calculate translation `x0` loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self.loss_cfg.coordinate_scaling
        pred_trans_x0 = pred_outputs["rigids"][..., 4:] * self.loss_cfg.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + eps)

        trans_loss = trans_score_loss * (
            batch["t"] > self.loss_cfg.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self.loss_cfg.trans_x0_threshold)
        trans_loss *= self.loss_cfg.trans_loss_weight
        trans_loss *= int(self.hparams.diffusion_cfg.diffuse_translations)

        # calculate rotation loss
        if self.loss_cfg.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + separate_rot_eps)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + separate_rot_eps)

            # separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (loss_mask.sum(dim=-1) + eps)
            axis_loss *= self.loss_cfg.rot_loss_weight

            # separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + eps)
            angle_loss *= self.loss_cfg.rot_loss_weight
            angle_loss *= batch["t"] > self.loss_cfg.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(rot_mse / rot_score_scaling[:, None, None] ** 2, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + eps
            )
            rot_loss *= self.loss_cfg.rot_loss_weight
            rot_loss *= batch["t"] > self.loss_cfg.rot_loss_t_threshold
        rot_loss *= int(self.hparams.diffusion_cfg.diffuse_rotations)

        # calculate backbone atom loss
        atom37_supervised_mask = pred_outputs["atom37_supervised_mask"]
        protein_atom37_supervised_mask = atom37_supervised_mask[is_protein_residue_mask].view(
            is_protein_residue_mask.shape[0], -1, 37
        )
        na_atom37_supervised_mask = atom37_supervised_mask[is_na_residue_mask].view(
            is_na_residue_mask.shape[0], -1, 37
        )

        # construct atom position predictions in atom-37 format #
        protein_pred_atom37 = pred_outputs["atom37"][is_protein_residue_mask].reshape(
            is_protein_residue_mask.shape[0], -1, 37, 3
        )
        na_pred_atom37 = pred_outputs["atom37"][is_na_residue_mask].reshape(
            is_na_residue_mask.shape[0], -1, 37, 3
        )
        if protein_inputs_present and na_inputs_present:
            protein_pred_atom37 = protein_pred_atom37[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1, 3
            )
            na_pred_atom37 = na_pred_atom37[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1, 3
            )
            pred_atom37 = torch.cat(
                (F.pad(protein_pred_atom37, (0, 0, 0, 7, 0, 0, 0, 0)), na_pred_atom37), dim=1
            )
        elif protein_inputs_present:
            protein_pred_atom37 = protein_pred_atom37[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1, 3
            )
            pred_atom37 = F.pad(
                protein_pred_atom37, (0, 0, 0, 7, 0, 0, 0, 0)
            )  # note: proteins have only 5 backbone atoms to supervise
        elif na_inputs_present:
            na_pred_atom37 = na_pred_atom37[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1, 3
            )
            pred_atom37 = (
                na_pred_atom37  # note: nucleic acid molecules have 12 backbone atoms to supervise
            )
        else:
            raise Exception(
                "Either protein chains or nucleic acid chains must be provided to score the composite loss."
            )

        gt_atom37, atom37_mask = batch["atom37_pos"], batch["atom37_mask"]

        # construct ground-truth atom positions in atom-37 format #
        protein_gt_atom37 = gt_atom37[is_protein_residue_mask].reshape(
            is_protein_residue_mask.shape[0], -1, 37, 3
        )
        na_gt_atom37 = gt_atom37[is_na_residue_mask].reshape(
            is_na_residue_mask.shape[0], -1, 37, 3
        )
        if protein_inputs_present and na_inputs_present:
            protein_gt_atom37 = protein_gt_atom37[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1, 3
            )
            na_gt_atom37_ = na_gt_atom37[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1, 3
            )
            if self.loss_cfg.supervise_n1_atom_positions:
                # note: we need to make sure that N1 ground-truth (GT) atom positions are used
                # to supervise predicted N9 atom positions for NA pyrimidine residues (i.e., C, T, and U),
                # while N9 atom positions should still be used to supervise predicted N9 positions
                # for NA purine residues (i.e., A and G)
                na_gt_atom37_variable_n9_atom_mask = ~torch.any(
                    na_gt_atom37[:, :, NA_ATOM37_N9_ATOM_INDEX, :], dim=-1
                )
                na_gt_atom37_[:, :, NA_SUPERVISED_ATOM_N9_ATOM_INDEX][
                    na_gt_atom37_variable_n9_atom_mask
                ] = na_gt_atom37[:, :, NA_ATOM37_N1_ATOM_INDEX][na_gt_atom37_variable_n9_atom_mask]
            gt_atom37 = torch.cat(
                (F.pad(protein_gt_atom37, (0, 0, 0, 7, 0, 0, 0, 0)), na_gt_atom37_), dim=1
            )
        elif protein_inputs_present:
            protein_gt_atom37 = protein_gt_atom37[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1, 3
            )
            gt_atom37 = F.pad(
                protein_gt_atom37, (0, 0, 0, 7, 0, 0, 0, 0)
            )  # note: proteins have only 5 backbone atoms to supervise
        elif na_inputs_present:
            na_gt_atom37_ = na_gt_atom37[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1, 3
            )
            if self.loss_cfg.supervise_n1_atom_positions:
                # note: we need to make sure that N1 ground-truth (GT) atom positions are used
                # to supervise predicted N9 atom positions for NA pyrimidine residues (i.e., C, T, and U),
                # while N9 atom positions should still be used to supervise predicted N9 positions
                # for NA purine residues (i.e., A and G)
                na_gt_atom37_variable_n9_atom_mask = ~torch.any(
                    na_gt_atom37[:, :, NA_ATOM37_N9_ATOM_INDEX, :], dim=-1
                )
                na_gt_atom37_[:, :, NA_SUPERVISED_ATOM_N9_ATOM_INDEX][
                    na_gt_atom37_variable_n9_atom_mask
                ] = na_gt_atom37[:, :, NA_ATOM37_N1_ATOM_INDEX][na_gt_atom37_variable_n9_atom_mask]
            gt_atom37 = (
                na_gt_atom37_  # note: nucleic acid molecules have 12 backbone atoms to supervise
            )

        # construct ground-truth atom mask in atom-37 format #
        protein_atom37_mask = atom37_mask[is_protein_residue_mask].reshape(
            is_protein_residue_mask.shape[0], -1, 37
        )
        na_atom37_mask = atom37_mask[is_na_residue_mask].reshape(
            is_na_residue_mask.shape[0], -1, 37
        )
        if protein_inputs_present and na_inputs_present:
            protein_atom37_mask = protein_atom37_mask[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1
            )
            na_atom37_mask_ = na_atom37_mask[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1
            )
            if self.loss_cfg.supervise_n1_atom_positions:
                # note: we need to make sure that N1 ground-truth (GT) atom mask values are used
                # to supervise predicted N9 atom positions for NA pyrimidine residues (i.e., C, T, and U),
                # while N9 atom mask values should still be used to supervise predicted N9 positions
                # for NA purine residues (i.e., A and G)
                na_atom37_mask_variable_n9_atom_mask = ~torch.any(
                    na_atom37_mask[:, :, NA_ATOM37_N9_ATOM_INDEX].unsqueeze(-1), dim=-1
                )
                na_atom37_mask_[:, :, NA_SUPERVISED_ATOM_N9_ATOM_INDEX][
                    na_atom37_mask_variable_n9_atom_mask
                ] = na_atom37_mask[:, :, NA_ATOM37_N1_ATOM_INDEX][
                    na_atom37_mask_variable_n9_atom_mask
                ]
            atom37_mask = torch.cat(
                (F.pad(protein_atom37_mask, (0, 7, 0, 0, 0, 0)), na_atom37_mask_), dim=1
            )
        elif protein_inputs_present:
            protein_atom37_mask = protein_atom37_mask[protein_atom37_supervised_mask].view(
                *protein_atom37_supervised_mask.shape[:2], -1
            )
            atom37_mask = F.pad(
                protein_atom37_mask, (0, 7, 0, 0, 0, 0)
            )  # note: proteins have only 5 backbone atoms to supervise
        elif na_inputs_present:
            na_atom37_mask_ = na_atom37_mask[na_atom37_supervised_mask].view(
                *na_atom37_supervised_mask.shape[:2], -1
            )
            if self.loss_cfg.supervise_n1_atom_positions:
                # note: we need to make sure that N1 ground-truth (GT) atom mask values are used
                # to supervise predicted N9 atom positions for NA pyrimidine residues (i.e., C, T, and U),
                # while N9 atom mask values should still be used to supervise predicted N9 positions
                # for NA purine residues (i.e., A and G)
                na_atom37_mask_variable_n9_atom_mask = ~torch.any(
                    na_atom37_mask[:, :, NA_ATOM37_N9_ATOM_INDEX].unsqueeze(-1), dim=-1
                )
                na_atom37_mask_[:, :, NA_SUPERVISED_ATOM_N9_ATOM_INDEX][
                    na_atom37_mask_variable_n9_atom_mask
                ] = na_atom37_mask[:, :, NA_ATOM37_N1_ATOM_INDEX][
                    na_atom37_mask_variable_n9_atom_mask
                ]
            atom37_mask = (
                na_atom37_mask_  # note: nucleic acid molecules have 12 backbone atoms to supervise
            )

        gt_atom37 = gt_atom37
        atom37_mask = atom37_mask
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None], dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + eps)
        bb_atom_loss *= self.loss_cfg.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self.loss_cfg.bb_atom_loss_t_filter
        bb_atom_loss *= self.loss_cfg.aux_loss_weight

        # calculate pairwise distance loss
        num_backbone_atoms_per_res = gt_atom37.shape[-2]
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * num_backbone_atoms_per_res, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape(
            [batch_size, num_res * num_backbone_atoms_per_res, 3]
        )
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        # ensure the two following losses take into account which atoms are supervised along the backbone
        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, num_backbone_atoms_per_res))
        flat_loss_mask = flat_loss_mask.reshape(
            [batch_size, num_res * num_backbone_atoms_per_res]
        ) * atom37_mask.reshape([batch_size, num_res * num_backbone_atoms_per_res])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, num_backbone_atoms_per_res))
        flat_res_mask = flat_res_mask.reshape(
            [batch_size, num_res * num_backbone_atoms_per_res]
        ) * atom37_mask.reshape([batch_size, num_res * num_backbone_atoms_per_res])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        is_protein_residue_atom_mask = torch.tile(
            is_protein_residue_mask[:, :, None], (1, 1, num_backbone_atoms_per_res)
        )
        flat_is_protein_residue_atom_mask = is_protein_residue_atom_mask.reshape(
            [batch_size, num_res * num_backbone_atoms_per_res]
        )
        protein_pair_mask = (
            flat_is_protein_residue_atom_mask[..., None]
            * flat_is_protein_residue_atom_mask[:, None, :]
        )

        is_na_residue_atom_mask = torch.tile(
            is_na_residue_mask[:, :, None], (1, 1, num_backbone_atoms_per_res)
        )
        flat_is_na_residue_atom_mask = is_na_residue_atom_mask.reshape(
            [batch_size, num_res * num_backbone_atoms_per_res]
        )
        na_pair_mask = (
            flat_is_na_residue_atom_mask[..., None] * flat_is_na_residue_atom_mask[:, None, :]
        )

        # ensure no distance loss is calculated on any
        # protein residue pairwise distances >= 6 Angstrom
        # and nucleic acid residue pairwise distances >= 10 Angstrom
        proximity_mask = (protein_pair_mask & (gt_pair_dists < 6)) | (
            na_pair_mask & (gt_pair_dists < 10)
        )
        pair_dist_mask_ = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask_, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask_, dim=(1, 2)) - num_res
        dist_mat_loss *= self.loss_cfg.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self.loss_cfg.dist_mat_loss_t_filter
        dist_mat_loss *= self.loss_cfg.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        if getattr(self.loss_cfg, "supervise_interfaces", False):
            interface_mask = batch["inter_chain_interacting_residue_mask"]
            num_interface_res = interface_mask.sum(dim=-1)
            interface_mask = (
                torch.tile(interface_mask[:, :, None], (1, 1, num_backbone_atoms_per_res))
            ).flatten(start_dim=1, end_dim=2)
            interface_mask = interface_mask.unsqueeze(-1) * interface_mask.unsqueeze(-2)
            interface_pair_dist_mask = pair_dist_mask * interface_mask

            interface_dist_mat_loss = torch.sum(
                (gt_pair_dists - pred_pair_dists) ** 2 * interface_pair_dist_mask, dim=(1, 2)
            )
            interface_dist_mat_loss /= (
                torch.sum(interface_pair_dist_mask, dim=(1, 2)) - num_interface_res + eps
            )
            interface_dist_mat_loss *= self.loss_cfg.interface_dist_mat_loss_weight
            interface_dist_mat_loss *= batch["t"] < self.loss_cfg.interface_dist_mat_loss_t_filter
            interface_dist_mat_loss *= self.loss_cfg.aux_loss_weight

            final_loss += interface_dist_mat_loss

        if getattr(self.loss_cfg, "supervise_torsion_angles", False):
            # note: replicates most of the behavior of Algorithm 27 (torsionAngleLoss) of AlphaFold 2
            pred_torsion_norm = torch.norm(pred_torsions, dim=-1)
            pred_torsions = pred_torsions / pred_torsion_norm.unsqueeze(-1)
            diff_norm_gt = torch.norm(pred_torsions - gt_torsions, dim=-1)
            diff_norm = diff_norm_gt**2
            if protein_inputs_present:
                # note: recall that protein residues only have one torsion to supervise
                pred_torsions[..., 1:, :][is_protein_residue_mask] *= 0.0
                gt_torsions[..., 1:, :][is_protein_residue_mask] *= 0.0
            torsion_mask = (gt_torsions != 0.0).all(dim=-1)
            torsion_loss_mask = torsion_mask * loss_mask[..., None]
            torsion_diff_loss = torch.sum(diff_norm * torsion_loss_mask, dim=(1, 2)) / (
                torsion_loss_mask.sum(dim=(-1, -2)) + eps
            )
            torsion_norm_loss = torch.sum(
                torch.abs(pred_torsion_norm - 1) * torsion_loss_mask, dim=(-1, -2)
            ) / (torsion_loss_mask.sum(dim=(-1, -2)) + eps)
            torsion_loss = (
                torsion_diff_loss + self.loss_cfg.torsion_norm_loss_weight * torsion_norm_loss
            )
            torsion_loss *= self.loss_cfg.torsion_loss_weight
            torsion_loss *= batch["t"] < self.loss_cfg.torsion_loss_t_filter
            torsion_loss *= self.loss_cfg.aux_loss_weight

            final_loss += torsion_loss

        if self.diffuse_sequence:
            aatype = batch["aatype"].clone()
            aatype[batch["is_na_residue_mask"]] = convert_na_aatype6_to_aatype9(
                aatype=aatype[batch["is_na_residue_mask"]],
                deoxy_offset_mask=batch["atom_deoxy"][batch["is_na_residue_mask"]],
            )
            onehot_aatype = torch.nn.functional.one_hot(
                aatype, num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES
            ).float()
            cce_seq_loss = (
                F.cross_entropy(
                    input=pred_seq.permute(0, 2, 1),
                    target=onehot_aatype.permute(0, 2, 1),
                    reduction="none",
                )
                * loss_mask
            )
            cce_seq_loss = torch.sum(cce_seq_loss, dim=-1) / (loss_mask.sum(dim=-1) + eps)
            cce_seq_loss *= self.loss_cfg.cce_seq_loss_weight

            # supervise the model to keep its predicted sequences close to the sequence that would be generated at timestep `t - 1`
            sequence_feats = {
                "is_protein_residue_mask": batch["is_protein_residue_mask"],
                "is_na_residue_mask": batch["is_na_residue_mask"],
                "fixed_mask": batch["fixed_node_mask"],
                "aatype": pred_seq,
            }
            pred_onehot_aatype_t_minus_1 = diffuse_sequence(
                sequence_feats=sequence_feats,
                t=batch["t"],
                min_t=self.hparams.data_cfg.min_t,
                num_t=self.hparams.data_cfg.num_sequence_t,
                random_seed=torch.initial_seed(),
                sequence_ddpm=self.sequence_ddpm,
                training=self.training,
                eps=batch["diffused_aatype_eps"],
                t_discrete_jump=-1,  # diffuse the sequence to timestep `t - 1`
                onehot_sequence_input=True,  # note: will preserve gradients to `aatype` input
            )[0]["diffused_onehot_aatype"]
            pred_onehot_aatype_t_minus_1_log_probs = F.log_softmax(
                pred_onehot_aatype_t_minus_1,
                dim=-1,
            )
            sequence_feats["aatype"] = onehot_aatype
            onehot_aatype_t_minus_1 = diffuse_sequence(
                sequence_feats=sequence_feats,
                t=batch["t"],
                min_t=self.hparams.data_cfg.min_t,
                num_t=self.hparams.data_cfg.num_sequence_t,
                random_seed=torch.initial_seed(),
                sequence_ddpm=self.sequence_ddpm,
                training=self.training,
                eps=batch["diffused_aatype_eps"],
                t_discrete_jump=-1,  # diffuse the sequence to timestep `t - 1`
                onehot_sequence_input=True,  # note: will preserve gradients to `aatype` input
            )[0]["diffused_onehot_aatype"]
            onehot_aatype_t_minus_1_probs = F.softmax(
                onehot_aatype_t_minus_1,
                dim=-1,
            )
            kl_seq_loss = (
                F.kl_div(
                    input=pred_onehot_aatype_t_minus_1_log_probs,
                    target=onehot_aatype_t_minus_1_probs,
                    reduction="none",
                )
                * loss_mask[..., None]
            )
            kl_seq_loss = torch.sum(kl_seq_loss, dim=(-1, -2)) / (loss_mask.sum(dim=-1) + eps)
            kl_seq_loss *= self.loss_cfg.kl_seq_loss_weight

            seq_loss = cce_seq_loss + kl_seq_loss
            final_loss += seq_loss

        loss_info = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": self.normalize_loss(final_loss, batch_loss_mask),
            "rot_loss": self.normalize_loss(rot_loss, batch_loss_mask),
            "trans_loss": self.normalize_loss(trans_loss, batch_loss_mask),
            "bb_atom_loss": self.normalize_loss(bb_atom_loss, batch_loss_mask),
            "dist_mat_loss": self.normalize_loss(dist_mat_loss, batch_loss_mask),
            "mean_seq_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        if getattr(self.loss_cfg, "supervise_interfaces", False):
            loss_info["batch_interface_dist_mat_loss"] = interface_dist_mat_loss
            loss_info["interface_dist_mat_loss"] = self.normalize_loss(
                interface_dist_mat_loss, batch_loss_mask
            )
        if getattr(self.loss_cfg, "supervise_torsion_angles", False):
            loss_info["batch_torsion_loss"] = torsion_loss
            loss_info["torsion_loss"] = self.normalize_loss(torsion_loss, batch_loss_mask)
        if self.diffuse_sequence:
            loss_info["batch_cce_seq_loss"] = cce_seq_loss
            loss_info["batch_kl_seq_loss"] = kl_seq_loss
            loss_info["cce_seq_loss"] = self.normalize_loss(cce_seq_loss, batch_loss_mask)
            loss_info["kl_seq_loss"] = self.normalize_loss(kl_seq_loss, batch_loss_mask)

        # maintain a history of the past `N` steps, as this information may be helpful for debugging
        self.loss_info_history.append(
            {"batch": batch, "loss_info": loss_info, "pred_outputs": pred_outputs}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return self.normalize_loss(final_loss, batch_loss_mask), loss_info

    @beartype
    def forward(self, batch: BATCH_TYPE) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the loss and auxiliary metrics for dense input predictions."""
        # condition the model's prediction on features derived from ones of its prior predictions
        if (
            self.hparams.model_cfg.embedding.embed_self_conditioning
            and random.random() > 0.5  # nosec
        ):  # nosec
            with torch.no_grad():
                batch = self.self_condition(batch)

        # make predictions and collect their metadata
        pred_outputs = self.model(batch, plot_pred_positions=False)

        # score predictions
        loss, loss_info = self.compute_loss(batch, pred_outputs)

        return loss, loss_info

    @beartype
    def step(self, batch: BATCH_TYPE) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # make a forward pass and score it
        batch = self.standardize_batch_features(batch)
        loss, loss_info = self.forward(batch)
        return loss, loss_info

    def on_train_start(self):
        # note: by default, Lightning executes validation step sanity checks before training starts,
        # so we need to make sure that val_`metric` doesn't store any values from these checks
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            torchmetric.reset()

        # ensure directory for storing sampling outputs is defined
        if not getattr(self, "sampling_output_dir", None):
            self.sampling_output_dir = Path(self.trainer.default_root_dir)

    def on_train_epoch_start(self):
        # track time to complete a training step
        self.train_epoch_start_time = time.time()

        # ensure our custom `Sampler` is up-to-date on what the current epoch is
        if self.trainer.datamodule.sampler_train is not None:
            self.trainer.datamodule.sampler_train.set_epoch(self.current_epoch)
            if hasattr(self.trainer.datamodule.sampler_train, "dataset") and hasattr(
                self.trainer.datamodule.sampler_train.dataset, "_sampler"
            ):
                # propagate epoch updates to the custom `TrainSampler`
                self.trainer.datamodule.sampler_train.dataset._sampler.set_epoch(
                    self.current_epoch
                )
        if self.trainer.datamodule.sampler_val is not None:
            self.trainer.datamodule.sampler_val.set_epoch(self.current_epoch)
            if hasattr(self.trainer.datamodule.sampler_val, "dataset") and hasattr(
                self.trainer.datamodule.sampler_val.dataset, "_sampler"
            ):
                self.trainer.datamodule.sampler_val.dataset._sampler.set_epoch(self.current_epoch)

    @beartype
    def training_step(self, batch: BATCH_TYPE, batch_idx: int) -> Optional[torch.Tensor]:
        try:
            loss, metrics_dict = self.step(batch)
            self.train_epoch_start_time = time.time() - self.train_epoch_start_time
            metrics_dict["examples_per_second"] = torch.tensor(
                self.trainer.datamodule.hparams.data_cfg.batch_size / self.train_epoch_start_time
            )
            self.train_epoch_start_time = time.time()
        except RuntimeError as e:
            self.train_epoch_start_time = time.time()
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping training batch with index {batch_idx} due to OOM error...")
            return

        # skip backpropagation if loss was invalid
        if loss.isnan().any() or loss.isinf().any():
            del batch, loss, metrics_dict
            torch.cuda.empty_cache()
            log.info(f"Skipping training batch with index {batch_idx} due to invalid loss...")
            return

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard loss from forward pass while preserving its gradients
        metrics_dict["loss"] = loss.mean(0)

        # update metrics
        for metric in self.metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict["loss"]

    def on_train_epoch_end(self):
        # log metrics
        for metric in self.metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            self.log(f"{self.train_phase}/{metric}", torchmetric, prog_bar=False)

    def on_validation_start(self):
        # ensure directory for storing sampling outputs is defined
        if not getattr(self, "sampling_output_dir", None):
            self.sampling_output_dir = Path(self.trainer.default_root_dir)

    def on_validation_epoch_start(self):
        # track time to complete a validation epoch
        self.val_epoch_start_time = time.time()

        # make a backup checkpoint before sampling from the model #
        ema_callback = EMAModelCheckpoint._get_ema_callback(self.trainer)
        evaluating_with_ema_weights = (
            ema_callback is not None
            and ema_callback.ema_initialized
            and ema_callback.evaluate_ema_weights_instead
        )
        if evaluating_with_ema_weights:
            # ensure EMA weights are not saved as the primary weight set
            ema_callback.restore_original_weights(self)
        self.trainer.save_checkpoint(
            Path(self.trainer.checkpoint_callback.dirpath)
            / f"model_epoch_{self.current_epoch}_step_{self.global_step}_on_validation_epoch_start.ckpt"
        )
        if evaluating_with_ema_weights:
            # ensure EMA weights are subsequently used when sampling from the model
            ema_callback.replace_model_weights(self)

    @beartype
    def validation_step(self, batch: List[Any], batch_idx: int):
        try:
            batch_ = batch[0]
            batch_["pdb_code"] = batch[1]
            batch_["pdb_filepath"] = batch[2]
            sampling_outputs_list = self.eval_sampling(
                batch=batch_,
                sequence_noise_scale=self.hparams.diffusion_cfg.sampling.sequence_noise_scale,
                structure_noise_scale=self.hparams.diffusion_cfg.sampling.structure_noise_scale,
                apply_na_consensus_sampling=self.hparams.diffusion_cfg.sampling.apply_na_consensus_sampling,
                force_na_seq_type=self.hparams.diffusion_cfg.sampling.force_na_seq_type,
            )
            if sampling_outputs_list is None or not len(sampling_outputs_list):
                # signal that sampling with the current batch has failed
                return
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # collect outputs
        step_outputs = getattr(self, f"{self.val_phase}_step_outputs")
        step_outputs.append(sampling_outputs_list)

    def on_validation_epoch_end(self):
        # tag current sampling outputs according to the phase in which they were generated
        step_outputs = getattr(self, f"{self.val_phase}_step_outputs")
        sampling_outputs_list_ = [
            outputs
            for outputs_list in step_outputs
            for outputs in outputs_list
            if outputs_list is not None
        ]
        sampling_outputs_list = [
            {f"{self.val_phase}/{key}": value for key, value in outputs.items()}
            for outputs in sampling_outputs_list_
        ]

        # compile sampling metrics collected by the current device (e.g., rank zero)
        sampling_metrics_csv_path = os.path.join(
            self.sampling_output_dir,
            f"{self.val_phase}_epoch_{self.current_epoch}_step_{self.global_step}_rank_{self.global_rank}_sampling_metrics.csv",
        )
        sampling_metrics_df = pd.DataFrame(sampling_outputs_list)
        sampling_metrics_df.to_csv(sampling_metrics_csv_path, index=False)

        sampling_time = time.time() - self.val_epoch_start_time
        sampling_logs = {f"{self.val_phase}/sampling_time": sampling_time}
        for metric_name in metrics.ALL_COMPLEX_METRICS:
            sampling_logs[f"{self.val_phase}/{metric_name}"] = sampling_metrics_df[
                f"{self.val_phase}/{metric_name}"
            ].mean()

        if (
            getattr(self, "logger", None) is not None
            and getattr(self.logger, "experiment", None) is not None
        ):
            if "wandb" in type(self.logger).__name__.lower():
                # use WandB as our experiment logger
                wandb_run = self.logger.experiment

                sampling_table = wandb.Table(
                    columns=sampling_metrics_df.columns.to_list() + [f"{self.val_phase}/structure"]
                )
                for _, row in sampling_metrics_df.iterrows():
                    pdb_path = row[f"{self.val_phase}/sample_protein_na_pdb_path"]
                    row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                    sampling_table.add_data(*row_metrics)
                sampling_logs[f"{self.val_phase}/sample_metrics"] = sampling_table

                wandb_run.log(sampling_logs)
            elif (
                "mlflow" in type(self.logger).__name__.lower()
                and getattr(self.logger.experiment, "log_artifact", None) is not None
            ):
                # use MLFlow as our experiment logger
                self.logger.experiment.log_artifact(
                    run_id=self.logger.experiment.run_id, local_path=sampling_metrics_csv_path
                )

        # also log sampling metrics directly so they can be used e.g., for early stopping
        for metric_name in metrics.ALL_COMPLEX_METRICS:
            self.log(
                f"{self.val_phase}/{metric_name}",
                sampling_logs[f"{self.val_phase}/{metric_name}"],
                sync_dist=True,
            )

        log.info(f"validation_epoch_end(): Sampling evaluation took {sampling_time:.2f} second(s)")
        step_outputs.clear()

    def on_test_epoch_start(self):
        # track time to complete the testing epoch
        self.test_epoch_start_time = time.time()

        # ensure directory for storing sampling outputs is defined
        if not getattr(self, "sampling_output_dir", None):
            self.sampling_output_dir = Path(self.trainer.default_root_dir)

        # ensure our custom `Sampler` is up-to-date on what the current epoch is
        if self.trainer.datamodule.sampler_test is not None:
            self.trainer.datamodule.sampler_test.set_epoch(self.current_epoch)
            if hasattr(self.trainer.datamodule.sampler_test, "dataset") and hasattr(
                self.trainer.datamodule.sampler_test.dataset, "_sampler"
            ):
                # propagate epoch updates to the custom `TrainSampler`
                self.trainer.datamodule.sampler_test.dataset._sampler.set_epoch(self.current_epoch)

        # as requested, switch to using EMA weights before sampling from the model #
        ema_callback = EMAModelCheckpoint._get_ema_callback(self.trainer)
        evaluating_with_ema_weights = (
            ema_callback is not None
            and ema_callback.ema_initialized
            and ema_callback.evaluate_ema_weights_instead
        )
        if evaluating_with_ema_weights:
            # ensure EMA weights are subsequently used when sampling from the model
            ema_callback.replace_model_weights(self)

    @beartype
    def test_step(self, batch: List[Any], batch_idx: int):
        batch_ = batch[0]
        batch_["pdb_code"] = batch[1]
        batch_["pdb_filepath"] = batch[2]
        sampling_outputs_list = self.eval_sampling(
            batch=batch_,
            sequence_noise_scale=self.hparams.diffusion_cfg.sampling.sequence_noise_scale,
            structure_noise_scale=self.hparams.diffusion_cfg.sampling.structure_noise_scale,
            apply_na_consensus_sampling=self.hparams.diffusion_cfg.sampling.apply_na_consensus_sampling,
            force_na_seq_type=self.hparams.diffusion_cfg.sampling.force_na_seq_type,
        )
        # collect outputs
        step_outputs = getattr(self, f"{self.test_phase}_step_outputs")
        step_outputs.append(sampling_outputs_list)

    def on_test_epoch_end(self):
        # tag current sampling outputs according to the phase in which they were generated
        step_outputs = getattr(self, f"{self.test_phase}_step_outputs")
        sampling_outputs_list_ = [
            outputs
            for outputs_list in step_outputs
            for outputs in outputs_list
            if outputs_list is not None
        ]
        sampling_outputs_list = [
            {f"{self.test_phase}/{key}": value for key, value in outputs.items()}
            for outputs in sampling_outputs_list_
        ]

        # compile sampling metrics collected by the current device (e.g., rank zero)
        sampling_metrics_csv_path = os.path.join(
            self.sampling_output_dir,
            f"{self.test_phase}_epoch_{self.current_epoch}_step_{self.global_step}_rank_{self.global_rank}_sampling_metrics.csv",
        )
        sampling_metrics_df = pd.DataFrame(sampling_outputs_list)
        sampling_metrics_df.to_csv(sampling_metrics_csv_path, index=False)

        sampling_time = time.time() - self.test_epoch_start_time
        sampling_logs = {f"{self.test_phase}/sampling_time": sampling_time}
        for metric_name in metrics.ALL_COMPLEX_METRICS:
            sampling_logs[f"{self.test_phase}/{metric_name}"] = sampling_metrics_df[
                f"{self.test_phase}/{metric_name}"
            ].mean()

        if (
            getattr(self, "logger", None) is not None
            and getattr(self.logger, "experiment", None) is not None
        ):
            if "wandb" in type(self.logger).__name__.lower():
                # use WandB as our experiment logger
                wandb_run = self.logger.experiment

                sampling_table = wandb.Table(
                    columns=sampling_metrics_df.columns.to_list()
                    + [f"{self.test_phase}/structure"]
                )
                for _, row in sampling_metrics_df.iterrows():
                    pdb_path = row[f"{self.test_phase}/sample_protein_na_pdb_path"]
                    row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                    sampling_table.add_data(*row_metrics)
                sampling_logs[f"{self.test_phase}/sample_metrics"] = sampling_table

                wandb_run.log(sampling_logs)
            elif (
                "mlflow" in type(self.logger).__name__.lower()
                and getattr(self.logger.experiment, "log_artifact", None) is not None
            ):
                # use MLFlow as our experiment logger
                self.logger.experiment.log_artifact(
                    run_id=self.logger.experiment.run_id, local_path=sampling_metrics_csv_path
                )

        # also log sampling metrics directly so they can be used e.g., for early stopping
        for metric_name in metrics.ALL_COMPLEX_METRICS:
            self.log(
                f"{self.test_phase}/{metric_name}",
                sampling_logs[f"{self.test_phase}/{metric_name}"],
                sync_dist=True,
            )

        log.info(f"test_epoch_end(): Sampling evaluation took {sampling_time:.2f} second(s)")
        step_outputs.clear()

    def on_predict_epoch_start(self):
        if self.hparams.inference_cfg.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self.hparams.inference_cfg.name
        self.predict_output_dir = os.path.join(self.hparams.inference_cfg.output_dir, dt_string)
        # define where the final predictions should be recorded
        self.predictions_csv_path = os.path.join(
            self.predict_output_dir,
            f"rank_{self.global_rank}_predictions.csv",
        )

    @beartype
    def predict_step(self, sampling_args: SAMPLING_ARGS, batch_idx: int, dataloader_idx: int = 0):
        # parse and construct arguments
        sample_length = sampling_args["sample_length"].item()
        sample_length_index = sampling_args["sample_length_index"].item()
        length_dir = os.path.join(self.predict_output_dir, f"length_{sample_length}")
        sampling_output_dir = os.path.join(length_dir, f"sample_{sample_length_index}")
        sc_output_dir = os.path.join(sampling_output_dir, "self_consistency")
        sampling_interrupted = os.path.exists(sc_output_dir) and not os.path.exists(
            self.predictions_csv_path
        )
        if os.path.isdir(sampling_output_dir) and not sampling_interrupted:
            return
        os.makedirs(sampling_output_dir, exist_ok=True)
        pdb_path = None
        if not sampling_interrupted:
            # build `batch` input
            batch = self.construct_batch_from_sampling_args(sampling_args)
            batch = self.standardize_batch_features(batch)
            # allow for sequence masking
            batch["gt_node_types"] = batch["node_types"].clone()
            batch["gt_node_deoxy"] = batch["node_deoxy"].clone()
            batch["gt_node_types"][batch["is_na_residue_mask"]] = convert_na_aatype6_to_aatype9(
                aatype=batch["gt_node_types"][batch["is_na_residue_mask"]],
                deoxy_offset_mask=batch["node_deoxy"][batch["is_na_residue_mask"]],
                return_na_within_original_range=True,
            )
            # sample and save diffusion trajectory
            sampling_output = self.sample(
                batch,
                num_timesteps=self.hparams.inference_cfg.diffusion.num_t,
                min_timestep=self.hparams.inference_cfg.diffusion.min_t,
                aux_traj=True,
                sequence_noise_scale=self.hparams.inference_cfg.diffusion.sequence_noise_scale,
                structure_noise_scale=self.hparams.inference_cfg.diffusion.structure_noise_scale,
                apply_na_consensus_sampling=self.hparams.inference_cfg.diffusion.apply_na_consensus_sampling,
                employ_random_baseline=self.hparams.inference_cfg.diffusion.employ_random_baseline,
                force_na_seq_type=self.hparams.inference_cfg.diffusion.force_na_seq_type,
            )
            if sampling_output is None:
                return
            sampling_output = tree.map_structure(lambda x: x[:, 0], sampling_output)
            complex_restype = (
                sampling_output["seq_0_traj"]
                if self.diffuse_sequence
                else detach_tensor_to_np(batch["node_types"].squeeze(0))
            )
            traj_paths = self.save_traj(
                sampling_output["mol_traj"],
                sampling_output["rigid_0_traj"],
                np.ones(sample_length),
                output_dir=sampling_output_dir,
                complex_restype=complex_restype,
                residue_chain_indices=detach_tensor_to_np(batch["node_chain_indices"].squeeze(0)),
                is_protein_residue_mask=detach_tensor_to_np(
                    batch["is_protein_residue_mask"].squeeze(0)
                ),
                is_na_residue_mask=detach_tensor_to_np(batch["is_na_residue_mask"].squeeze(0)),
                # avoid saving large (e.g., 50MB) multi-state PDB trajectories e.g., when running full sampling evaluation
                save_multi_state_traj=(not self.hparams.inference_cfg.run_statified_eval),
            )
            # use RoseTTAFold2NA for scoring
            pdb_path = traj_paths["sample_path"]
        if self.diffuse_sequence:
            if not sampling_interrupted:
                os.makedirs(sc_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path)))
            design_results = self.eval_self_consistency(
                sc_output_dir,
                motif_mask=None,
                generate_protein_sequences_using_pmpnn=self.hparams.inference_cfg.generate_protein_sequences_using_pmpnn,
                measure_auxiliary_na_metrics=self.hparams.inference_cfg.measure_auxiliary_na_metrics,
            )
            # collect outputs
            step_outputs = getattr(self, f"{self.predict_phase}_step_outputs")
            step_outputs.append(design_results)
        log.info(f"Finished generating sample {sample_length_index}: {pdb_path}")

    @torch.inference_mode()
    @beartype
    def on_predict_epoch_end(self):
        step_outputs = getattr(self, f"{self.predict_phase}_step_outputs")
        outputs_dfs = [outputs_df for outputs_df in step_outputs if outputs_df is not None]
        if len(outputs_dfs):
            # compile final predictions and metadata collected by the current device (e.g., rank zero)
            combined_outputs_csv_df = pd.concat(outputs_dfs, axis=0)
            combined_outputs_csv_df.to_csv(self.predictions_csv_path, index=False)
        step_outputs.clear()

    @torch.inference_mode()
    @jaxtyped
    @beartype
    def construct_batch_from_sampling_args(self, sampling_args: SAMPLING_ARGS) -> BATCH_TYPE:
        # process sequence arguments
        sample_length = sampling_args["sample_length"].item()
        res_mask = np.ones(sample_length, dtype=np.float32)
        fixed_mask = np.zeros_like(res_mask)
        # initialize data
        ref_sample = self.ddpm.sample_ref(
            num_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = np.arange(1, sample_length + 1, dtype=np.int64)
        init_feats = {
            "residue_index": res_idx,
            "chain_idx": sampling_args["sample_chain_idx"].squeeze(0),
            "res_mask": res_mask,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros(
                (sample_length, complex_constants.NUM_PROT_NA_TORSIONS, 2), dtype=np.float32
            ),
            "sc_pos_t": np.zeros((sample_length, 3), dtype=np.float32),
            "sc_aatype_t": np.zeros(
                (sample_length, NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES), dtype=np.float32
            ),
            "molecule_type_encoding": sampling_args["sample_molecule_type_encoding"].squeeze(0),
            "is_protein_residue_mask": sampling_args["sample_is_protein_residue_mask"].squeeze(0),
            "is_na_residue_mask": sampling_args["sample_is_na_residue_mask"].squeeze(0),
            "asym_id": sampling_args["sample_asym_id"].squeeze(0),
            "sym_id": sampling_args["sample_sym_id"].squeeze(0),
            "entity_id": sampling_args["sample_entity_id"].squeeze(0),
            **ref_sample,
        }
        # NOTE: To perform masked sequence design, one currently needs to manually
        # modify `aatype` here to contain a ground-truth (starting) sequence
        # in `aatype6` format (for sequence regions of nucleic acid residues);
        # one would also need to update `fixed_mask` to reflect which residues
        # are to be fixed in sequence and structure
        init_feats["aatype"] = np.zeros_like(init_feats["residue_index"])
        init_feats["aatype"][
            detach_tensor_to_np(init_feats["is_protein_residue_mask"])
        ] = complex_constants.default_protein_restype
        init_feats["aatype"][
            detach_tensor_to_np(init_feats["is_na_residue_mask"])
        ] = complex_constants.default_na_restype
        init_feats["atom_deoxy"] = (
            detach_tensor_to_np(init_feats["is_na_residue_mask"])
            & (init_feats["aatype"] >= 21)
            & (init_feats["aatype"] <= 24)
        )
        # add batch dimension and move to current `device` (e.g., GPU, rank 0)
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)
        return init_feats

    @jaxtyped
    @beartype
    def enforce_na_consensus_sampling(
        self,
        min_na_logit_value: Union[float, torch.Tensor],
        batch: Optional[BATCH_TYPE] = None,
        seq_pred: Optional[torch.Tensor] = None,
        is_na_residue_mask: Optional[torch.Tensor] = None,
        consensus_threshold: float = 0.5,
        force_na_seq_type: Optional[str] = None,
    ) -> Union[BATCH_TYPE, torch.Tensor]:
        # e.g., apply a 50% majority rule that transforms all generated
        # nucleotide residue types to be exclusively of DNA or RNA types
        if batch is not None:
            node_types = batch["onehot_node_types"].argmax(-1)
            node_deoxy = batch["is_na_residue_mask"] & (node_types >= 21) & (node_types <= 24)
            node_deoxy_ratio = (node_deoxy.sum() / batch["is_na_residue_mask"].sum()).item()
            if (node_deoxy_ratio >= consensus_threshold or (force_na_seq_type is not None and force_na_seq_type == "DNA")) and not (force_na_seq_type is not None and force_na_seq_type == "RNA"):
                batch["onehot_node_types"][
                    ...,
                    NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 4 : NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 8,
                ][batch["is_na_residue_mask"]] = min_na_logit_value
            else:
                batch["onehot_node_types"][
                    ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES : NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 4
                ][batch["is_na_residue_mask"]] = min_na_logit_value
            # also prohibit the mask token from being selected
            batch["onehot_node_types"][..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 8][
                batch["is_na_residue_mask"]
            ] = min_na_logit_value
            return batch
        elif seq_pred is not None:
            assert (
                is_na_residue_mask is not None
            ), "For Tensor inputs, an NA residue mask must be given."
            node_types = seq_pred.argmax(-1)
            node_deoxy = is_na_residue_mask & (node_types >= 21) & (node_types <= 24)
            node_deoxy_ratio = (node_deoxy.sum() / is_na_residue_mask.sum()).item()
            if (node_deoxy_ratio >= consensus_threshold or (force_na_seq_type is not None and force_na_seq_type == "DNA")) and not (force_na_seq_type is not None and force_na_seq_type == "RNA"):
                seq_pred[
                    ...,
                    NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 4 : NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 8,
                ][is_na_residue_mask] = min_na_logit_value
            else:
                seq_pred[
                    ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES : NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 4
                ][is_na_residue_mask] = min_na_logit_value
            # also prohibit the mask token from being selected
            seq_pred[..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES + 8][
                is_na_residue_mask
            ] = min_na_logit_value
            return seq_pred

    @torch.inference_mode()
    @jaxtyped
    @beartype
    def sample(
        self,
        batch: BATCH_TYPE,
        num_timesteps: Optional[TIMESTEP_TYPE] = None,
        min_timestep: Optional[TIMESTEP_TYPE] = None,
        center: bool = True,
        aux_traj: bool = False,
        self_condition: bool = True,
        sequence_noise_scale: float = 1.0,
        structure_noise_scale: float = 1.0,
        apply_na_consensus_sampling: bool = False,
        employ_random_baseline: bool = False,
        force_na_seq_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        # prepare arguments for the reverse process
        sample_batch = copy.deepcopy(batch)
        device = sample_batch["rigids_t"].device
        fixed_node_mask = sample_batch["fixed_node_mask"] * sample_batch["node_mask"]
        diffuse_mask = (1 - sample_batch["fixed_node_mask"]) * sample_batch["node_mask"]
        is_protein_residue_mask = sample_batch["is_protein_residue_mask"].cpu()
        is_na_residue_mask = sample_batch["is_na_residue_mask"].cpu()
        protein_inputs_present = is_protein_residue_mask.any().item()
        na_inputs_present = is_na_residue_mask.any().item()
        if sample_batch["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,), device=device)
        else:
            t_placeholder = torch.ones((sample_batch["rigids_t"].shape[0],), device=device)
        if num_timesteps is None:
            num_timesteps = self.hparams.diffusion_cfg.num_timesteps
        if min_timestep is None:
            min_timestep = self.hparams.diffusion_cfg.min_timestep

        # prepare data structures for reverse process iterations
        dt = 1 / num_timesteps
        all_bb_mols = []
        all_mol_seqs = []
        all_bb_0_pred = []
        all_trans_0_pred = []
        reverse_steps = np.linspace(min_timestep, 1.0, num_timesteps)[::-1]
        all_rigids = [detach_tensor_to_np(copy.deepcopy(sample_batch["rigids_t"]))]

        if self.diffuse_sequence:
            sequence_feats = {
                "is_protein_residue_mask": sample_batch["is_protein_residue_mask"],
                "is_na_residue_mask": sample_batch["is_na_residue_mask"],
                "fixed_mask": sample_batch["fixed_node_mask"],
                # start by assuming each residue is a `mask`-type residue, for amino acid and nucleic acids, respectively
                "aatype": torch.full(
                    sample_batch["fixed_node_mask"].shape,
                    vocabulary.protein_restype_num,
                    device=sample_batch["fixed_node_mask"].device,
                ),
            }
            sequence_feats["aatype"][
                sample_batch["is_na_residue_mask"]
            ] = complex_constants.unknown_nucleic_token
            sequence_feats["aatype"] = (
                2
                * torch.nn.functional.one_hot(
                    sequence_feats["aatype"], num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES
                )
                - 1
            ).float()
            sample_batch["sc_aatype_t"] = torch.zeros_like(sequence_feats["aatype"])

        # perform reverse process
        with torch.no_grad():
            # initialize with self-conditioning embeddings as requested
            if self.hparams.model_cfg.embedding.embed_self_conditioning and self_condition:
                if self.diffuse_sequence:
                    diffused_sequence = diffuse_sequence(
                        sequence_feats=copy.deepcopy(sequence_feats),
                        t=reverse_steps[0],
                        min_t=min_timestep,
                        num_t=num_timesteps,
                        random_seed=None,  # enable random sequence generation
                        sequence_ddpm=self.sequence_ddpm,
                        training=False,
                        onehot_sequence_input=True,  # note: will preserve gradients to `aatype` input
                        noise_scale=sequence_noise_scale,
                    )[0]
                    sample_batch["node_types"] = diffused_sequence["diffused_aatype"]
                    sample_batch["onehot_node_types"] = diffused_sequence["diffused_onehot_aatype"]
                    sample_batch["node_deoxy"] = diffused_sequence["diffused_atom_deoxy"]
                    # fix motif within sequence
                    if all(field in sample_batch for field in ["gt_node_types", "gt_node_deoxy"]):
                        fixed_mask = sample_batch["fixed_node_mask"].bool()
                        sample_batch["node_types"][fixed_mask] = sample_batch["gt_node_types"][
                            fixed_mask
                        ]
                        sample_batch["onehot_node_types"][fixed_mask] = (
                            2
                            * torch.nn.functional.one_hot(
                                sample_batch["node_types"][fixed_mask],
                                num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
                            )
                            - 1
                        ).float()
                        sample_batch["node_deoxy"][fixed_mask] = sample_batch["gt_node_deoxy"][
                            fixed_mask
                        ]
                    # ensure that diffused protein sequences assign minimum likelihood to nucleic acid residues
                    if protein_inputs_present:
                        min_protein_onehot_node_types_logit = sample_batch["onehot_node_types"][
                            ..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
                        ][sample_batch["is_protein_residue_mask"]].min()
                        # also prohibit the mask token from being selected
                        sample_batch["onehot_node_types"][
                            ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES - 1
                        ][sample_batch["is_protein_residue_mask"]] = (
                            min_protein_onehot_node_types_logit - 0.1
                        )
                    # ensure that diffused nucleic acid sequences assign minimum likelihood to protein residues
                    if na_inputs_present:
                        min_na_onehot_node_types_logit = sample_batch["onehot_node_types"][
                            ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:
                        ][sample_batch["is_na_residue_mask"]].min()
                        sample_batch["onehot_node_types"][..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES][
                            sample_batch["is_na_residue_mask"]
                        ] = (min_na_onehot_node_types_logit - 0.1)
                        if apply_na_consensus_sampling:
                            sample_batch = self.enforce_na_consensus_sampling(
                                min_na_logit_value=(min_na_onehot_node_types_logit - 0.1),
                                batch=sample_batch,
                                force_na_seq_type=force_na_seq_type,
                            )
                    # update feature dependencies after guarding against out-of-vocabulary token assignments
                    sample_batch["node_types"] = sample_batch["onehot_node_types"].argmax(-1)
                    sample_batch["node_deoxy"] = (
                        sample_batch["is_na_residue_mask"]
                        & (sample_batch["node_types"] >= 21)
                        & (sample_batch["node_types"] <= 24)
                    )
                    sample_batch["node_types"][
                        sample_batch["is_na_residue_mask"]
                    ] = convert_na_aatype9_to_aatype6(
                        aatype=sample_batch["node_types"][sample_batch["is_na_residue_mask"]],
                        deoxy_offset_mask=sample_batch["node_deoxy"][
                            sample_batch["is_na_residue_mask"]
                        ],
                        return_na_within_original_range=True,
                    )
                # self-condition
                sample_batch = self.set_t_features_in_batch(
                    sample_batch, reverse_steps[0], t_placeholder
                )
                sample_batch = self.self_condition(sample_batch)

            # iteratively generate samples through progressive refinement
            for t in reverse_steps:
                if self.diffuse_sequence:
                    diffused_sequence = diffuse_sequence(
                        sequence_feats=sequence_feats,
                        t=t,
                        min_t=min_timestep,
                        num_t=num_timesteps,
                        random_seed=None,  # enable random sequence generation
                        sequence_ddpm=self.sequence_ddpm,
                        training=False,
                        onehot_sequence_input=True,  # note: will preserve gradients to `aatype` input
                        noise_scale=sequence_noise_scale,
                    )[0]
                    sample_batch["node_types"] = diffused_sequence["diffused_aatype"]
                    sample_batch["onehot_node_types"] = diffused_sequence["diffused_onehot_aatype"]
                    sample_batch["node_deoxy"] = diffused_sequence["diffused_atom_deoxy"]
                    # fix motif within sequence
                    if all(field in sample_batch for field in ["gt_node_types", "gt_node_deoxy"]):
                        fixed_mask = sample_batch["fixed_node_mask"].bool()
                        sample_batch["node_types"][fixed_mask] = sample_batch["gt_node_types"][
                            fixed_mask
                        ]
                        sample_batch["onehot_node_types"][fixed_mask] = (
                            2
                            * torch.nn.functional.one_hot(
                                sample_batch["node_types"][fixed_mask],
                                num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
                            )
                            - 1
                        ).float()
                        sample_batch["node_deoxy"][fixed_mask] = sample_batch["gt_node_deoxy"][
                            fixed_mask
                        ]
                    # ensure that diffused protein sequences assign minimum likelihood to nucleic acid residues
                    if protein_inputs_present:
                        min_protein_onehot_node_types_logit = sample_batch["onehot_node_types"][
                            ..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
                        ][is_protein_residue_mask].min()
                        # also prohibit the mask token from being selected
                        sample_batch["onehot_node_types"][
                            ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES - 1
                        ][is_protein_residue_mask] = (min_protein_onehot_node_types_logit - 0.1)
                    # ensure that diffused nucleic acid sequences assign minimum likelihood to protein residues
                    if na_inputs_present:
                        min_na_onehot_node_types_logit = sample_batch["onehot_node_types"][
                            ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:
                        ][is_na_residue_mask].min()
                        sample_batch["onehot_node_types"][..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES][
                            is_na_residue_mask
                        ] = (min_na_onehot_node_types_logit - 0.1)
                        if apply_na_consensus_sampling:
                            sample_batch = self.enforce_na_consensus_sampling(
                                min_na_logit_value=(min_na_onehot_node_types_logit - 0.1),
                                batch=sample_batch,
                                force_na_seq_type=force_na_seq_type,
                            )
                    # update feature dependencies after guarding against out-of-vocabulary token assignments
                    sample_batch["node_types"] = sample_batch["onehot_node_types"].argmax(-1)
                    sample_batch["node_deoxy"] = (
                        sample_batch["is_na_residue_mask"]
                        & (sample_batch["node_types"] >= 21)
                        & (sample_batch["node_types"] <= 24)
                    )
                    sample_batch["node_types"][
                        sample_batch["is_na_residue_mask"]
                    ] = convert_na_aatype9_to_aatype6(
                        aatype=sample_batch["node_types"][sample_batch["is_na_residue_mask"]],
                        deoxy_offset_mask=sample_batch["node_deoxy"][
                            sample_batch["is_na_residue_mask"]
                        ],
                        return_na_within_original_range=True,
                    )
                if t > min_timestep:
                    # perform iteration of diffusion reversal
                    sample_batch = self.set_t_features_in_batch(sample_batch, t, t_placeholder)
                    model_out = self.model(sample_batch)
                    rot_score = model_out["rot_score"]
                    trans_score = model_out["trans_score"]
                    rigid_pred = model_out["rigids"]
                    seq_pred = model_out.get("pred_node_types", None)
                    if self.hparams.model_cfg.embedding.embed_self_conditioning:
                        # condition the model's prediction on structural features derived from ones of its prior predictions
                        sample_batch["sc_pos_t"] = rigid_pred[..., 4:]
                    rigids_t = self.ddpm.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_batch["rigids_t"]),
                        rot_score=detach_tensor_to_np(rot_score),
                        trans_score=detach_tensor_to_np(trans_score),
                        t=t,
                        dt=dt,
                        diffuse_mask=detach_tensor_to_np(diffuse_mask),
                        center=center,
                        noise_scale=structure_noise_scale,
                    )
                else:
                    model_out = self.model(sample_batch)
                    rigids_t = ru.Rigid.from_tensor_7(model_out["rigids"].cpu())
                    seq_pred = model_out.get("pred_node_types", None)
                sample_batch["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(detach_tensor_to_np(rigids_t.to_tensor_7()))

                if self.diffuse_sequence:
                    # ensure that diffused protein sequences assign minimum likelihood to nucleic acid residues
                    if protein_inputs_present:
                        min_protein_seq_pred_logit = seq_pred[
                            ..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
                        ][is_protein_residue_mask].min()
                        seq_pred[..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:][
                            is_protein_residue_mask
                        ] = (min_protein_seq_pred_logit - 0.1)
                    # ensure that diffused nucleic acid sequences assign minimum likelihood to protein residues
                    if na_inputs_present:
                        min_na_seq_pred_logit = seq_pred[..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:][
                            is_na_residue_mask
                        ].min()
                        seq_pred[..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES][is_na_residue_mask] = (
                            min_na_seq_pred_logit - 0.1
                        )
                        if apply_na_consensus_sampling:
                            seq_pred = self.enforce_na_consensus_sampling(
                                min_na_logit_value=(min_na_seq_pred_logit - 0.1),
                                seq_pred=seq_pred,
                                is_na_residue_mask=is_na_residue_mask.to(device),
                                force_na_seq_type=force_na_seq_type,
                            )

                    all_mol_seqs.append(detach_tensor_to_np(seq_pred.argmax(-1)))

                    if (
                        t > min_timestep
                        and self.hparams.model_cfg.embedding.embed_self_conditioning
                    ):
                        # condition the model's prediction on sequence features derived from ones of its prior predictions
                        sample_batch["sc_aatype_t"] = seq_pred
                        sequence_feats["aatype"] = seq_pred

                # calculate `x0` prediction derived from score predictions
                is_pyrimidine_residue_mask = None
                pred_torsions = model_out["torsions"]
                if aux_traj:
                    gt_trans_0 = sample_batch["rigids_t"][..., 4:]
                    pred_trans_0 = rigid_pred[..., 4:]
                    trans_pred_0 = (
                        diffuse_mask[..., None] * pred_trans_0
                        + fixed_node_mask[..., None] * gt_trans_0
                    )
                    atom37_0 = all_atom.compute_backbone(
                        bb_rigids=ru.Rigid.from_tensor_7(rigid_pred),
                        torsions=pred_torsions,
                        is_protein_residue_mask=is_protein_residue_mask,
                        is_na_residue_mask=is_na_residue_mask,
                        aatype=seq_pred.argmax(-1) if seq_pred is not None else seq_pred,
                    )[0]
                    if self.diffuse_sequence:
                        # for pyrimidine NA residues, move their predicted N9 positions to the respective N1 index
                        is_pyrimidine_residue_mask = torch.isin(
                            elements=seq_pred.argmax(-1),
                            test_elements=torch.tensor(
                                complex_constants.PYRIMIDINE_RESIDUE_TOKENS, device=device
                            ),
                        )
                        atom37_0[..., 12, :][is_pyrimidine_residue_mask] = atom37_0[..., 18, :][
                            is_pyrimidine_residue_mask
                        ]
                        atom37_0[..., 18, :][is_pyrimidine_residue_mask] = 0.0
                    all_bb_0_pred.append(detach_tensor_to_np(atom37_0))
                    all_trans_0_pred.append(detach_tensor_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    bb_rigids=rigids_t,
                    torsions=pred_torsions.cpu(),
                    is_protein_residue_mask=is_protein_residue_mask,
                    is_na_residue_mask=is_na_residue_mask,
                    aatype=seq_pred.argmax(-1).cpu() if seq_pred is not None else seq_pred,
                )[0]
                if self.diffuse_sequence:
                    # for pyrimidine NA residues, move their predicted N9 positions to the respective N1 index
                    is_pyrimidine_residue_mask = (
                        is_pyrimidine_residue_mask.cpu()
                        if is_pyrimidine_residue_mask is not None
                        else torch.isin(
                            elements=seq_pred.argmax(-1),
                            test_elements=torch.tensor(
                                complex_constants.PYRIMIDINE_RESIDUE_TOKENS, device=device
                            ),
                        ).cpu()
                    )
                    atom37_t[..., 12, :][is_pyrimidine_residue_mask] = atom37_t[..., 18, :][
                        is_pyrimidine_residue_mask
                    ]
                    atom37_t[..., 18, :][is_pyrimidine_residue_mask] = 0.0
                all_bb_mols.append(detach_tensor_to_np(atom37_t))

                if employ_random_baseline:
                    # return initial noise representations as a baseline
                    all_bb_mols = flip_traj(all_bb_mols)
                    if aux_traj:
                        all_rigids = flip_traj(all_rigids)
                        all_trans_0_pred = flip_traj(all_trans_0_pred)
                        all_bb_0_pred = flip_traj(all_bb_0_pred)
                    if self.diffuse_sequence:
                        all_mol_seqs = flip_traj(all_mol_seqs)
                    ret = {"mol_traj": all_bb_mols}
                    if aux_traj:
                        ret["rigid_traj"] = all_rigids
                        ret["trans_traj"] = all_trans_0_pred
                        ret["pred_torsions"] = pred_torsions[None]
                        ret["rigid_0_traj"] = all_bb_0_pred
                    if self.diffuse_sequence:
                        ret["seq_traj"] = all_mol_seqs
                        if aux_traj:
                            # note: no need to duplicate the contents of `all_mol_seqs`
                            ret["seq_0_traj"] = all_mol_seqs
                    return ret

        # flip trajectory so that it starts from `t=0`, as this can help with visualizations and feature inspection
        all_bb_mols = flip_traj(all_bb_mols)
        if aux_traj:
            all_rigids = flip_traj(all_rigids)
            all_trans_0_pred = flip_traj(all_trans_0_pred)
            all_bb_0_pred = flip_traj(all_bb_0_pred)
        if self.diffuse_sequence:
            all_mol_seqs = flip_traj(all_mol_seqs)

        ret = {"mol_traj": all_bb_mols}
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["pred_torsions"] = pred_torsions[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        if self.diffuse_sequence:
            ret["seq_traj"] = all_mol_seqs
            if aux_traj:
                # note: no need to duplicate the contents of `all_mol_seqs`
                ret["seq_0_traj"] = all_mol_seqs

        return ret

    @beartype
    def save_traj(
        self,
        bb_mol_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        complex_restype: Optional[np.ndarray] = None,
        residue_chain_indices: Optional[np.ndarray] = None,
        is_protein_residue_mask: Optional[np.ndarray] = None,
        is_na_residue_mask: Optional[np.ndarray] = None,
        save_multi_state_traj: bool = True,
    ) -> Dict[str, str]:
        """Write final sample and reverse diffusion trajectory.

        Args:
            bb_mol_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                `T` is number of time steps. First time step is `t=eps`,
                i.e. `bb_mol_traj[0]` is the final sample after reverse diffusion.
                `N` is number of residues.
            x0_traj: [T, N, 3] `x_0` predictions of Ca/C4' at each time step.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.
            complex_restype: type of each residue in the diffusion
                trajectory provided, optional.
            residue_chain_indices: chain index of each residue
                in the diffusion trajectory provided, optional.
            is_protein_residue_mask: annotation of whether each residue
                belongs to a protein chain, optional.
            is_na_residue_mask: annotation of whether each residue
                belongs to a nucleic acid chain, optional.
            save_multi_state_traj: whether to save potentially-large
                multi-state PDB trajectories, optional.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file of all intermediate diffused states.
                'x0_traj_path': PDB file of Ca `x_0` predictions at each state.
            note: `b_factors` are set to `100` for diffused residues and `0` for motif
            residues if there are any.
        """

        # write sample
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, "sample")
        mol_traj_path = os.path.join(output_dir, "bb_traj")
        x0_traj_path = os.path.join(output_dir, "x0_traj")

        # use b-factors to specify which residues are diffused
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        if complex_restype is not None:
            sample_path_restype = (
                complex_restype[0] if complex_restype.ndim == 2 else complex_restype
            )
            assert (
                sample_path_restype.ndim == 1
            ), "When writing a single complex to a PDB file, only a single sequence may be provided."
        else:
            sample_path_restype = complex_restype
        sample_path = au.write_complex_to_pdbs(
            complex_pos=bb_mol_traj[0],
            output_filepath=sample_path,
            restype=sample_path_restype,
            chain_index=residue_chain_indices,
            b_factors=b_factors,
            is_protein_residue_mask=is_protein_residue_mask,
            is_na_residue_mask=is_na_residue_mask,
        )[-1]
        if save_multi_state_traj:
            mol_traj_path = au.write_complex_to_pdbs(
                complex_pos=bb_mol_traj,
                output_filepath=mol_traj_path,
                restype=complex_restype,
                chain_index=residue_chain_indices,
                b_factors=b_factors,
                is_protein_residue_mask=is_protein_residue_mask,
                is_na_residue_mask=is_na_residue_mask,
            )[-1]
            x0_traj_path = au.write_complex_to_pdbs(
                complex_pos=x0_traj,
                output_filepath=x0_traj_path,
                restype=complex_restype,
                chain_index=residue_chain_indices,
                b_factors=b_factors,
                is_protein_residue_mask=is_protein_residue_mask,
                is_na_residue_mask=is_na_residue_mask,
            )[-1]
        return {
            "sample_path": sample_path,
            "traj_path": mol_traj_path,
            "x0_traj_path": x0_traj_path,
        }

    @torch.inference_mode()
    @jaxtyped
    @beartype
    def eval_sampling(
        self,
        batch: BATCH_TYPE,
        num_timesteps: Optional[TIMESTEP_TYPE] = None,
        min_timestep: Optional[TIMESTEP_TYPE] = None,
        center: bool = True,
        aux_traj: bool = False,
        self_condition: bool = True,
        sequence_noise_scale: float = 1.0,
        structure_noise_scale: float = 1.0,
        apply_na_consensus_sampling: bool = False,
        force_na_seq_type: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        # collect inputs
        batch = self.standardize_batch_features(batch)
        node_mask = detach_tensor_to_np(batch["node_mask"].bool())
        fixed_node_mask = detach_tensor_to_np(batch["fixed_node_mask"].bool())
        gt_pos = detach_tensor_to_np(batch["atom37_pos"])
        is_protein_residue_batch_mask = detach_tensor_to_np(batch["is_protein_residue_mask"])
        is_na_residue_batch_mask = detach_tensor_to_np(batch["is_na_residue_mask"])
        batch_size = node_mask.shape[0]

        batch["gt_node_types"] = batch["node_types"].clone()
        batch["gt_node_deoxy"] = batch["node_deoxy"].clone() if "node_deoxy" in batch else None
        batch["gt_node_types"][batch["is_na_residue_mask"]] = convert_na_aatype6_to_aatype9(
            aatype=batch["gt_node_types"][batch["is_na_residue_mask"]],
            deoxy_offset_mask=batch["node_deoxy"][batch["is_na_residue_mask"]],
            return_na_within_original_range=True,
        )
        gt_node_types = detach_tensor_to_np(batch["gt_node_types"])

        # perform sampling
        sampling_output = self.sample(
            batch,
            num_timesteps=num_timesteps,
            min_timestep=min_timestep,
            center=center,
            aux_traj=aux_traj,
            self_condition=self_condition,
            sequence_noise_scale=sequence_noise_scale,
            structure_noise_scale=structure_noise_scale,
            apply_na_consensus_sampling=apply_na_consensus_sampling,
            force_na_seq_type=force_na_seq_type,
        )
        if sampling_output is None:
            return
        final_pred_pos = sampling_output["mol_traj"][0]
        final_pred_seq = sampling_output["seq_traj"][0] if self.diffuse_sequence else None

        # unravel and process batch elements
        sample_metrics_list = []
        for i in range(batch_size):
            gt_pdb_filepath = batch["pdb_filepath"][i]
            num_nodes = int(np.sum(node_mask[i]).item())
            unpad_fixed_mask = fixed_node_mask[i][node_mask[i]]
            unpad_diffuse_mask = 1 - unpad_fixed_mask
            unpad_pred_pos = final_pred_pos[i][node_mask[i]]
            unpad_gt_pos = gt_pos[i][node_mask[i]]
            unpad_node_chain_indices = batch["node_chain_indices"][i][node_mask[i]].cpu().numpy()
            percent_diffused = np.sum(unpad_diffuse_mask) / num_nodes
            b_factors = np.tile(1 - unpad_fixed_mask[..., None], 37) * 100
            unpad_is_protein_residue_mask = is_protein_residue_batch_mask[i][node_mask[i]]
            unpad_is_na_residue_mask = is_na_residue_batch_mask[i][node_mask[i]]
            unpad_gt_aatype = gt_node_types[i][node_mask[i]]
            # ensure residue indices for proteins are in range `[0, 20]` and indices for nucleic acids are in range `[21, 29]`
            if (
                len(unpad_gt_aatype[unpad_is_na_residue_mask]) > 0
                and np.max(unpad_gt_aatype[unpad_is_na_residue_mask])
                < complex_constants.na_restype_num
            ):
                unpad_gt_aatype[unpad_is_na_residue_mask] += vocabulary.protein_restype_num + 1
            if self.diffuse_sequence:
                unpad_pred_aatype = final_pred_seq[i][node_mask[i]]
                unpad_complex_restype = copy.deepcopy(unpad_pred_aatype)
            else:
                unpad_pred_aatype = None
                # set residue indices to baseline values (for structure-based protein-nucleic acid design)
                unpad_complex_restype = copy.deepcopy(unpad_gt_aatype)
                unpad_complex_restype[
                    unpad_is_protein_residue_mask
                ] = 0  # denote the default amino acid type (i.e., Alanine)
                unpad_complex_restype[
                    unpad_is_na_residue_mask
                ] = 21  # denote the default nucleic acid type (i.e., Adenine)

            # write out position predictions as PDB files specific to each molecule type
            (
                saved_protein_pdb_path,
                saved_na_pdb_path,
                saved_protein_na_pdb_path,
            ) = au.write_complex_to_pdbs(
                complex_pos=unpad_pred_pos,
                output_filepath=os.path.join(
                    self.sampling_output_dir,
                    f"len_{num_nodes}_sample_{i}_diffused_{percent_diffused:.2f}.pdb",
                ),
                restype=unpad_complex_restype,
                chain_index=unpad_node_chain_indices,
                b_factors=b_factors,
                is_protein_residue_mask=unpad_is_protein_residue_mask,
                is_na_residue_mask=unpad_is_na_residue_mask,
                no_indexing=True,
            )
            try:
                sample_metrics_list.append(
                    metrics.complex_metrics(
                        protein_pdb_filepath=saved_protein_pdb_path,
                        na_pdb_filepath=saved_na_pdb_path,
                        protein_na_pdb_filepath=saved_protein_na_pdb_path,
                        atom37_pos=unpad_pred_pos,
                        gt_atom37_pos=unpad_gt_pos,
                        gt_aatype=unpad_gt_aatype,
                        diffuse_mask=unpad_diffuse_mask,
                        is_protein_residue_mask=unpad_is_protein_residue_mask,
                        is_na_residue_mask=unpad_is_na_residue_mask,
                        pred_aatype=unpad_pred_aatype,
                    )
                )
            except ValueError as e:
                log.warning(
                    f"Failed sampling evaluation of batch sample {i} with length {num_nodes} due to: {e}"
                )
                continue
            sample_metrics_list[-1]["global_step"] = self.trainer.global_step
            sample_metrics_list[-1]["num_nodes"] = num_nodes
            sample_metrics_list[-1]["fixed_nodes"] = np.sum(unpad_fixed_mask)
            sample_metrics_list[-1]["diffused_percentage"] = percent_diffused
            sample_metrics_list[-1]["sample_protein_pdb_path"] = saved_protein_pdb_path
            sample_metrics_list[-1]["sample_na_pdb_path"] = saved_na_pdb_path
            sample_metrics_list[-1]["sample_protein_na_pdb_path"] = saved_protein_na_pdb_path
            sample_metrics_list[-1]["gt_pdb_filepath"] = gt_pdb_filepath
            if self.diffuse_sequence:
                sample_metrics_list[-1]["pred_aatype"] = [
                    complex_constants.restypes_1[idx] for idx in unpad_pred_aatype.tolist()
                ]
                sample_metrics_list[-1]["gt_aatype"] = [
                    complex_constants.restypes_1[idx] for idx in unpad_gt_aatype.tolist()
                ]

        return sample_metrics_list

    @beartype
    def run_proteinmpnn(
        self,
        decoy_pdb_dir: str,
        sequences_to_generate_per_sample: int,
    ) -> Dict[str, Any]:
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen(
            # nosec
            [
                "python",
                f"{self.hparams.path_cfg.proteinmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                f"--input_path={decoy_pdb_dir}",
                f"--output_path={output_path}",
            ]
        )
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self.hparams.path_cfg.proteinmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            decoy_pdb_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            str(sequences_to_generate_per_sample),
            "--sampling_temp",
            "1e-9",  # note: we employ a null temperature specifically when using PMPNN as a baseline
            "--seed",
            "38",
            "--batch_size",
            "1",
        ]
        pmpnn_args.append("--device")
        pmpnn_args.append(str(self.device))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT  # nosec
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                log.warning(
                    f"ProteinMPNN has failed to execute with provided inputs. Attempt {num_tries}/5"
                )
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e

        @beartype
        def parse_and_save_chain_sequences_separately(
            input_fasta_filepath: str,
            output_dir: str,
        ) -> Dict[str, Any]:
            with open(input_fasta_filepath) as fasta_file:
                fasta_contents = fasta_file.read()
            lines = fasta_contents.split("\n")
            pmpnn_chain_seq_dict = {}
            for line_index, line in enumerate(lines):
                if line.startswith(">sample_"):
                    chain_ids = line.split("designed_chains=['")[1].split("']")[0].split("', '")
                elif line.startswith(">T="):
                    seq_recovery = float(line.split(",")[-1].strip().split("=")[-1])
                    sequence_data = lines[line_index + 1].split("/")
            for i, chain_id in enumerate(chain_ids):
                pmpnn_sequence = sequence_data[i].replace("/", "")
                filepath = os.path.join(
                    output_dir,
                    "seqs",
                    f"P:{Path(os.path.basename(input_fasta_filepath)).stem}_chain_{chain_id}.fa",
                )
                with open(filepath, "w") as file:
                    file.write(f">{chain_id}\n{pmpnn_sequence}")
                pmpnn_chain_seq_dict[chain_id] = (pmpnn_sequence, filepath, seq_recovery)
            return pmpnn_chain_seq_dict

        pmpnn_out_fasta_path = os.path.join(decoy_pdb_dir, "seqs", "sample_1.fa")
        pmpnn_chain_seq_dict = parse_and_save_chain_sequences_separately(
            pmpnn_out_fasta_path, decoy_pdb_dir
        )
        os.remove(pmpnn_out_fasta_path)  # remove redundant file
        return pmpnn_chain_seq_dict

    @beartype
    def eval_self_consistency(
        self,
        decoy_pdb_dir: str,
        motif_mask: Optional[np.ndarray] = None,
        generate_protein_sequences_using_pmpnn: bool = False,
        measure_auxiliary_na_metrics: bool = False,
    ) -> pd.DataFrame:
        """Run self-consistency on designed macromolecules against reference macromolecule.

        Args:
            decoy_pdb_dir: directory where designed macromolecule files are stored.
            motif_mask: optional mask denoting which residues are the motif.
            generate_protein_sequences_using_pmpnn: whether to use ProteinMPNN as a sequence
                generation baseline for protein-only structure -> sequence generation.
            measure_auxiliary_na_metrics: whether to collect additional structural metrics
                for generated nucleic acids. Should only be used for nucleic acid-only generation.

        Returns:
            writes designed sequence outputs to decoy_pdb_dir/seqs
            writes RoseTTAFold2NA outputs to decoy_pdb_dir/rf2na
            writes results in decoy_pdb_dir/sc_results.csv
            returns a Pandas DataFrame containing the contents of `sc_results.csv`
        """
        # run RoseTTAFold2NA on each generated sequence and calculate metrics
        log.info(
            f"Evaluating self-consistency of the current batch of samples within {decoy_pdb_dir}"
        )
        design_results = {
            "rmsd": [],
            "tm_score": [],
            "initial_sample_path": [],
            "predicted_sample_path": [],
            "fasta_path": [],
            "sequence": [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            design_results["motif_rmsd"] = []
        decoy_seqs_dir = os.path.join(decoy_pdb_dir, "seqs")
        prediction_output_dir = os.path.join(decoy_pdb_dir, "rf2na")
        os.makedirs(decoy_seqs_dir, exist_ok=True)
        os.makedirs(prediction_output_dir, exist_ok=True)
        for decoy_pdb_filepath in glob.glob(os.path.join(decoy_pdb_dir, "*.pdb")):
            decoy_chain_id_molecule_type_mappings = du.pdb_to_fasta(
                input_pdb_filepath=decoy_pdb_filepath,
                output_dir=decoy_seqs_dir,
                output_id=Path(decoy_pdb_filepath).stem,
                use_rf2na_fasta_naming_scheme=True,
            )
            if generate_protein_sequences_using_pmpnn:
                pmpnn_chain_seq_dict = self.run_proteinmpnn(
                    decoy_pdb_dir,
                    sequences_to_generate_per_sample=1,
                )
                for chain_id in pmpnn_chain_seq_dict.keys():
                    # update FASTA sequences and filepaths in-place
                    decoy_chain_id_molecule_type_mappings[chain_id][
                        "sequence"
                    ] = pmpnn_chain_seq_dict[chain_id][0]
                    decoy_chain_id_molecule_type_mappings[chain_id][
                        "fasta_path"
                    ] = pmpnn_chain_seq_dict[chain_id][1]
                    pmpnn_seq_recovery = pmpnn_chain_seq_dict[chain_id][2]
            # assume we are not running RoseTTAFold2NA by default
            predicted_sample_path = decoy_pdb_filepath
            if self.run_self_consistency_eval:
                # run RoseTTAFold2NA
                predicted_sample_path = os.path.join(
                    prediction_output_dir, "models", "model_00.pdb"
                )  # select the first RF2NA model
                fasta_filepaths = glob.glob(os.path.join(decoy_seqs_dir, "*.fasta"))
                if not os.path.exists(predicted_sample_path):
                    metrics.run_rf2na(
                        output_dir=prediction_output_dir,
                        fasta_filepaths=fasta_filepaths,
                        rf2na_exec_path=self.hparams.path_cfg.rf2na_exec_path,
                        use_single_sequence_mode=self.hparams.inference_cfg.use_rf2na_single_sequence_mode,
                    )
                if not os.path.exists(predicted_sample_path):
                    raise Exception(
                        f"RoseTTAFold2NA failed to generate the structure {predicted_sample_path} for the FASTA inputs {fasta_filepaths}."
                    )
                rf2na_feats = du.parse_pdb_feats(
                    "folded_sample", predicted_sample_path, return_na_aatype9=True
                )
                sample_feats = du.parse_pdb_feats(
                    "sample", decoy_pdb_filepath, return_na_aatype9=True
                )
                # calculate scTM and scRMSD of RoseTTAFold2NA outputs with reference macromolecule, using US-align
                usalign_metrics = metrics.calculate_usalign_metrics(
                    pred_pdb_filepath=predicted_sample_path,
                    reference_pdb_filepath=decoy_pdb_filepath,
                    usalign_exec_path=self.hparams.path_cfg.usalign_exec_path,
                    flags=[
                        "-ter",
                        "0",  # note: biological unit alignment
                        "-TMscore",
                        "7",  # note: sequence-dependent alignment
                    ],
                )
                tm_score = usalign_metrics["TM-score_2"]
                rmsd = usalign_metrics["RMSD"]
                if motif_mask is not None:
                    sample_motif = np.concatenate(
                        [
                            sample_feats[chain_id]["bb_positions"][motif_mask]
                            for chain_id in sample_feats
                        ],
                        axis=0,
                    )
                    of_motif = np.concatenate(
                        [
                            rf2na_feats[chain_id]["bb_positions"][motif_mask]
                            for chain_id in rf2na_feats
                        ],
                        axis=0,
                    )
                    motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                    design_results["motif_rmsd"].append(motif_rmsd)
                design_results["rmsd"].append(rmsd)
                design_results["tm_score"].append(tm_score)
            design_results["initial_sample_path"].append(decoy_pdb_filepath)
            design_results["predicted_sample_path"].append(predicted_sample_path)
            design_results["fasta_path"].append(
                ",".join(
                    [
                        decoy_chain_id_molecule_type_mappings[chain]["fasta_path"]
                        for chain in decoy_chain_id_molecule_type_mappings
                    ]
                )
            )
            design_results["sequence"].append(
                ",".join(
                    [
                        decoy_chain_id_molecule_type_mappings[chain]["sequence"]
                        for chain in decoy_chain_id_molecule_type_mappings
                    ]
                )
            )
            design_results["pmpnn_seq_recovery"] = (
                pmpnn_seq_recovery if generate_protein_sequences_using_pmpnn else np.nan
            )
            if not self.run_self_consistency_eval:
                design_results["rmsd"] = np.nan
                design_results["tm_score"] = np.nan
            if self.run_self_consistency_eval and measure_auxiliary_na_metrics:
                # find all base pairs
                try:
                    helix = load_structure(predicted_sample_path)
                    basepairs = base_pairs(helix)
                except Exception as e:
                    basepairs = []
                    log.warning(
                        f"Skipped base pairs calculation for sample {predicted_sample_path} due to: {e}"
                    )
                design_results["num_rf2na_base_pairs"] = (
                    len(helix[basepairs].res_name) if len(basepairs) else np.nan
                )
                # determine radius of gyration
                try:
                    traj = md.load(decoy_pdb_filepath)
                except Exception as e:
                    traj = None
                try:
                    # DG calculation
                    pdb_rg = md.compute_rg(traj) if traj is not None else None
                except Exception as e:
                    pdb_rg = None
                design_results["radius_gyration"] = pdb_rg[0] if pdb_rg is not None else np.nan
            else:
                design_results["num_rf2na_base_pairs"] = np.nan
                design_results["radius_gyration"] = np.nan
        # save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, "sc_results.csv")
        design_results = pd.DataFrame(design_results)
        design_results.to_csv(csv_path)
        log.info(
            f"Finished evaluating self-consistency of the current batch of samples within {decoy_pdb_dir}"
        )
        return design_results

    @staticmethod
    @rank_zero_only
    @beartype
    def eval_diversity(samples_df: pd.DataFrame, qtmclust_exec_path: str) -> pd.DataFrame:
        log.info("Evaluating diversity of the generated samples...")
        temp_dir = tempfile.mkdtemp(prefix="diversity_temp_dir_")
        chain_list_filepath = os.path.join(temp_dir, "chain_list")
        num_chains, num_complexes = 0, len(samples_df)
        with open(chain_list_filepath, "w") as f:
            for sample_row_index, sample_row in samples_df.iterrows():
                # copy each PDB sample file to a shared temporary directory
                src_filepath = sample_row["initial_sample_path"]
                src_filepath_parts = src_filepath.split("/")
                src_filename = f"{src_filepath_parts[-4]}_{src_filepath_parts[-3]}_{src_filepath_parts[-2]}_{src_filepath_parts[-1]}"
                dst_filepath = os.path.join(temp_dir, src_filename)
                shutil.copy(src_filepath, dst_filepath)
                # record the name of each PDB file in a temporary text file input
                sample_name_without_extension = os.path.basename(os.path.splitext(dst_filepath)[0])
                sample_name_postfix = "" if sample_row_index == (len(samples_df) - 1) else "\n"
                f.write(f"{sample_name_without_extension}{sample_name_postfix}")
                # keep track of the total number of chains across all generated complexes
                num_chains += len(du.get_unique_chain_names(dst_filepath))
        output_cluster_filepath = os.path.join(temp_dir, "cluster.txt")
        # run qTMclust on all generated complex structures and parse results
        chain_output_clusters_df = metrics.run_qtmclust(
            chain_dir=temp_dir,
            chain_list_filepath=chain_list_filepath,
            qtmclust_exec_path=qtmclust_exec_path,
            output_cluster_filepath=output_cluster_filepath,
            tm_cluster_threshold=0.5,  # note: clusters two chains if their TM-score is 0.5 or greater
            chain_ter_mode=0,  # note: reads all chains
            chain_split_mode=2,  # note: parses all chains separately
        )
        complex_output_clusters_df = metrics.run_qtmclust(
            chain_dir=temp_dir,
            chain_list_filepath=chain_list_filepath,
            qtmclust_exec_path=qtmclust_exec_path,
            output_cluster_filepath=output_cluster_filepath,
            tm_cluster_threshold=0.5,  # note: clusters two chains if their TM-score is 0.5 or greater
            chain_ter_mode=0,  # note: reads all chains
            chain_split_mode=0,  # note: parses all chains as a single chain
        )
        # calculate chain diversity among generated structures (i.e., `num_chain_clusters / num_chains`)
        samples_df["chain_diversity"] = len(chain_output_clusters_df) / num_chains
        # calculate single-chain complex diversity among generated structures (i.e., `num_complexes_for_which_one_of_its_chains_is_a_chain_cluster_representative / num_complexes`)
        single_chain_cluster_reps = chain_output_clusters_df.iloc[:, 0].str.split(
            ":", expand=True
        )[0]
        samples_df["single_chain_complex_diversity"] = (
            single_chain_cluster_reps.nunique() / num_complexes
        )
        # calculate all-chain complex diversity among generated structures (i.e., `num_complexes_for_which_each_of_its_chains_is_a_chain_cluster_representative / num_complexes`)
        all_chain_cluster_reps = set()
        for _, cluster_row in chain_output_clusters_df.iterrows():
            cluster_name = cluster_row[0].split(":")[0]
            if not any(
                chain_output_clusters_df.iloc[:, 1:].apply(
                    lambda x: cluster_name in x.values, axis=1
                )
            ):
                all_chain_cluster_reps.add(cluster_name)
        samples_df["all_chain_complex_diversity"] = len(all_chain_cluster_reps) / num_complexes
        # calculate complex diversity among generated structures (i.e., `num_complex_clusters / num_complexes`)
        samples_df["complex_diversity"] = len(complex_output_clusters_df) / num_complexes
        log.info("Finished evaluating diversity of the generated samples")
        return samples_df

    @staticmethod
    @rank_zero_only
    @beartype
    def eval_novelty(
        samples_df: pd.DataFrame,
        train_data_df: pd.DataFrame,
        usalign_exec_path: str,
        num_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        # compute US-align's TM-score between each sample and each training example,
        # and record each sample's maximum TM-score example to assess the sample's overall novelty
        log.info("Evaluating novelty of each generated sample...")

        def calculate_trainTM(sample_row):
            # define a `Pandarallel`-compatible function to determine the maximum training example TM-score for each sample
            for _, train_row in train_data_df.iterrows():
                try:
                    usalign_metrics = metrics.calculate_usalign_metrics(
                        pred_pdb_filepath=sample_row["initial_sample_path"],
                        reference_pdb_filepath=train_row["raw_path"],
                        usalign_exec_path=usalign_exec_path,
                        flags=[
                            "-mm",
                            "1",  # note: sequence-independent alignment
                            "-ter",
                            "0",  # note: biological unit alignment
                            "-split",
                            "0",  # note: combined-chain alignment - to handle comparisons between single-chain CIF files in the training dataset
                        ],
                    )
                    tm_score = usalign_metrics["TM-score_2"]
                    if "trainTM" not in sample_row:
                        sample_row["trainTM"] = tm_score
                        sample_row["trainTM_example_path"] = train_row["raw_path"]
                    if tm_score > sample_row["trainTM"]:
                        sample_row["trainTM"] = tm_score
                        sample_row["trainTM_example_path"] = train_row["raw_path"]
                except subprocess.CalledProcessError as e:
                    log.warning(
                        f"In `eval_novelty()`, unable to computer US-align metrics due to: {e.stdout}"
                    )
                    sample_row["trainTM"] = np.nan
                    sample_row["trainTM_example_path"] = ""
                    continue
                except Exception as e:
                    log.warning(
                        f"In `eval_novelty()`, unable to computer US-align metrics due to: {str(e)}"
                    )
                    sample_row["trainTM"] = np.nan
                    sample_row["trainTM_example_path"] = ""
                    continue
            return (
                sample_row["trainTM"],
                sample_row["trainTM_example_path"],
                sample_row["initial_sample_path"],
            )

        if num_workers is not None:
            pandarallel.initialize(nb_workers=num_workers, progress_bar=False, verbose=2)
        else:
            # otherwise, use all available CPU threads for processing
            pandarallel.initialize(progress_bar=False, verbose=2)
        trainTM_outputs = list(samples_df.parallel_apply(calculate_trainTM, axis=1))
        trainTM_df = pd.DataFrame(
            trainTM_outputs, columns=["trainTM", "trainTM_example_path", "initial_sample_path"]
        )
        samples_df = samples_df.merge(trainTM_df, on="initial_sample_path", how="left")
        log.info("Finished evaluating novelty of each generated sample")
        return samples_df

    def on_after_backward(self):
        # periodically log gradient flow
        time_to_log_grad_flow = (
            self.trainer.is_global_zero
            and (self.global_step + 1) % self.hparams.model_cfg.log_grad_flow_steps == 0
        )
        logger_type = ""
        if (
            getattr(self, "logger", None) is not None
            and getattr(self.logger, "experiment", None) is not None
        ):
            if "wandb" in type(self.logger).__name__.lower():
                logger_type = "wandb"
            elif (
                "mlflow" in type(self.logger).__name__.lower()
                and getattr(self.logger.experiment, "log_artifact", None) is not None
            ):
                logger_type = "mlflow"

        if time_to_log_grad_flow and logger_type:
            if logger_type in ["wandb"]:
                log_grad_flow_lite_wandb(
                    self.named_parameters(),
                    wandb_run=self.logger.experiment,
                )
            elif logger_type in ["mlflow"]:
                log_grad_flow_lite_mlflow(
                    self.named_parameters(),
                    experiment=self.logger.experiment,
                    global_step=self.global_step,
                    current_epoch=self.current_epoch,
                    logger_type=logger_type,
                )

    @beartype
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/na_num_c4_prime_steric_clashes",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @beartype
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        verbose: bool = False,
    ):
        if not self.hparams.model_cfg.clip_gradients:
            return

        # allow gradient norm to be 150% + 2 * stdev of recent gradient history
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()

        # get current `grad_norm`
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params, device=self.device)

        # note: Lightning will then handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if verbose:
            log.info(f"Current gradient norm: {grad_norm}")

        if float(grad_norm) > max_grad_norm:
            log.info(
                f"Clipped gradient with value {grad_norm:.1f}, since the maximum value currently allowed is {max_grad_norm:.1f}"
            )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "pdb_prot_na_gen_se3_module.yaml")
    _ = hydra.utils.instantiate(cfg)
