# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Metrics."""
import subprocess  # nosec

import mdtraj as md
import numpy as np
import pandas as pd
import tree
from beartype import beartype
from beartype.typing import Any, Dict, List, Literal, Optional
from tmtools import tm_align

from src.data.components.pdb import data_utils as du
from src.data.components.pdb import nucleotide_constants, protein_constants
from src.data.components.pdb.relax import amber_minimize
from src.models.components.pdb import analysis_utils as au
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


DEFAULT_MISSING_VALUE = np.nan

PROTEIN_CA_IDX = protein_constants.atom_order["CA"]
NA_C4_PRIME_IDX = nucleotide_constants.atom_order["C4'"]

# protein metrics #
PROTEIN_SHAPE_METRICS = [
    "protein_coil_percent",
    "protein_helix_percent",
    "protein_strand_percent",
    "protein_radius_of_gyration",
]
PROTEIN_INTER_VIOLATION_METRICS = [
    "protein_bonds_c_n_loss_mean",
    "protein_angles_ca_c_n_loss_mean",
    "protein_clashes_mean_loss",
]
PROTEIN_CA_VIOLATION_METRICS = [
    "protein_ca_ca_bond_dev",
    "protein_ca_ca_valid_percent",
    "protein_ca_steric_clash_percent",
    "protein_num_ca_steric_clashes",
]
PROTEIN_EVAL_METRICS = [
    "protein_tm_score",
]
ALL_PROTEIN_METRICS = (
    PROTEIN_INTER_VIOLATION_METRICS
    + PROTEIN_SHAPE_METRICS
    + PROTEIN_CA_VIOLATION_METRICS
    + PROTEIN_EVAL_METRICS
)

# nucleic acid (NA) metrics #
NA_SHAPE_METRICS = [
    "na_coil_percent",
    "na_helix_percent",
    "na_strand_percent",
    "na_radius_of_gyration",
]
NA_C4_PRIME_VIOLATION_METRICS = [
    "na_c4_prime_c4_prime_bond_dev",
    "na_c4_prime_c4_prime_valid_percent",
    "na_c4_prime_steric_clash_percent",
    "na_num_c4_prime_steric_clashes",
]
ALL_NA_METRICS = NA_SHAPE_METRICS + NA_C4_PRIME_VIOLATION_METRICS

# protein-NA metrics #
PROTEIN_NA_SHAPE_METRICS = [
    "protein_na_coil_percent",
    "protein_na_helix_percent",
    "protein_na_strand_percent",
    "protein_na_radius_of_gyration",
]
PROTEIN_NA_SEQUENCE_METRICS = [
    "protein_na_sequence_recovery",
]
ALL_PROTEIN_NA_METRICS = PROTEIN_NA_SHAPE_METRICS
ALL_COMPLEX_METRICS = ALL_PROTEIN_METRICS + ALL_NA_METRICS + ALL_PROTEIN_NA_METRICS


@beartype
def run_rf2na(
    output_dir: str,
    fasta_filepaths: List[str],
    rf2na_exec_path: str,
    use_single_sequence_mode: bool = False,
    verbose: bool = True,
):
    """Run RoseTTAFold2NA on a list of FASTA chain filepaths."""
    single_sequence_mode = 1 if use_single_sequence_mode else 0
    cmd = [rf2na_exec_path, output_dir, str(single_sequence_mode), *fasta_filepaths]
    output = subprocess.run(" ".join(cmd), capture_output=True, shell=True)  # nosec
    if verbose:
        log.info(f"RF2NA output: {output}")


@beartype
def calculate_usalign_metrics(
    pred_pdb_filepath: str,
    reference_pdb_filepath: str,
    usalign_exec_path: str,
    flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates US-align structural metrics between predicted and reference protein structures.

    Args:
        pred_pdb_filepath (str): Filepath to predicted protein structure in PDB format.
        reference_pdb_filepath (str): Filepath to reference protein structure in PDB format.
        usalign_exec_path (str): Path to US-align executable.
        flags (List[str]): Command-line flags to pass to US-align, optional.

    Returns:
        Dict[str, Any]: Dictionary containing macromolecular US-align structural metrics and metadata.
    """
    # run US-align with subprocess and capture output
    cmd = [usalign_exec_path, pred_pdb_filepath, reference_pdb_filepath]
    if flags is not None:
        cmd += flags
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)  # nosec

    # parse US-align output to extract structural metrics
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Name of Structure_1:"):
            metrics["Name of Structure_1"] = line.split(": ", 1)[1]
        elif line.startswith("Name of Structure_2:"):
            metrics["Name of Structure_2"] = line.split(": ", 1)[1]
        elif line.startswith("Length of Structure_1:"):
            metrics["Length of Structure_1"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Length of Structure_2:"):
            metrics["Length of Structure_2"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Aligned length="):
            aligned_length = line.split("=")[1].split(",")[0]
            rmsd = line.split("=")[2].split(",")[0]
            seq_id = line.split("=")[4]
            metrics["Aligned length"] = int(aligned_length.strip())
            metrics["RMSD"] = float(rmsd.strip())
            metrics["Seq_ID"] = float(seq_id.strip())
        elif line.startswith("TM-score="):
            if "normalized by length of Structure_1" in line:
                metrics["TM-score_1"] = float(line.split("=")[1].split()[0])
            elif "normalized by length of Structure_2" in line:
                metrics["TM-score_2"] = float(line.split("=")[1].split()[0])

    return metrics


@beartype
def parse_qtmclust_cluster_file(file_path: str) -> List[List[Any]]:
    clusters = {}
    with open(file_path) as file:
        for line in file:
            columns = line.strip().split("\t")
            valid_columns = [col for col in columns if col]  # filter out empty columns
            cluster_repr = valid_columns[0]
            clusters[cluster_repr] = valid_columns
    return list(clusters.values())


@beartype
def run_qtmclust(
    chain_dir: str,
    chain_list_filepath: str,
    qtmclust_exec_path: str,
    output_cluster_filepath: Optional[str] = None,
    tm_cluster_threshold: float = 0.5,
    chain_ter_mode: Literal[0, 1, 2, 3] = 3,
    chain_split_mode: Literal[0, 1, 2] = 0,
) -> Optional[pd.DataFrame]:
    cmd = [
        qtmclust_exec_path,
        "-dir",
        (chain_dir if chain_dir.endswith("/") else chain_dir + "/"),
        chain_list_filepath,
        "-TMcut",
        str(tm_cluster_threshold),
        "-ter",
        str(chain_ter_mode),
        "-split",
        str(chain_split_mode),
    ]
    if output_cluster_filepath is not None:
        cmd += ["-o", output_cluster_filepath]
    subprocess.run(" ".join(cmd), capture_output=True, shell=True)  # nosec
    if output_cluster_filepath is not None:
        output_clusters = parse_qtmclust_cluster_file(output_cluster_filepath)
        output_clusters_df = pd.DataFrame(output_clusters)
        return output_clusters_df


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def calc_perplexity(pred, labels, mask):
    one_hot_labels = np.eye(pred.shape[-1])[labels]
    true_probs = np.sum(pred * one_hot_labels, axis=-1)
    ce = -np.log(true_probs + 1e-5)
    per_res_perplexity = np.exp(ce)
    return np.sum(per_res_perplexity * mask) / np.sum(mask)


def calc_mdtraj_metrics(pdb_path, metrics_prefix=""):
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == "C")
    pdb_helix_percent = np.mean(pdb_ss == "H")
    pdb_strand_percent = np.mean(pdb_ss == "E")
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
    pdb_ss_annotations_unavailable = (pdb_ss == "NA").all()
    pdb_rg = md.compute_rg(traj)[0]
    return {
        f"{metrics_prefix}non_coil_percent": DEFAULT_MISSING_VALUE
        if pdb_ss_annotations_unavailable
        else pdb_ss_percent,
        f"{metrics_prefix}coil_percent": DEFAULT_MISSING_VALUE
        if pdb_ss_annotations_unavailable
        else pdb_coil_percent,
        f"{metrics_prefix}helix_percent": DEFAULT_MISSING_VALUE
        if pdb_ss_annotations_unavailable
        else pdb_helix_percent,
        f"{metrics_prefix}strand_percent": DEFAULT_MISSING_VALUE
        if pdb_ss_annotations_unavailable
        else pdb_strand_percent,
        f"{metrics_prefix}radius_of_gyration": pdb_rg,
    }


def calc_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def complex_metrics(
    protein_pdb_filepath,
    na_pdb_filepath,
    protein_na_pdb_filepath,
    atom37_pos,
    gt_atom37_pos,
    gt_aatype,
    diffuse_mask,
    is_protein_residue_mask,
    is_na_residue_mask,
    pred_aatype=None,
    verbose=True,
):
    protein_inputs_present = is_protein_residue_mask.any().item()
    na_inputs_present = is_na_residue_mask.any().item()
    atom37_mask = np.any(atom37_pos, axis=-1)

    # assess secondary structure (SS) percentages
    # protein shape metrics #
    if protein_inputs_present:
        try:
            protein_mdtraj_metrics = calc_mdtraj_metrics(
                protein_pdb_filepath, metrics_prefix="protein_"
            )
        except Exception as e:
            if verbose:
                log.warning(
                    f"Could not generate MDTraj metrics for protein input {protein_pdb_filepath} due to {e}. Nullifying its MDTraj metrics..."
                )
            protein_mdtraj_metrics = {
                metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_SHAPE_METRICS
            }
            protein_inter_violations = {
                metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_INTER_VIOLATION_METRICS
            }
        protein_atom37_diffuse_mask = (diffuse_mask[..., None] * atom37_mask)[
            is_protein_residue_mask
        ]
        pred_prot_mol = au.create_full_prot(
            atom37_pos[is_protein_residue_mask], protein_atom37_diffuse_mask
        )
        protein_violation_metrics = amber_minimize.get_violation_metrics(pred_prot_mol)
        protein_struct_violations = protein_violation_metrics["structural_violations"]
        protein_inter_violations = {
            f"protein_{key}": value
            for key, value in protein_struct_violations["between_residues"].items()
        }
    else:
        protein_mdtraj_metrics = {
            metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_SHAPE_METRICS
        }
        protein_inter_violations = {
            metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_INTER_VIOLATION_METRICS
        }
    # NA shape metrics #
    if na_inputs_present:
        try:
            na_mdtraj_metrics = calc_mdtraj_metrics(na_pdb_filepath, metrics_prefix="na_")
        except Exception as e:
            if verbose:
                log.warning(
                    f"Could not generate MDTraj metrics for nucleic acid molecule input {na_pdb_filepath} due to {e}. Nullifying its MDTraj metrics..."
                )
            na_mdtraj_metrics = {metric: DEFAULT_MISSING_VALUE for metric in NA_SHAPE_METRICS}
    else:
        na_mdtraj_metrics = {metric: DEFAULT_MISSING_VALUE for metric in NA_SHAPE_METRICS}
    # protein-NA shape metrics #
    if protein_inputs_present and na_inputs_present:
        try:
            protein_na_mdtraj_metrics = calc_mdtraj_metrics(
                protein_na_pdb_filepath, metrics_prefix="protein_na_"
            )
        except Exception as e:
            if verbose:
                log.warning(
                    f"Could not generate MDTraj metrics for joint protein-nucleic acid molecule input {na_pdb_filepath} due to {e}. Nullifying its MDTraj metrics..."
                )
            protein_na_mdtraj_metrics = {
                metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_NA_SHAPE_METRICS
            }
    else:
        protein_na_mdtraj_metrics = {
            metric: DEFAULT_MISSING_VALUE for metric in PROTEIN_NA_SHAPE_METRICS
        }

    # assess geometry
    # protein geometry metrics #
    bb_mask = np.any(atom37_mask, axis=-1)

    if protein_inputs_present:
        protein_ca_pos = atom37_pos[..., PROTEIN_CA_IDX, :][
            is_protein_residue_mask & bb_mask.astype(bool)
        ]
        protein_ca_ca_bond_dev, protein_ca_ca_valid_percent = backbone_rep_pairwise_distance(
            protein_ca_pos, lit_pairwise_dist=protein_constants.ca_ca
        )
        (
            protein_num_ca_steric_clashes,
            protein_ca_steric_clash_percent,
        ) = backbone_rep_pairwise_clashes(protein_ca_pos, tol=1.5)
    else:
        protein_ca_ca_bond_dev, protein_ca_ca_valid_percent = (
            DEFAULT_MISSING_VALUE,
            DEFAULT_MISSING_VALUE,
        )
        protein_num_ca_steric_clashes, protein_ca_steric_clash_percent = (
            DEFAULT_MISSING_VALUE,
            DEFAULT_MISSING_VALUE,
        )
    # NA geometry metrics #
    if na_inputs_present:
        na_c4_prime_pos = atom37_pos[..., NA_C4_PRIME_IDX, :][
            is_na_residue_mask & bb_mask.astype(bool)
        ]
        (
            na_c4_prime_c4_prime_bond_dev,
            na_c4_prime_c4_prime_valid_percent,
        ) = backbone_rep_pairwise_distance(
            na_c4_prime_pos, lit_pairwise_dist=nucleotide_constants.c4_c4
        )
        (
            na_num_c4_prime_steric_clashes,
            na_c4_prime_steric_clash_percent,
        ) = backbone_rep_pairwise_clashes(na_c4_prime_pos, tol=1.5)
    else:
        na_c4_prime_c4_prime_bond_dev, na_c4_prime_c4_prime_valid_percent = (
            DEFAULT_MISSING_VALUE,
            DEFAULT_MISSING_VALUE,
        )
        na_num_c4_prime_steric_clashes, na_c4_prime_steric_clash_percent = (
            DEFAULT_MISSING_VALUE,
            DEFAULT_MISSING_VALUE,
        )

    # assess protein structural alignment
    if protein_inputs_present:
        bb_diffuse_mask = (diffuse_mask * bb_mask).astype(bool)
        unpad_protein_gt_scaffold_pos = gt_atom37_pos[..., PROTEIN_CA_IDX, :][
            is_protein_residue_mask & bb_diffuse_mask
        ]
        unpad_protein_pred_scaffold_pos = atom37_pos[..., PROTEIN_CA_IDX, :][
            is_protein_residue_mask & bb_diffuse_mask
        ]
        seq = du.aatype_to_seq(gt_aatype[is_protein_residue_mask & bb_diffuse_mask])
        _, protein_tm_score = calc_tm_score(
            unpad_protein_pred_scaffold_pos, unpad_protein_gt_scaffold_pos, seq, seq
        )
    else:
        protein_tm_score = DEFAULT_MISSING_VALUE

    metrics_dict = {
        "protein_ca_ca_bond_dev": protein_ca_ca_bond_dev,
        "protein_ca_ca_valid_percent": protein_ca_ca_valid_percent,
        "protein_ca_steric_clash_percent": protein_ca_steric_clash_percent,
        "protein_num_ca_steric_clashes": protein_num_ca_steric_clashes,
        "na_c4_prime_c4_prime_bond_dev": na_c4_prime_c4_prime_bond_dev,
        "na_c4_prime_c4_prime_valid_percent": na_c4_prime_c4_prime_valid_percent,
        "na_c4_prime_steric_clash_percent": na_c4_prime_steric_clash_percent,
        "na_num_c4_prime_steric_clashes": na_num_c4_prime_steric_clashes,
        "protein_tm_score": protein_tm_score,
        **protein_mdtraj_metrics,
        **na_mdtraj_metrics,
        **protein_na_mdtraj_metrics,
    }

    # assess sequence recovery when provided with a predicted sequence
    if pred_aatype is not None:
        assert (
            pred_aatype.shape == gt_aatype.shape
        ), "To assess sequence recovery, both the predicted and ground-truth sequence must be of the same length."
        metrics_dict["protein_na_sequence_recovery"] = np.sum(
            np.equal(pred_aatype, gt_aatype)
        ) / len(gt_aatype)
        if any([metric not in ALL_COMPLEX_METRICS for metric in PROTEIN_NA_SEQUENCE_METRICS]):
            # report e.g., sequence recovery when predicted sequences are provided as an input
            global ALL_COMPLEX_METRICS
            ALL_COMPLEX_METRICS += PROTEIN_NA_SEQUENCE_METRICS
            ALL_COMPLEX_METRICS = list(set(ALL_COMPLEX_METRICS))  # remove any duplicated metrics

    for k in PROTEIN_INTER_VIOLATION_METRICS:
        metrics_dict[k] = protein_inter_violations[k]
    metrics_dict = tree.map_structure(lambda x: np.mean(x).item(), metrics_dict)
    return metrics_dict


def backbone_rep_pairwise_distance(backbone_rep_pos, lit_pairwise_dist: float, tol=0.1):
    backbone_rep_bond_dists = np.linalg.norm(
        backbone_rep_pos - np.roll(backbone_rep_pos, 1, axis=0), axis=-1
    )[1:]
    backbone_rep_pairwise_dev = np.mean(np.abs(backbone_rep_bond_dists - lit_pairwise_dist))
    backbone_rep_pairwise_valid = np.mean(backbone_rep_bond_dists < (lit_pairwise_dist + tol))
    return backbone_rep_pairwise_dev, backbone_rep_pairwise_valid


def backbone_rep_pairwise_clashes(backbone_rep_pos, tol=1.5):
    backbone_rep_pairwise_dists2d = np.linalg.norm(
        backbone_rep_pos[:, None, :] - backbone_rep_pos[None, :, :], axis=-1
    )
    inter_dists = backbone_rep_pairwise_dists2d[
        np.where(np.triu(backbone_rep_pairwise_dists2d, k=0) > 0)
    ]
    clashes = inter_dists < tol
    return np.sum(clashes), np.mean(clashes)
