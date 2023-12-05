# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import collections
import copy
import dataclasses
import gzip
import io
import os
import pickle  # nosec
import string
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Any, Dict, List
from Bio import PDB
from Bio.SeqUtils import seq1
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils import data

from src.data.components.pdb import chemical, complex_constants
from src.data.components.pdb import data_utils as du
from src.data.components.pdb import (
    nucleotide_constants,
    parsers,
    protein,
    protein_constants,
    rigid_utils,
    so3_utils,
    vocabulary,
)
from src.data.components.pdb.data_transforms import convert_na_aatype6_to_aatype9
from src.utils.pylogger import get_pylogger

Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

CHAIN_FEATS = [
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "b_factors",
    "asym_id",
    "sym_id",
    "entity_id",
]
UNPADDED_FEATS = [
    "t",
    "rot_score_scaling",
    "trans_score_scaling",
    "t_seq",
    "t_struct",
    "molecule_type_encoding",
    "is_protein_residue_mask",
    "is_na_residue_mask",
]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]

MAX_NUM_ATOMS_PER_RESIDUE = (
    23  # note: `23` comes from the maximum number of atoms in a nucleic acid
)
RESIDUE_ATOM_FEATURES_AXIS_MAPPING = {"atom_positions": -2, "atom_mask": -1, "atom_b_factors": -1}

COMPLEX_FEATURE_CONCAT_MAP = {
    # note: follows the format `(protein_feature_name, na_feature_name, complex_feature_name, padding_dim): max_feature_dim_size`
    ("all_atom_positions", "all_atom_positions", "all_atom_positions", 1): 37,
    ("all_atom_mask", "all_atom_mask", "all_atom_mask", 1): 37,
    ("atom_deoxy", "atom_deoxy", "atom_deoxy", 0): 0,
    ("residx_atom14_to_atom37", "residx_atom23_to_atom27", "residx_atom23_to_atom37", 1): 23,
    ("atom14_gt_positions", "atom23_gt_positions", "atom23_gt_positions", 1): 23,
    ("rigidgroups_gt_frames", "rigidgroups_gt_frames", "rigidgroups_gt_frames", 1): 11,
    ("torsion_angles_sin_cos", "torsion_angles_sin_cos", "torsion_angles_sin_cos", 1): 10,
}

log = get_pylogger(__name__)


def aatype_to_seq(aatype):
    return "".join([complex_constants.restypes[x] for x in aatype])


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)  # nosec
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f"Failed to read {read_path}. First error: {e}\n Second error: {e2}")
            raise (e)


@beartype
def parse_pdb_feats(
    pdb_name: str,
    pdb_path: str,
    scale_factor: float = 1.0,
    chain_id=None,
    return_na_aatype9: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        chain_id: if given, which chain (or chains) to parse; defaults to all chains.
        return_na_aatype9: whether to return the `aatype9` version of an NA input sequence
        verbose: whether to log everything.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {chain.id: chain for chain in structure.get_chains()}

    def _process_chain_id(x):
        # Convert chain id into int
        chain = struct_chains[x]
        chain_index = du.chain_str_to_int(x)
        chain_mol = parsers.process_chain_pdb(chain, chain_index, x, verbose=verbose)
        if chain_mol[-1]["molecule_type"] not in ["protein", "na"]:
            raise NotImplementedError(
                f"An unknown molecule type {chain_mol[-1]['molecule_type']} was found. Only proteins or nucleic acids are currently supported."
            )
        chain_mol_constants = chain_mol[-1]["molecule_constants"]
        chain_mol_backbone_atom_name = chain_mol[-1]["molecule_backbone_atom_name"]
        chain_dict = parsers.macromolecule_outputs_to_dict(chain_mol)
        chain_dict = du.parse_chain_feats_pdb(
            chain_feats=chain_dict,
            molecule_constants=chain_mol_constants,
            molecule_backbone_atom_name=chain_mol_backbone_atom_name,
            scale_factor=scale_factor,
        )
        if return_na_aatype9 and chain_mol[-1]["molecule_type"] == "na":
            chain_dict["aatype"] = convert_na_aatype6_to_aatype9(
                torch.from_numpy(chain_dict["aatype"]), chain_dict["atom_deoxy"]
            )
        return chain_dict

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {x: _process_chain_id(x) for x in chain_id}
    elif chain_id is None:
        return {x: _process_chain_id(x) for x in struct_chains}
    else:
        raise ValueError(f"Unrecognized chain list {chain_id}")


def compare_conf(conf1, conf2):
    return OmegaConf.to_yaml(conf1) == OmegaConf.to_yaml(conf2)


def parse_pdb(filename):
    lines = open(filename).readlines()
    return parse_pdb_lines(lines)


def parse_pdb_lines(lines):
    # indices of residues observed in the structure
    idx_s = [
        int(line[22:26]) for line in lines if line[:4] == "ATOM" and line[12:16].strip() == "CA"
    ]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    seq = []
    for line in lines:
        if line[:4] != "ATOM":
            continue
        resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
        seq.append(vocabulary.restype_3to1[aa])
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(chemical.aa2long[chemical.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx, i_atm, :] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    return xyz, mask, np.array(idx_s), "".join(seq)


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def chain_int_to_str(chain_int: int):
    if chain_int < len(ALPHANUMERIC):
        return INT_TO_CHAIN[chain_int]

    chain_str = ""
    while chain_int > 0:
        remainder = chain_int % len(ALPHANUMERIC)
        chain_str = INT_TO_CHAIN[remainder] + chain_str
        chain_int = chain_int // len(ALPHANUMERIC)
    return chain_str


def rigid_frames_from_atom_14(atom_14):
    n_atoms = atom_14[:, 0]
    ca_atoms = atom_14[:, 1]
    c_atoms = atom_14[:, 2]
    return rigid_utils.Rigid.from_3_points(n_atoms, ca_atoms, c_atoms)


def compose_rotvec(r1, r2):
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum("...ij,...jk->...ik", R1, R2)
    return matrix_to_rotvec(cR)


def rotvec_to_matrix(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(mat):
    return Rotation.from_matrix(mat).as_rotvec()


def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats


def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False
    )
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)


def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)


# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):
    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # print(filename)

    if filename.split(".")[-1] == "gz":
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)

    # read file line by line
    for line in fp:
        # skip labels
        if line[0] == ">":
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c == "-" else 1 for c in line])
        i = np.zeros(L)

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a == 1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos, num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = list("ACDEFGHIKLMNPQRSTVWYX")
    encoding = np.array(alphabet, dtype="|S1").view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype="|S1").view(np.uint8)
    for letter, enc in zip(alphabet, encoding):
        res_cat = vocabulary.protein_restype_order_with_x.get(
            letter, vocabulary.protein_restype_num_with_x
        )
        msa[msa == enc] = res_cat

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa, ins


def write_checkpoint(
    ckpt_path: str,
    model,
    conf,
    optimizer,
    epoch,
    step,
    logger=None,
    use_torch=True,
):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    for fname in os.listdir(ckpt_dir):
        if ".pkl" in fname or ".pth" in fname:
            os.remove(os.path.join(ckpt_dir, fname))
    if logger is not None:
        logger.info(f"Serializing experiment state to {ckpt_path}")
    else:
        print(f"Serializing experiment state to {ckpt_path}")
    write_pkl(
        ckpt_path,
        {"model": model, "conf": conf, "optimizer": optimizer, "epoch": epoch, "step": step},
        use_torch=use_torch,
    )


def add_padding_to_array_axis(
    array: np.ndarray,
    axis: int,
    max_axis_size: int,
    pad_front: bool = False,
    pad_mode: str = "constant",
    pad_value: Any = 0,
) -> np.ndarray:
    assert (
        axis < array.ndim
    ), "Requested axis for padding must be in range for number of array axes."
    pad_dims = [(0, 0)] * array.ndim
    pad_length = max(max_axis_size - array.shape[axis], 0)
    pad_dims[axis] = (pad_length, 0) if pad_front else (0, pad_length)
    return np.pad(array, pad_width=pad_dims, mode=pad_mode, constant_values=pad_value)


def add_padding_to_tensor_dim(
    tensor: torch.Tensor,
    dim: int,
    max_dim_size: int,
    pad_front: bool = False,
    pad_mode: str = "constant",
    pad_value: Any = 0,
) -> torch.Tensor:
    assert (
        dim < tensor.ndim
    ), "Requested dimension for padding must be in range for number of tensor dimensions."
    pad_dims = [(0, 0)] * tensor.ndim
    pad_length = max(max_dim_size - tensor.shape[dim], 0)
    dim_padding = (pad_length, 0) if pad_front else (0, pad_length)
    dim_to_pad = ((tensor.ndim - 1) - dim) if tensor.ndim % 2 == 0 else dim
    pad_dims[dim_to_pad] = dim_padding
    pad_dims = tuple(entry for dim_pad in pad_dims for entry in dim_pad)
    return F.pad(tensor, pad=pad_dims, mode=pad_mode, value=pad_value)


def concat_np_features(
    np_dicts: List[Dict[str, np.ndarray]],
    add_batch_dim: bool,
    max_residue_atom_features_axis_size: int = MAX_NUM_ATOMS_PER_RESIDUE,
):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.
        max_residue_atom_features_axis_size: the maximum number of atoms
            per residue to which to pad each residue's atomic features
            (e.g., 3D coordinates).

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            if feat_name in RESIDUE_ATOM_FEATURES_AXIS_MAPPING:
                feat_val = add_padding_to_array_axis(
                    array=feat_val,
                    axis=RESIDUE_ATOM_FEATURES_AXIS_MAPPING[feat_name],
                    max_axis_size=max_residue_atom_features_axis_size,
                    pad_value=0,
                )
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def concat_complex_torch_features(
    complex_torch_dict: Dict[str, torch.Tensor],
    protein_torch_dict: Dict[str, torch.Tensor],
    na_torch_dict: Dict[str, torch.Tensor],
    feature_concat_map: Dict[Tuple[str, str], int],
    add_batch_dim: bool,
):
    """Performs a concatenation of complex feature dicts.

    Args:
        complex_torch_dict: dict in which to store concatenated complex features.
        protein_torch_dict: dict in which to find available protein features.
        na_torch_dict: dict in which to find available nucleic acid (NA) features.
        add_batch_dim: whether to add a batch dimension to each complex feature.

    Returns:
        A single dict with all the complex features concatenated.
    """
    for (
        protein_feature,
        na_feature,
        complex_feature,
        padding_dim,
    ), max_feature_dim_size in feature_concat_map.items():
        # Parse available protein and nucleic acid features
        protein_feature_tensor, na_feature_tensor = None, None
        if protein_feature in protein_torch_dict:
            protein_feature_tensor = protein_torch_dict[protein_feature]
        if na_feature in na_torch_dict:
            na_feature_tensor = na_torch_dict[na_feature]
        # Add batch dimension as requested
        if add_batch_dim:
            protein_feature_tensor = (
                protein_feature_tensor[None]
                if protein_feature_tensor is not None
                else protein_feature_tensor
            )
            na_feature_tensor = (
                na_feature_tensor[None] if na_feature_tensor is not None else na_feature_tensor
            )
        # Pad features for each type of molecule
        padded_protein_feat_val = (
            add_padding_to_tensor_dim(
                tensor=protein_feature_tensor,
                dim=padding_dim,
                max_dim_size=max_feature_dim_size,
                pad_value=0,
            )
            if protein_feature_tensor is not None
            else protein_feature_tensor
        )
        padded_na_feat_val = (
            add_padding_to_tensor_dim(
                tensor=na_feature_tensor,
                dim=padding_dim,
                max_dim_size=max_feature_dim_size,
                pad_value=0,
            )
            if na_feature_tensor is not None
            else na_feature_tensor
        )
        # Concatenate features between molecule types as necessary
        if padded_protein_feat_val is not None and padded_na_feat_val is not None:
            concat_padded_feat_val = torch.concatenate(
                (padded_protein_feat_val, padded_na_feat_val), dim=(1 if add_batch_dim else 0)
            )
        elif padded_protein_feat_val is not None:
            concat_padded_feat_val = padded_protein_feat_val
        elif padded_na_feat_val is not None:
            concat_padded_feat_val = padded_na_feat_val
        else:
            raise Exception("Features for at least one type of molecule must be provided.")
        complex_torch_dict[complex_feature] = concat_padded_feat_val
    return complex_torch_dict


def get_len(x, mask_name: str):
    return x[mask_name].shape[0]


def length_batching(
    np_dicts: List[Dict[str, np.ndarray]],
    max_squared_res: int,
):
    dicts_by_length = [(get_len(x, "res_mask"), x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = int(max_squared_res // max_len**2)
    padded_batch = [pad_feats(x, max_len) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)


def create_data_loader(
    torch_dataset: data.Dataset,
    batch_size,
    shuffle,
    sampler=None,
    num_workers=0,
    np_collate=False,
    max_squared_res=1e6,
    length_batch=False,
    drop_last=False,
    prefetch_factor=2,
):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = partial(concat_np_features, add_batch_dim=True)
    elif length_batch:
        collate_fn = partial(length_batching, max_squared_res=max_squared_res)
    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context="fork" if num_workers != 0 else None,
    )


def parse_chain_feats_pdb(
    chain_feats, molecule_constants, molecule_backbone_atom_name, scale_factor=1.0
):
    core_atom_idx = molecule_constants.atom_order[molecule_backbone_atom_name]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, core_atom_idx]
    bb_pos = chain_feats["atom_positions"][:, core_atom_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, core_atom_idx]
    return chain_feats


def parse_chain_feats_mmcif(
    chain_feats, molecule_constants, molecule_backbone_atom_name, scale_factor=1.0
):
    core_atom_idx = molecule_constants.atom_order[molecule_backbone_atom_name]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, core_atom_idx]
    bb_pos = chain_feats["atom_positions"][:, core_atom_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, core_atom_idx]
    return chain_feats


def get_complex_is_ca_mask(complex_feats):
    is_protein_residue_mask = complex_feats["molecule_type_encoding"][:, 0] == 1
    is_na_residue_mask = (complex_feats["molecule_type_encoding"][:, 1] == 1) | (
        complex_feats["molecule_type_encoding"][:, 2] == 1
    )
    complex_is_ca_mask = np.zeros_like(complex_feats["atom_mask"], dtype=np.bool_)
    complex_is_ca_mask[is_protein_residue_mask, protein_constants.atom_order["CA"]] = True
    complex_is_ca_mask[is_na_residue_mask, nucleotide_constants.atom_order["C4'"]] = True
    return complex_is_ca_mask


def parse_complex_feats(complex_feats, scale_factor=1.0):
    complex_is_ca_mask = get_complex_is_ca_mask(complex_feats)
    complex_feats["bb_mask"] = complex_feats["atom_mask"][complex_is_ca_mask]
    bb_pos = complex_feats["atom_positions"][complex_is_ca_mask]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(complex_feats["bb_mask"]) + 1e-5)
    centered_pos = complex_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    complex_feats["atom_positions"] = scaled_pos * complex_feats["atom_mask"][..., None]
    complex_feats["bb_positions"] = complex_feats["atom_positions"][complex_is_ca_mask]
    return complex_feats


def rigid_frames_from_all_atom(all_atom_pos):
    rigid_atom_pos = []
    for atom in ["N", "CA", "C"]:
        atom_idx = protein_constants.atom_order[atom]
        atom_pos = all_atom_pos[..., atom_idx, :]
        rigid_atom_pos.append(atom_pos)
    return rigid_utils.Rigid.from_3_points(*rigid_atom_pos)


def pad_pdb_feats(raw_feats, max_len):
    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats


def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def quat_to_rotvec(quat, eps=1e-6):
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec


def quat_to_rotmat(quat, eps=1e-6):
    rot_vec = quat_to_rotvec(quat, eps)
    return so3_utils.Exp(rot_vec)


def save_fasta(
    pred_seqs,
    seq_names,
    file_path,
):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as f:
        for x, y in zip(seq_names, pred_seqs):
            f.write(f">{x}\n{y}\n")


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


@beartype
def add_alphafold_multimer_chain_assembly_features(
    chain_feat_dicts: List[Dict[str, Any]], chain_feat_name: str = "aatype"
):
    # Add assembly features from AlphaFold-Multimer,
    # relying on references to each `chain_dict` object
    # to propagate back to the contents of `chain_feat_dicts` below
    # (via in-place operations)
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_dict in chain_feat_dicts:
        seq = tuple(chain_dict[chain_feat_name])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_dict)

    new_all_chain_dict = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_dict in enumerate(group_chain_features, start=1):
            new_all_chain_dict[f"{du.int_id_to_str_id(entity_id)}_{sym_id}"] = chain_dict
            seq_length = len(chain_dict[chain_feat_name])
            chain_dict["asym_id"] = chain_id * np.ones(seq_length)
            chain_dict["sym_id"] = sym_id * np.ones(seq_length)
            chain_dict["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1


@beartype
def pdb_to_fasta(
    input_pdb_filepath: str,
    output_dir: str,
    output_id: str,
    use_rf2na_fasta_naming_scheme: bool = False,
) -> Dict[str, Dict[str, bool]]:
    # Create a PDB parser
    parser = PDB.PDBParser()

    # Load the PDB structure
    structure = parser.get_structure("pdb", input_pdb_filepath)

    # Extract the chain sequences as separate FASTA files
    dna_tokens = nucleotide_constants.deoxy_restypes
    rna_tokens = [
        restype for restype in nucleotide_constants.restypes if restype not in dna_tokens
    ]
    molecule_types = [
        "is_protein_sequence",
        "is_dna_sequence",
        "is_rna_sequence",
        "is_dna_rna_sequence",
        "is_protein_dna_sequence",
        "is_protein_rna_sequence",
        "is_protein_dna_rna_sequence",
    ]
    default_molecule_type_mapping = {molecule_type: False for molecule_type in molecule_types}
    chain_id_molecule_type_mappings = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            chain_id_molecule_type_mappings[str(chain_id)] = copy.deepcopy(
                default_molecule_type_mapping
            )

            # Check if the chain is of amino acids or nucleic acids while assembling its sequence
            if all(PDB.is_aa(residue.get_resname()) for residue in chain.get_residues()):
                # note: indicates the sequence is a protein sequence
                chain_id_molecule_type_mappings[str(chain_id)]["is_protein_sequence"] = True
                sequence = "".join(seq1(residue.get_resname()) for residue in chain.get_residues())
            elif not any(PDB.is_aa(residue.get_resname()) for residue in chain.get_residues()):
                # note: indicates the sequence is a type of nucleic acid sequence
                resnames = [residue.get_resname() for residue in chain.get_residues()]
                contains_all_dna_tokens = all(resname in dna_tokens for resname in resnames)
                contains_all_rna_tokens = all(resname in rna_tokens for resname in resnames)
                if contains_all_dna_tokens:
                    chain_id_molecule_type_mappings[str(chain_id)]["is_dna_sequence"] = True
                elif contains_all_rna_tokens:
                    chain_id_molecule_type_mappings[str(chain_id)]["is_rna_sequence"] = True
                else:
                    chain_id_molecule_type_mappings[str(chain_id)]["is_dna_rna_sequence"] = True
                sequence = "".join(
                    (resname[-1] if len(resname) >= 2 else resname) for resname in resnames
                )
            else:
                # note: indicates the sequence is a type of protein-nucleic acid sequence
                resnames = [residue.get_resname() for residue in chain.get_residues()]
                contains_no_dna_tokens = not any(resname in dna_tokens for resname in resnames)
                contains_no_rna_tokens = not any(resname in rna_tokens for resname in resnames)
                if contains_no_rna_tokens:
                    chain_id_molecule_type_mappings[str(chain_id)][
                        "is_protein_dna_sequence"
                    ] = True
                elif contains_no_dna_tokens:
                    chain_id_molecule_type_mappings[str(chain_id)][
                        "is_protein_rna_sequence"
                    ] = True
                else:
                    chain_id_molecule_type_mappings[str(chain_id)][
                        "is_protein_dna_rna_sequence"
                    ] = True
                # note: for DNA residue names, strip off the leading `D` character
                sequence_resnames = [
                    (seq1(resname) if PDB.is_aa(resname) else resname) for resname in resnames
                ]
                sequence = "".join(
                    (resname[-1] if len(resname) >= 2 else resname)
                    for resname in sequence_resnames
                )

            assert (
                sum(
                    [
                        assignment
                        for assignment in chain_id_molecule_type_mappings[str(chain_id)].values()
                    ]
                )
                == 1
            ), "Within `pdb_to_fasta()`, only a single molecule type may be assigned to each input chain."

            # Export each chain sequence to a separate FASTA file, using RoseTTAFold2NA's input file naming convention to name each FASTA
            if (
                use_rf2na_fasta_naming_scheme
                and chain_id_molecule_type_mappings[str(chain_id)]["is_protein_sequence"]
            ):
                output_fasta_filepath = f"{output_dir}/P:{output_id}_chain_{chain_id}.fasta"
            elif (
                use_rf2na_fasta_naming_scheme
                and chain_id_molecule_type_mappings[str(chain_id)]["is_dna_sequence"]
            ):
                output_fasta_filepath = f"{output_dir}/S:{output_id}_chain_{chain_id}.fasta"
            elif (
                use_rf2na_fasta_naming_scheme
                and chain_id_molecule_type_mappings[str(chain_id)]["is_rna_sequence"]
            ):
                output_fasta_filepath = f"{output_dir}/R:{output_id}_chain_{chain_id}.fasta"
            elif (
                use_rf2na_fasta_naming_scheme
                and chain_id_molecule_type_mappings[str(chain_id)]["is_protein_rna_sequence"]
            ):
                # note: RoseTTAFold2NA supports paired protein-RNA sequence inputs, via protein-RNA MSA pairing
                output_fasta_filepath = f"{output_dir}/PR:{output_id}_chain_{chain_id}.fasta"
            else:
                if use_rf2na_fasta_naming_scheme:
                    log.warning(
                        f"For chain {chain_id} within {input_pdb_filepath}, sequence does not meet RoseTTAFold2NA's FASTA file naming standards."
                    )
                output_fasta_filepath = f"{output_dir}/{output_id}_chain_{chain_id}.fasta"
            with open(output_fasta_filepath, "w") as fasta_out:
                fasta_out.write(f">{output_id}_chain_{chain_id}\n")
                fasta_out.write(f"{sequence}\n")

            # store sequence metadata within mappings
            chain_id_molecule_type_mappings[str(chain_id)]["sequence"] = sequence
            chain_id_molecule_type_mappings[str(chain_id)]["fasta_path"] = output_fasta_filepath

    # inform the user of this function about the contents of the given PDB sequence
    return chain_id_molecule_type_mappings


@beartype
def get_unique_chain_names(pdb_file_path: str) -> List[str]:
    # Create a PDBParser object
    parser = PDB.PDBParser()

    # Parse the PDB file
    structure = parser.get_structure("pdb_structure", pdb_file_path)

    # Initialize an empty set to store unique chain names
    unique_chains = set()

    # Iterate over all models in the structure
    for model in structure:
        # Iterate over all chains in the model
        for chain in model:
            # Get the chain identifier (name)
            chain_name = chain.get_id()
            # Add the chain name to the set of unique chains
            unique_chains.add(chain_name)

    # Convert the set of unique chains to a list and return it
    return list(unique_chains)
