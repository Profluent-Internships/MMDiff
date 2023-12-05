# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Utilities for calculating all atom representations."""

from typing import Optional

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Bool, Float, Int64, jaxtyped
from openfold.data import data_transforms

from src.data.components.pdb import nucleotide_constants, protein_constants
from src.data.components.pdb import rigid_utils as ru
from src.data.components.pdb import vocabulary
from src.data.components.pdb.complex_constants import (
    NUM_NA_TORSIONS,
    NUM_PROT_NA_TORSIONS,
)

NODES_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes"]
NODES_LONG_TENSOR_TYPE = Int64[torch.Tensor, "... num_nodes"]
NODES_MASK_TYPE = Bool[torch.Tensor, "... num_nodes 37"]
COORDINATES_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 3"]
COORDINATES_14_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 14 3"]
COORDINATES_23_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 23 3"]
COORDINATES_27_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 27 3"]
COORDINATES_37_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 37 3"]
COORDINATES_37_MASK_TYPE = Float[torch.Tensor, "... num_nodes 37 3"]


# Residue Constants from OpenFold/AlphaFold2/OpenComplex.
IDEALIZED_PROTEIN_ATOM_POS37 = torch.tensor(protein_constants.restype_atom37_rigid_group_positions)
IDEALIZED_PROTEIN_ATOM_POS37_MASK = torch.any(IDEALIZED_PROTEIN_ATOM_POS37, axis=-1)
IDEALIZED_PROTEIN_ATOM_POS14 = torch.tensor(
    protein_constants.restype_compact_atom_rigid_group_positions
)
DEFAULT_PROTEIN_RESIDUE_FRAMES = torch.tensor(protein_constants.restype_rigid_group_default_frame)
PROTEIN_RESIDUE_ATOM14_MASK = torch.tensor(protein_constants.restype_compact_atom_mask)
PROTEIN_RESIDUE_GROUP_IDX = torch.tensor(protein_constants.restype_compact_atom_to_rigid_group)

IDEALIZED_NA_ATOM_POS27 = torch.tensor(nucleotide_constants.nttype_atom27_rigid_group_positions)
IDEALIZED_NA_ATOM_POS27_MASK = torch.any(IDEALIZED_NA_ATOM_POS27, axis=-1)
IDEALIZED_NA_ATOM_POS23 = torch.tensor(
    nucleotide_constants.nttype_compact_atom_rigid_group_positions
)
DEFAULT_NA_RESIDUE_FRAMES = torch.tensor(nucleotide_constants.nttype_rigid_group_default_frame)
NA_RESIDUE_ATOM23_MASK = torch.tensor(nucleotide_constants.nttype_compact_atom_mask)
NA_RESIDUE_GROUP_IDX = torch.tensor(nucleotide_constants.nttype_compact_atom_to_rigid_group)


@jaxtyped
@beartype
def of_protein_torsion_angles_to_frames(
    r: ru.Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> ru.Rigid:
    # [*, N, 11, 4, 4], where the last 3 dimensions of the `11` are zero-padding
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 11] transformations, i.e.
    #   One [*, N, 11, 3, 3] rotation matrix and
    #   One [*, N, 11, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 11, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 11, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1  # The upper-left diagonal value for 3D rotations
    all_rots[..., 1, 1] = alpha[..., 1]  # The first sine angle for 3D rotations
    all_rots[..., 1, 2] = -alpha[..., 0]  # The first cosine angle for 3D rotations
    all_rots[..., 2, 1:] = alpha  # The remaining sine and cosine angles for 3D rotations

    all_rots = ru.Rigid(ru.Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = ru.Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


@jaxtyped
@beartype
def of_na_torsion_angles_to_frames(
    r: ru.Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> ru.Rigid:
    # [*, N, 11, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 11] transformations, i.e.
    #   One [*, N, 11, 3, 3] rotation matrix and
    #   One [*, N, 11, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot1 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot1[..., 1] = 1

    alpha = torch.cat(
        [
            bb_rot1.expand(*alpha.shape[:-2], -1, -1),
            alpha,
        ],
        dim=-2,
    )

    # [*, N, 11, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1  # The upper-left diagonal value for 3D rotations
    all_rots[..., 1, 1] = alpha[..., 1]  # The first sine angle for 3D rotations
    all_rots[..., 1, 2] = -alpha[..., 0]  # The first cosine angle for 3D rotations
    all_rots[..., 2, 1:] = alpha  # The remaining sine and cosine angles for 3D rotations

    all_rots = ru.Rigid(ru.Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    backbone2_atom1_frame = all_frames[..., 1]  # C2'
    backbone2_atom2_frame = all_frames[..., 2]  # C1'
    backbone2_atom3_frame = all_frames[..., 3]  # N9/N1
    delta_frame_to_frame = all_frames[..., 4]  # O3'
    gamma_frame_to_frame = all_frames[..., 5]  # O5'
    beta_frame_to_frame = all_frames[..., 6]  # P
    alpha1_frame_to_frame = all_frames[..., 7]  # OP1
    alpha2_frame_to_frame = all_frames[..., 8]  # OP2
    tm_frame_to_frame = all_frames[..., 9]  # O2'
    chi_frame_to_frame = all_frames[..., 10]  # N1, N3, N6, N7, C2, C4, C5, C6, and C8

    backbone2_atom1_frame_to_bb = backbone2_atom1_frame
    backbone2_atom2_frame_to_bb = backbone2_atom2_frame
    # note: N9/N1 is built off the relative position of C1'
    backbone2_atom3_frame_to_bb = backbone2_atom2_frame.compose(backbone2_atom3_frame)
    delta_frame_to_bb = delta_frame_to_frame
    gamma_frame_to_bb = gamma_frame_to_frame
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)
    alpha1_frame_to_bb = beta_frame_to_bb.compose(alpha1_frame_to_frame)
    alpha2_frame_to_bb = beta_frame_to_bb.compose(alpha2_frame_to_frame)
    # use `backbone2_atom1/3_frames` to compose `tm` and `chi` frames,
    # since `backbone2_atom1/3_frames` place the `C2'` and `N9/N1` atoms
    # (i.e., two of the second backbone group's atoms)
    tm_frame_to_bb = backbone2_atom1_frame_to_bb.compose(tm_frame_to_frame)
    chi_frame_to_bb = backbone2_atom3_frame_to_bb.compose(chi_frame_to_frame)

    all_frames_to_bb = ru.Rigid.cat(
        [
            all_frames[..., 0].unsqueeze(-1),
            backbone2_atom1_frame_to_bb.unsqueeze(-1),
            backbone2_atom2_frame_to_bb.unsqueeze(-1),
            backbone2_atom3_frame_to_bb.unsqueeze(-1),
            delta_frame_to_bb.unsqueeze(-1),
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha1_frame_to_bb.unsqueeze(-1),
            alpha2_frame_to_bb.unsqueeze(-1),
            tm_frame_to_bb.unsqueeze(-1),
            chi_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global


@jaxtyped
@beartype
def prot_to_torsion_angles(
    aatype: NODES_TENSOR_TYPE,
    atom37: COORDINATES_37_TENSOR_TYPE,
    atom37_mask: COORDINATES_37_MASK_TYPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


@jaxtyped
@beartype
def protein_frames_to_atom14_pos(
    r: ru.Rigid, aatype: NODES_LONG_TENSOR_TYPE
) -> COORDINATES_14_TENSOR_TYPE:
    """Convert protein frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:
    """

    # [*, N, 14]
    group_mask = PROTEIN_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_PROTEIN_RESIDUE_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    frame_atom_mask = PROTEIN_RESIDUE_ATOM14_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_PROTEIN_ATOM_POS14.to(r.device)[aatype, ...]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


@jaxtyped
@beartype
def na_frames_to_atom23_pos(
    r: ru.Rigid, aatype: NODES_LONG_TENSOR_TYPE
) -> COORDINATES_23_TENSOR_TYPE:
    """Convert nucleic acid (NA) frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 11, 3]
        aatype: Residue types. [..., N]

    Returns:
    """

    # [*, N, 23]
    group_mask = NA_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    # [*, N, 23, 11]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_NA_RESIDUE_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 23, 11]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 23, 1]
    frame_atom_mask = NA_RESIDUE_ATOM23_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    frame_null_pos = IDEALIZED_NA_ATOM_POS23.to(r.device)[aatype, ...]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


@jaxtyped
@beartype
def compute_backbone(
    bb_rigids: ru.Rigid,
    torsions: torch.Tensor,
    is_protein_residue_mask: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    aatype: Optional[torch.Tensor] = None,
) -> Tuple[
    COORDINATES_37_TENSOR_TYPE,
    NODES_MASK_TYPE,
    NODES_MASK_TYPE,
    NODES_LONG_TENSOR_TYPE,
    COORDINATES_23_TENSOR_TYPE,
]:
    protein_inputs_present = is_protein_residue_mask.any().item()
    na_inputs_present = is_na_residue_mask.any().item()

    torsion_angles = torch.tile(
        torsions[..., None, :2],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (NUM_PROT_NA_TORSIONS, 1),
    )
    if na_inputs_present:
        # note: for nucleic acid molecules, we insert their predicted torsion angles
        # as the first eight torsion entries and tile the remaining torsion entries
        # using the first of their eight predicted torsion angles
        torsion_angles[..., :NUM_NA_TORSIONS, :][is_na_residue_mask] = torsions[
            is_na_residue_mask
        ].view(torsions[is_na_residue_mask].shape[0], -1, 2)
    rot_dim = (
        (4,) if bb_rigids._rots._quats is not None else (3, 3)
    )  # i.e., otherwise, anticipate rotations being used
    aatype = (
        aatype
        if aatype is not None
        else torch.zeros(bb_rigids.shape, device=bb_rigids.device, dtype=torch.long)
    )
    if protein_inputs_present:
        protein_bb_rigids = bb_rigids[is_protein_residue_mask].reshape(
            new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
            new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
        )
        protein_torsion_angles = torsion_angles[is_protein_residue_mask].view(
            torsion_angles.shape[0], -1, NUM_PROT_NA_TORSIONS, 2
        )
        protein_aatype = aatype[is_protein_residue_mask].view(aatype.shape[0], -1)
        padded_default_protein_residue_frames = F.pad(
            DEFAULT_PROTEIN_RESIDUE_FRAMES.to(bb_rigids.device), (0, 0, 0, 0, 0, 3)
        )
        all_protein_frames = of_protein_torsion_angles_to_frames(
            r=protein_bb_rigids,
            alpha=protein_torsion_angles,
            aatype=protein_aatype,
            rrgdf=padded_default_protein_residue_frames,
        )
        protein_atom14_pos = protein_frames_to_atom14_pos(all_protein_frames, protein_aatype)
    if na_inputs_present:
        na_bb_rigids = bb_rigids[is_na_residue_mask].reshape(
            new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
            new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
        )
        na_torsion_angles = torsion_angles[is_na_residue_mask].view(
            torsion_angles.shape[0], -1, NUM_PROT_NA_TORSIONS, 2
        )
        na_aatype = aatype[is_na_residue_mask].view(aatype.shape[0], -1)
        na_aatype_in_original_range = na_aatype.min() > vocabulary.protein_restype_num
        effective_na_aatype = (
            na_aatype - (vocabulary.protein_restype_num + 1)
            if na_aatype_in_original_range
            else na_aatype
        )
        all_na_frames = of_na_torsion_angles_to_frames(
            r=na_bb_rigids,
            alpha=na_torsion_angles,
            aatype=effective_na_aatype,
            rrgdf=DEFAULT_NA_RESIDUE_FRAMES.to(bb_rigids.device),
        )
        na_atom23_pos = na_frames_to_atom23_pos(all_na_frames, effective_na_aatype)
    if protein_inputs_present and na_inputs_present:
        atom23_pos = torch.cat((F.pad(protein_atom14_pos, (0, 0, 0, 9)), na_atom23_pos), dim=1)
    elif protein_inputs_present:
        atom23_pos = F.pad(protein_atom14_pos, (0, 0, 0, 9))
    elif na_inputs_present:
        atom23_pos = na_atom23_pos
    else:
        raise Exception("Either protein or nucleic acid chains must be provided as inputs.")
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    atom37_bb_supervised_mask = torch.zeros(
        bb_rigids.shape + (37,), device=bb_rigids.device, dtype=torch.bool
    )
    if protein_inputs_present:
        # note: proteins' atom23 bb order = ['N', 'CA', 'C', 'O', 'CB', ...]
        # note: proteins' atom37 bb order = ['N', 'CA', 'C', 'CB', 'O', ...]
        # fmt: off
        atom37_bb_pos[..., :3, :][is_protein_residue_mask] = atom23_pos[..., :3, :][is_protein_residue_mask]  # reindex N, CA, and C
        atom37_bb_pos[..., 4, :][is_protein_residue_mask] = atom23_pos[..., 3, :][is_protein_residue_mask]  # reindex O
        atom37_bb_pos[..., 3, :][is_protein_residue_mask] = atom23_pos[..., 4, :][is_protein_residue_mask]  # reindex CB, which will be all zeros for `GLY`
        # fmt: on
        atom37_bb_supervised_mask[..., :5][is_protein_residue_mask] = True
    if na_inputs_present:
        # note: nucleic acids' atom23 bb order = ['C3'', 'C4'', 'O4'', 'C2'', 'C1'', 'C5'', 'O3'', 'O5'', 'P', 'OP1', 'OP2', 'N9', ...]
        # note: nucleic acids' atom37 bb order = ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
        # fmt: off
        atom37_bb_pos[..., 2:4, :][is_na_residue_mask] = atom23_pos[..., :2, :][is_na_residue_mask]  # reindex C3' and C4'
        atom37_bb_pos[..., 6, :][is_na_residue_mask] = atom23_pos[..., 2, :][is_na_residue_mask]  # reindex O4'
        atom37_bb_pos[..., 1, :][is_na_residue_mask] = atom23_pos[..., 3, :][is_na_residue_mask]  # reindex C2'
        atom37_bb_pos[..., 0, :][is_na_residue_mask] = atom23_pos[..., 4, :][is_na_residue_mask]  # reindex C1'
        atom37_bb_pos[..., 4, :][is_na_residue_mask] = atom23_pos[..., 5, :][is_na_residue_mask]  # reindex C5'
        atom37_bb_pos[..., 7, :][is_na_residue_mask] = atom23_pos[..., 6, :][is_na_residue_mask]  # reindex O3'
        atom37_bb_pos[..., 5, :][is_na_residue_mask] = atom23_pos[..., 7, :][is_na_residue_mask]  # reindex O5'
        atom37_bb_pos[..., 9:12, :][is_na_residue_mask] = atom23_pos[..., 8:11, :][is_na_residue_mask]  # reindex P, OP1, and OP2
        atom37_bb_pos[..., 18, :][is_na_residue_mask] = atom23_pos[..., 11, :][is_na_residue_mask]  # reindex N9, which instead reindexes N1 for pyrimidine residues
        # fmt: on
        atom37_bb_supervised_mask[..., :8][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 9:12][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 18][is_na_residue_mask] = True
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, atom37_bb_supervised_mask, aatype, atom23_pos


@jaxtyped
@beartype
def calculate_neighbor_angles(
    R_ac: COORDINATES_TENSOR_TYPE, R_ab: COORDINATES_TENSOR_TYPE
) -> NODES_TENSOR_TYPE:
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle


@jaxtyped
@beartype
def vector_projection(
    R_ab: COORDINATES_TENSOR_TYPE, P_n: COORDINATES_TENSOR_TYPE
) -> COORDINATES_TENSOR_TYPE:
    """Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n
