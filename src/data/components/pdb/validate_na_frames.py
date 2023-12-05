import copy
import os
from typing import Any, Dict

import numpy as np
import torch

import src.models.components.pdb.analysis_utils as au
from src.data.components.pdb import all_atom
from src.data.components.pdb import rigid_utils as ru
from src.data.components.pdb import vocabulary
from src.data.components.pdb.data_transforms import (
    atom27_to_frames,
    atom27_to_torsion_angles,
    convert_na_aatype6_to_aatype9,
)


def run_mock_forward_pass(
    batch: Dict[str, Any],
    torsions: torch.Tensor,
    is_protein_residue_mask: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    output_dir: str = os.getcwd(),
) -> Dict[str, Any]:
    # construct frames
    frame_outputs = atom27_to_frames(batch)
    rigids = ru.Rigid.from_tensor_4x4(frame_outputs["rigidgroups_gt_frames"])
    atom37 = all_atom.compute_backbone(
        bb_rigids=rigids[..., 0],  # note: only frames for backbone group 1 are to be used here
        torsions=torsions,
        is_protein_residue_mask=is_protein_residue_mask,
        is_na_residue_mask=is_na_residue_mask,
        aatype=convert_na_aatype6_to_aatype9(
            copy.deepcopy(batch["aatype"]), deoxy_offset_mask=batch["atom_deoxy"]
        ),
    )[0]

    # collect unpadded outputs
    unpad_is_protein_residue_mask = is_protein_residue_mask[0].tolist()
    unpad_is_na_residue_mask = is_na_residue_mask[0].tolist()
    unpad_pred_pos = atom37[0].cpu().numpy()
    unpad_complex_restype = (
        convert_na_aatype6_to_aatype9(
            copy.deepcopy(batch["aatype"]), deoxy_offset_mask=batch["atom_deoxy"]
        )[0]
        .cpu()
        .numpy()
    )
    unpad_complex_restype[unpad_is_na_residue_mask] += vocabulary.protein_restype_num + 1
    unpad_complex_restype[
        unpad_is_protein_residue_mask
    ] = 0  # denote the default amino acid type (i.e., Alanine)
    unpad_complex_restype[
        unpad_is_na_residue_mask
    ] = 21  # denote the default nucleic acid type (i.e., Adenine)
    unpad_node_chain_indices = batch["residue_chain_indices"][0].cpu().numpy()
    unpad_fixed_mask = batch["fixed_residue_mask"][0].cpu().numpy()
    b_factors = np.tile(1 - unpad_fixed_mask[..., None], 37) * 100

    # save outputs as PDB files
    (
        saved_protein_pdb_path,
        saved_na_pdb_path,
        saved_protein_na_pdb_path,
    ) = au.write_complex_to_pdbs(
        complex_pos=unpad_pred_pos,
        output_filepath=os.path.join(
            output_dir,
            "len_1_sample_0.pdb",
        ),
        restype=unpad_complex_restype,
        chain_index=unpad_node_chain_indices,
        b_factors=b_factors,
        is_protein_residue_mask=unpad_is_protein_residue_mask,
        is_na_residue_mask=unpad_is_na_residue_mask,
        no_indexing=True,
    )
    return saved_protein_na_pdb_path


def main():
    # construct batch
    na_aatype = torch.tensor([24], dtype=torch.long)
    na_all_atom_positions = torch.tensor(
        [
            # note: derived manually from a random `U` residue within `6vrd.pdb`
            [3.949, 32.665, -7.611],  # C1'
            [2.992, 33.291, -6.608],  # C2'
            [3.963, 34.097, -5.764],  # C3'
            [4.914, 34.654, -6.813],  # C4'
            [6.270, 35.060, -6.315],  # C5'
            [6.792, 34.041, -5.495],  # O5'
            [5.074, 33.531, -7.724],  # O4'
            [3.331, 35.088, -4.967],  # O3'
            [2.107, 34.157, -7.319],  # O2'
            [8.220, 34.129, -4.801],  # P
            [8.332, 35.423, -4.024],  # OP1
            [8.462, 32.883, -4.052],  # OP2
            [4.389, 31.320, -7.187],  # N1
            [0.00, 0.00, 0.00],  # N2
            [0.00, 0.00, 0.00],  # N3
            [0.00, 0.00, 0.00],  # N4
            [0.00, 0.00, 0.00],  # N6
            [0.00, 0.00, 0.00],  # N7
            [0.00, 0.00, 0.00],  # N9
            [0.00, 0.00, 0.00],  # C2
            [0.00, 0.00, 0.00],  # C4
            [0.00, 0.00, 0.00],  # C5
            [0.00, 0.00, 0.00],  # C6
            [0.00, 0.00, 0.00],  # C8
            [0.00, 0.00, 0.00],  # O2
            [0.00, 0.00, 0.00],  # O4
            [0.00, 0.00, 0.00],  # O6
        ]
    )
    na_all_atom_mask = torch.tensor(
        [
            [
                1.0,  # C1'
                1.0,  # C2'
                1.0,  # C3'
                1.0,  # C4'
                1.0,  # C5'
                1.0,  # O5'
                1.0,  # O4'
                1.0,  # O3'
                1.0,  # O2'
                1.0,  # P
                1.0,  # OP1
                1.0,  # OP2
                1.0,  # N1
                0.0,  # N2
                0.0,  # N3
                0.0,  # N4
                0.0,  # N6
                0.0,  # N7
                0.0,  # N9
                0.0,  # C2
                0.0,  # C4
                0.0,  # C5
                0.0,  # C6
                0.0,  # C8
                0.0,  # O2
                0.0,  # O4
                0.0,  # O6
            ]
        ]
    )
    na_atom_deoxy = torch.tensor([0], dtype=torch.bool)
    na_atom_chain_indices = torch.tensor([1], dtype=torch.long)
    na_fixed_residue_mask = torch.tensor([0], dtype=torch.bool)
    batch = {
        "aatype": copy.deepcopy(na_aatype.unsqueeze(0)),
        "all_atom_positions": na_all_atom_positions[None, None, ...].double(),
        "all_atom_mask": na_all_atom_mask.unsqueeze(0).double(),
        "atom_deoxy": na_atom_deoxy.unsqueeze(0),
        "residue_chain_indices": na_atom_chain_indices.unsqueeze(0),
        "fixed_residue_mask": na_fixed_residue_mask.unsqueeze(0),
    }
    batch = atom27_to_torsion_angles(randomly_noise_torsion_atoms_to_place=False)(batch)
    batch["aatype"] = na_aatype.unsqueeze(0)  # restore original NA residue type

    # construct arguments
    torsions_start_index = 0
    torsions_end_index = 8
    num_torsions = torsions_end_index - torsions_start_index
    assert (
        0 <= torsions_start_index < torsions_end_index < batch["torsion_angles_sin_cos"].shape[-2]
    ), "Selected torsion angles must be sequential and located at valid indices."
    torsions = batch["torsion_angles_sin_cos"][
        :, :, torsions_start_index:torsions_end_index, :
    ].reshape(1, -1, num_torsions * 2)
    is_protein_residue_mask = torch.tensor([0.0], dtype=torch.bool).unsqueeze(0)
    is_na_residue_mask = torch.tensor([1.0], dtype=torch.bool).unsqueeze(0)

    # run simulated forward pass
    prot_na_pdb_filepath = run_mock_forward_pass(
        batch=batch,
        torsions=torsions,
        is_protein_residue_mask=is_protein_residue_mask,
        is_na_residue_mask=is_na_residue_mask,
    )
    print(f"PDB file created for input: {prot_na_pdb_filepath}")


if __name__ == "__main__":
    main()
