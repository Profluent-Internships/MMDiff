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
    atom37_to_frames,
    atom37_to_torsion_angles,
)


def run_mock_forward_pass(
    batch: Dict[str, Any],
    torsions: torch.Tensor,
    is_protein_residue_mask: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    output_dir: str = os.getcwd(),
) -> Dict[str, Any]:
    # construct frames
    frame_outputs = atom37_to_frames(batch)
    rigids = ru.Rigid.from_tensor_4x4(frame_outputs["rigidgroups_gt_frames"])
    atom37 = all_atom.compute_backbone(
        bb_rigids=rigids[..., 0],  # note: only frames for backbone group 1 are to be used here
        torsions=torsions,
        is_protein_residue_mask=is_protein_residue_mask,
        is_na_residue_mask=is_na_residue_mask,
        aatype=batch["aatype"],
    )[0]

    # collect unpadded outputs
    unpad_is_protein_residue_mask = is_protein_residue_mask[0].tolist()
    unpad_is_na_residue_mask = is_na_residue_mask[0].tolist()
    unpad_pred_pos = atom37[0].cpu().numpy()
    unpad_complex_restype = batch["aatype"][0].cpu().numpy()
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
    protein_aatype = torch.tensor([0], dtype=torch.long)
    protein_all_atom_positions = torch.tensor(
        [
            # note: derived manually from a random `ALA` residue within `data/PDB-NA-Largest/PDB/yz/1yz9.pdb` as shown above
            [34.918, 0.465, 30.651],  # "N"
            [36.171, 0.074, 30.018],  # "CA"
            [37.264, -0.318, 31.021],  # "C"
            [36.674, 1.211, 29.121],  # "CB"
            [38.029, -1.261, 30.786],  # "O"
            [0.0, 0.0, 0.0],  # "CG"
            [0.0, 0.0, 0.0],  # "CG1"
            [0.0, 0.0, 0.0],  # "CG2"
            [0.0, 0.0, 0.0],  # "OG"
            [0.0, 0.0, 0.0],  # "OG1"
            [0.0, 0.0, 0.0],  # "SG"
            [0.0, 0.0, 0.0],  # "CD"
            [0.0, 0.0, 0.0],  # "CD1"
            [0.0, 0.0, 0.0],  # "CD2"
            [0.0, 0.0, 0.0],  # "ND1"
            [0.0, 0.0, 0.0],  # "ND2"
            [0.0, 0.0, 0.0],  # "OD1"
            [0.0, 0.0, 0.0],  # "OD2"
            [0.0, 0.0, 0.0],  # "SD"
            [0.0, 0.0, 0.0],  # "CE"
            [0.0, 0.0, 0.0],  # "CE1"
            [0.0, 0.0, 0.0],  # "CE2"
            [0.0, 0.0, 0.0],  # "CE3"
            [0.0, 0.0, 0.0],  # "NE"
            [0.0, 0.0, 0.0],  # "NE1"
            [0.0, 0.0, 0.0],  # "NE2"
            [0.0, 0.0, 0.0],  # "OE1"
            [0.0, 0.0, 0.0],  # "OE2"
            [0.0, 0.0, 0.0],  # "CH2"
            [0.0, 0.0, 0.0],  # "NH1"
            [0.0, 0.0, 0.0],  # "NH2"
            [0.0, 0.0, 0.0],  # "OH"
            [0.0, 0.0, 0.0],  # "CZ"
            [0.0, 0.0, 0.0],  # "CZ2"
            [0.0, 0.0, 0.0],  # "CZ3"
            [0.0, 0.0, 0.0],  # "NZ"
            [0.0, 0.0, 0.0],  # "OXT"
        ]
    )
    protein_all_atom_mask = torch.tensor(
        [
            [
                1.0,  # "N"
                1.0,  # "CA"
                1.0,  # "C"
                1.0,  # "CB"
                1.0,  # "O"
                0.0,  # "CG"
                0.0,  # "CG1"
                0.0,  # "CG2"
                0.0,  # "OG"
                0.0,  # "OG1"
                0.0,  # "SG"
                0.0,  # "CD"
                0.0,  # "CD1"
                0.0,  # "CD2"
                0.0,  # "ND1"
                0.0,  # "ND2"
                0.0,  # "OD1"
                0.0,  # "OD2"
                0.0,  # "SD"
                0.0,  # "CE"
                0.0,  # "CE1"
                0.0,  # "CE2"
                0.0,  # "CE3"
                0.0,  # "NE"
                0.0,  # "NE1"
                0.0,  # "NE2"
                0.0,  # "OE1"
                0.0,  # "OE2"
                0.0,  # "CH2"
                0.0,  # "NH1"
                0.0,  # "NH2"
                0.0,  # "OH"
                0.0,  # "CZ"
                0.0,  # "CZ2"
                0.0,  # "CZ3"
                0.0,  # "NZ"
                0.0,  # "OXT"
            ]
        ]
    )
    protein_atom_deoxy = torch.tensor([0], dtype=torch.bool)
    protein_atom_chain_indices = torch.tensor([1], dtype=torch.long)
    protein_fixed_residue_mask = torch.tensor([0], dtype=torch.bool)
    batch = {
        "aatype": copy.deepcopy(protein_aatype.unsqueeze(0)),
        "all_atom_positions": protein_all_atom_positions[None, None, ...].double(),
        "all_atom_mask": protein_all_atom_mask.unsqueeze(0).double(),
        "atom_deoxy": protein_atom_deoxy.unsqueeze(0),
        "residue_chain_indices": protein_atom_chain_indices.unsqueeze(0),
        "fixed_residue_mask": protein_fixed_residue_mask.unsqueeze(0),
    }
    batch = atom37_to_torsion_angles(randomly_noise_torsion_atoms_to_place=False)(batch)

    # construct arguments
    torsions_start_index = 2  # select the psi angle
    torsions_end_index = 3  # select only the psi angle
    num_torsions = torsions_end_index - torsions_start_index
    assert (
        0 <= torsions_start_index < torsions_end_index < batch["torsion_angles_sin_cos"].shape[-2]
    ), "Selected torsion angles must be sequential and located at valid indices."
    torsions = batch["torsion_angles_sin_cos"][
        :, :, torsions_start_index:torsions_end_index, :
    ].reshape(1, -1, num_torsions * 2)
    is_protein_residue_mask = torch.tensor([1.0], dtype=torch.bool).unsqueeze(0)
    is_na_residue_mask = torch.tensor([0.0], dtype=torch.bool).unsqueeze(0)

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
