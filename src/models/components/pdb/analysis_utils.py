# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.data.components.pdb import complex, protein


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def create_full_complex(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    restype=None,
    chain_index=None,
    b_factors=None,
    is_protein_residue_mask=None,
    is_na_residue_mask=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    if restype is None:
        restype = np.zeros(n, dtype=int)
    residue_index = np.arange(n) + 1
    if chain_index is None:
        chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    residue_molecule_type = np.zeros([n, 2], dtype=np.int64)
    if is_protein_residue_mask is not None:
        residue_molecule_type[:, 0][is_protein_residue_mask] = 1
    if is_na_residue_mask is not None:
        residue_molecule_type[:, 1][is_na_residue_mask] = 1
    return complex.Complex(
        atom_positions=atom37,
        restype=restype,
        atom_mask=atom37_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
        residue_molecule_type=residue_molecule_type,
    )


def write_complex_to_pdbs(
    complex_pos: np.ndarray,
    output_filepath: str,
    restype: np.ndarray = None,
    chain_index: np.ndarray = None,
    b_factors=None,
    is_protein_residue_mask=None,
    is_na_residue_mask=None,
    overwrite=False,
    no_indexing=False,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(output_filepath)
        file_name = os.path.basename(output_filepath).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = output_filepath.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = output_filepath
    protein_save_path = str(Path(save_path).parent / ("protein_" + os.path.basename(save_path)))
    na_save_path = str(Path(save_path).parent / ("na_" + os.path.basename(save_path)))
    with open(protein_save_path, "w") as protein_f, open(na_save_path, "w") as na_f, open(
        save_path, "w"
    ) as complex_f:
        if complex_pos.ndim == 4:
            for t, pos37 in enumerate(complex_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                if restype is not None:
                    effective_restype = restype[t] if restype.ndim == 2 else restype
                    assert (
                        effective_restype.ndim == 1
                    ), "When writing multiple complexes to PDB files, only a single sequence may be provided for each complex."
                else:
                    effective_restype = restype
                full_complex = create_full_complex(
                    atom37=pos37,
                    atom37_mask=atom37_mask,
                    restype=effective_restype,
                    chain_index=chain_index,
                    b_factors=b_factors,
                    is_protein_residue_mask=is_protein_residue_mask,
                    is_na_residue_mask=is_na_residue_mask,
                )
                pdb_protein = complex.complex_to_pdb(
                    full_complex, model=t + 1, add_end=False, molecule_type_to_write=0
                )
                pdb_na = complex.complex_to_pdb(
                    full_complex, model=t + 1, add_end=False, molecule_type_to_write=1
                )
                pdb_complex = complex.complex_to_pdb(
                    full_complex, model=t + 1, add_end=False, molecule_type_to_write=-1
                )
                protein_f.write(pdb_protein)
                na_f.write(pdb_na)
                complex_f.write(pdb_complex)
        elif complex_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(complex_pos), axis=-1) > 1e-7
            if restype is not None:
                assert (
                    restype.ndim == 1
                ), "When writing a single complex to a PDB file, only a single sequence may be provided."
            full_complex = create_full_complex(
                atom37=complex_pos,
                atom37_mask=atom37_mask,
                restype=restype,
                chain_index=chain_index,
                b_factors=b_factors,
                is_protein_residue_mask=is_protein_residue_mask,
                is_na_residue_mask=is_na_residue_mask,
            )
            pdb_protein = complex.complex_to_pdb(
                full_complex, model=1, add_end=False, molecule_type_to_write=0
            )
            pdb_na = complex.complex_to_pdb(
                full_complex, model=1, add_end=False, molecule_type_to_write=1
            )
            pdb_complex = complex.complex_to_pdb(
                full_complex, model=1, add_end=False, molecule_type_to_write=-1
            )
            protein_f.write(pdb_protein)
            na_f.write(pdb_na)
            complex_f.write(pdb_complex)
        else:
            raise ValueError(f"Invalid positions shape {complex_pos.shape}")
        protein_f.write("END")
        na_f.write("END")
        complex_f.write("END")
    return protein_save_path, na_save_path, save_path


def rigids_to_se3_vec(frame, scale_factor=1.0):
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec
