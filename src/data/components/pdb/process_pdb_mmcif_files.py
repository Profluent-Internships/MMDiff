# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""

import argparse
import collections
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import time

import mdtraj as md
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBIO, MMCIFParser
from tqdm import tqdm

from src import utils
from src.data.components.pdb import data_utils as du
from src.data.components.pdb import errors, mmcif_parsing, parsers, protein_constants

log = utils.get_pylogger(__name__)

# Define the parser
parser = argparse.ArgumentParser(description="mmCIF processing script.")
parser.add_argument("--mmcif_dir", help="Path to directory with mmcif files.", type=str)
parser.add_argument(
    "--max_file_size", help="Max file size.", type=int, default=3000000
)  # Only process files up to 3MB large.
parser.add_argument(
    "--min_file_size", help="Min file size.", type=int, default=1000
)  # Files must be at least 1KB.
parser.add_argument("--max_resolution", help="Max resolution of files.", type=float, default=5.0)
parser.add_argument("--max_len", help="Max length of protein.", type=int, default=3000)
parser.add_argument("--num_processes", help="Number of processes.", type=int, default=100)
parser.add_argument(
    "--write_dir", help="Path to write results to.", type=str, default="./data/PDB/processed_pdb"
)
parser.add_argument(
    "--skip_existing", help="Whether to skip processed files.", action="store_true"
)
parser.add_argument("--debug", help="Turn on for debugging.", action="store_true")
parser.add_argument("--verbose", help="Whether to log everything.", action="store_true")


def _retrieve_mmcif_files(mmcif_dir: str, max_file_size: int, min_file_size: int, debug: bool):
    """Set up all the mmcif files to read."""
    log.info("Gathering mmCIF paths")
    total_num_files = 0
    all_mmcif_paths = []
    for subdir in tqdm(os.listdir(mmcif_dir)):
        mmcif_file_dir = os.path.join(mmcif_dir, subdir)
        if not os.path.isdir(mmcif_file_dir):
            continue
        for mmcif_file in os.listdir(mmcif_file_dir):
            if not mmcif_file.endswith(".cif"):
                continue
            mmcif_path = os.path.join(mmcif_file_dir, mmcif_file)
            total_num_files += 1
            if min_file_size <= os.path.getsize(mmcif_path) <= max_file_size:
                all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 100:
            # Don't process all files for debugging
            break
    log.info(f"Processing {len(all_mmcif_paths)} files our of {total_num_files}")
    return all_mmcif_paths


def process_mmcif(
    mmcif_path: str,
    max_resolution: int,
    max_len: int,
    write_dir: str,
    inter_chain_interact_dist_threshold: float = 7.0,
    skip_existing: bool = False,
):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.
        inter_chain_interact_dist_threshold: Euclidean distance under which
            to classify a pairwise inter-chain residue-atom distance as an interaction.
        skip_existing: Whether to skip processed files.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    metadata = {}
    mmcif_name = os.path.basename(mmcif_path).replace(".cif", "")
    metadata["pdb_name"] = mmcif_name
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f"{mmcif_name}.pkl")
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    metadata["processed_path"] = processed_mmcif_path
    if skip_existing and os.path.exists(metadata["processed_path"]):
        return None
    try:
        with open(mmcif_path) as f:
            parsed_mmcif = mmcif_parsing.parse(file_id=mmcif_name, mmcif_string=f.read())
    except Exception:
        raise errors.FileExistsError(f"Error file do not exist {mmcif_path}")
    metadata["raw_path"] = mmcif_path
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(f"Encountered errors {parsed_mmcif.errors}")
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if "_pdbx_struct_assembly.oligomeric_count" in raw_mmcif:
        raw_olig_count = raw_mmcif["_pdbx_struct_assembly.oligomeric_count"]
        oligomeric_count = ",".join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if "_pdbx_struct_assembly.oligomeric_details" in raw_mmcif:
        raw_olig_detail = raw_mmcif["_pdbx_struct_assembly.oligomeric_details"]
        oligomeric_detail = ",".join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata["oligomeric_count"] = oligomeric_count
    metadata["oligomeric_detail"] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header["resolution"]
    metadata["resolution"] = mmcif_resolution
    metadata["structure_method"] = mmcif_header["structure_method"]
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(f"Too high resolution {mmcif_resolution}")
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(f"Invalid resolution {mmcif_resolution}")

    # Extract all chains
    struct_chains = {chain.id.upper(): chain for chain in parsed_mmcif.structure.get_chains()}

    # Record molecule metadata
    metadata["num_chains"] = len(struct_chains)
    metadata["num_protein_chains"] = len(struct_chains)
    metadata["num_na_chains"] = 0

    # Extract features
    all_seqs = set()
    struct_feats = []
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_index = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain_mmcif(chain, chain_index)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats_mmcif(
            chain_feats=chain_dict,
            molecule_constants=protein_constants,
            molecule_backbone_atom_name="CA",
        )
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"

    # Add assembly features from AlphaFold-Multimer,
    # relying on references to each `chain_dict` object
    # to propagate back to the contents of `struct_feats` below
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_dict in struct_feats:
        seq = tuple(chain_dict["aatype"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_dict)

    new_all_chain_dict = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_dict in enumerate(group_chain_features, start=1):
            new_all_chain_dict[f"{du.int_id_to_str_id(entity_id)}_{sym_id}"] = chain_dict
            seq_length = len(chain_dict["aatype"])
            chain_dict["asym_id"] = chain_id * np.ones(seq_length)
            chain_dict["sym_id"] = sym_id * np.ones(seq_length)
            chain_dict["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    # Concatenate all collected features, refactoring features beforehand as necessary for compatibility with other data types
    for chain_dict in struct_feats:
        chain_dict["atom_chain_id_mask"] = np.array(
            [-1 for _ in range(len(chain_dict["aatype"]))], dtype=np.int64
        )
        chain_dict["atom_chain_indices"] = chain_dict["chain_index"]
        chain_dict["atom_deoxy"] = np.array(
            [False for _ in range(len(chain_dict["aatype"]))], dtype=np.bool_
        )
        chain_dict["atom_b_factors"] = chain_dict["b_factors"]
        chain_dict["molecule_type_encoding"] = np.array(
            [[1, 0, 0, 0] for _ in range(len(chain_dict["aatype"]))], dtype=np.int64
        )
        del chain_dict["residue_index"], chain_dict["chain_index"], chain_dict["b_factors"]
    complex_feats = du.concat_np_features(struct_feats, add_batch_dim=False)

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    protein_aatype = complex_feats[
        "aatype"
    ]  # only protein residues are currently present when parsing mmCIF files
    metadata["seq_len"] = len(complex_aatype)
    metadata["protein_seq_len"] = 0 if protein_aatype is None else len(protein_aatype)
    metadata["na_seq_len"] = 0
    # Note: Residue index `20` corresponds to the missing protein residue type
    modeled_idx = np.where(complex_aatype != 20)[0]
    protein_modeled_idx = None if protein_aatype is None else np.where(protein_aatype != 20)[0]
    na_modeled_idx = None
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError("No modeled residues")
    metadata["modeled_seq_len"] = np.max(modeled_idx) - np.min(modeled_idx) + 1
    metadata["modeled_protein_seq_len"] = (
        0
        if protein_aatype is None
        else np.max(protein_modeled_idx) - np.min(protein_modeled_idx) + 1
    )
    metadata["modeled_na_seq_len"] = 0
    complex_feats["modeled_idx"] = modeled_idx
    complex_feats["protein_modeled_idx"] = protein_modeled_idx
    complex_feats["na_modeled_idx"] = na_modeled_idx
    if complex_aatype.shape[0] > max_len:
        raise errors.LengthError(f"Too long {complex_aatype.shape[0]}")

    # Find all inter-chain interface residues (e.g., <= 7.0 Angstrom from an inter-chain non-hydrogen atom)
    num_atoms_per_res = complex_feats["atom_positions"].shape[1]
    bb_pos = torch.from_numpy(complex_feats["bb_positions"]).unsqueeze(0)
    atom_pos = (
        torch.from_numpy(complex_feats["atom_positions"])
        .unsqueeze(0)
        .flatten(start_dim=1, end_dim=2)
    )
    bb_asym_id = torch.from_numpy(complex_feats["asym_id"]).unsqueeze(0)
    atom_asym_id = torch.repeat_interleave(
        bb_asym_id.unsqueeze(2), num_atoms_per_res, dim=2
    ).flatten(start_dim=1, end_dim=2)
    dist_mat = torch.cdist(bb_pos, atom_pos)
    inter_chain_mask = bb_asym_id.unsqueeze(-1) != atom_asym_id.unsqueeze(-2)
    non_h_mask = torch.ones_like(
        inter_chain_mask
    )  # Note: This assumes that the PDB parsing excluded all `H` atoms
    interacting_res_mask = (
        inter_chain_mask & non_h_mask & (dist_mat <= inter_chain_interact_dist_threshold)
    )
    complex_feats["inter_chain_interacting_idx"] = torch.nonzero(
        interacting_res_mask.squeeze(0), as_tuple=False
    )[..., 0].unique()

    try:
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMCIFParser()
        struct = p.get_structure("", mmcif_path)
        io = PDBIO()
        io.set_structure(struct)
        pdb_path = mmcif_path.replace(".cif", ".pdb")
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_rg = md.compute_rg(traj)
        os.remove(pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        raise errors.DataError(f"Mdtraj failed with error {e}")

    chain_dict["ss"] = pdb_ss[0]
    metadata["coil_percent"] = np.sum(pdb_ss == "C") / metadata["modeled_seq_len"]
    metadata["helix_percent"] = np.sum(pdb_ss == "H") / metadata["modeled_seq_len"]
    metadata["strand_percent"] = np.sum(pdb_ss == "E") / metadata["modeled_seq_len"]

    # Radius of gyration
    metadata["radius_gyration"] = pdb_rg[0]

    # Write features to pickles.
    du.write_pkl(processed_mmcif_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(
    all_mmcif_paths, max_resolution, max_len, write_dir, skip_existing=False, verbose=False
):
    all_metadata = []
    for mmcif_path in all_mmcif_paths:
        try:
            start_time = time.time()
            metadata = process_mmcif(
                mmcif_path, max_resolution, max_len, write_dir, skip_existing=skip_existing
            )
            elapsed_time = time.time() - start_time
            if verbose:
                log.info(f"Finished {mmcif_path} in {elapsed_time:2.2f}s")
            all_metadata.append(metadata)
        except errors.DataError as e:
            if verbose:
                log.info(f"Failed {mmcif_path}: {e}")
    return all_metadata


def process_fn(
    mmcif_path,
    max_resolution=None,
    max_len=None,
    write_dir=None,
    skip_existing=False,
    verbose=False,
):
    try:
        start_time = time.time()
        metadata = process_mmcif(
            mmcif_path, max_resolution, max_len, write_dir, skip_existing=skip_existing
        )
        elapsed_time = time.time() - start_time
        if verbose:
            log.info(f"Finished {mmcif_path} in {elapsed_time:2.2f}s")
        return metadata
    except errors.DataError as e:
        if verbose:
            log.info(f"Failed {mmcif_path}: {e}")


def main(args):
    # Get all mmcif files to read.
    all_mmcif_paths = _retrieve_mmcif_files(
        args.mmcif_dir, args.max_file_size, args.min_file_size, args.debug
    )
    total_num_paths = len(all_mmcif_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = "protein_metadata_debug.csv"
    else:
        metadata_file_name = "protein_metadata.csv"
    metadata_path = os.path.join(write_dir, metadata_file_name)
    log.info(f"Files will be written to {write_dir}")

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_mmcif_paths,
            args.max_resolution,
            args.max_len,
            write_dir,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
    else:
        _process_fn = fn.partial(
            process_fn,
            max_resolution=args.max_resolution,
            max_len=args.max_len,
            write_dir=write_dir,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
        # Uses max number of available cores.
        with mp.Pool() as pool:
            all_metadata = pool.map(_process_fn, all_mmcif_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    log.info(f"Finished processing {succeeded}/{total_num_paths} files")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)
