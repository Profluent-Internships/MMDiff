# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import functools as fn
import os
import random
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tree
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from biopandas.pdb import PandasPdb
from graphein.protein.utils import cast_pdb_column_to_type
from omegaconf import DictConfig, ListConfig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from src import utils
from src.data.components.pdb import complex_constants, data_transforms
from src.data.components.pdb import data_utils as du
from src.data.components.pdb import protein_constants, rigid_utils
from src.data.components.pdb.complex import PDB_CHAIN_IDS
from src.models.components.pdb import metrics
from src.utils.utils import is_integer

SAMPLING_ARGS = Dict[str, Any]
BATCH_TYPE = Dict[str, Any]
INDEX_TYPE = Union[int, np.int16, np.int32, np.int64]
TIMESTEP_TYPE = Union[torch.Tensor, np.ndarray, np.float64, np.int64, float, int]

NUM_PROTEIN_RESIDUE_ATOMS = 14
NUM_NA_RESIDUE_ATOMS = 23
NUM_PROTEIN_ONEHOT_AATYPE_CLASSES = complex_constants.protein_restype_num
NUM_NA_ONEHOT_AATYPE_CLASSES = complex_constants.na_restype_num
NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES = complex_constants.protein_na_restype_num

MOLECULE_TYPE_INDEX_MAPPING = {"A": 0, "D": 1, "R": 2}

log = utils.get_pylogger(__name__)


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y


@beartype
def diffuse_sequence(
    sequence_feats: Dict[
        str, Any
    ],  # a collection of required sequence feature inputs, one being the `aatype` tensor to diffuse
    t: TIMESTEP_TYPE,  # note: should be a float value in range [0.0, 1.0)
    min_t: TIMESTEP_TYPE,  # note: should be a float value in range [0.0, 1.0)
    num_t: TIMESTEP_TYPE,  # note: should be an integer value
    random_seed: Optional[
        int
    ],  # note: should correspond to the same random seed used within one's random number generator (RNG) for training and sampling (specifically for SE(3) frame noising)
    sequence_ddpm: Any,  # note: should correspond e.g., to an instance of `GaussianSequenceDiffusion`
    training: bool,  # whether or not a model is currently performing a training epoch
    eps: Optional[
        torch.Tensor
    ] = None,  # the sequence noise to apply to the input `sequence_feats['aatype']`
    t_discrete_jump: Optional[
        TIMESTEP_TYPE
    ] = None,  # an integer-like value that determines how many discrete sequence diffusion timesteps to jump forward (e.g., +1) or backward (e.g., -1)
    onehot_sequence_input: bool = False,  # note: if `True`, sequence inputs (i.e., `sequence_feats['aatype']`) are expected to be of shape `[batch_size, num_nodes, num_node_types]`
    noise_scale: float = 1.0,
) -> Tuple[
    Dict[str, np.ndarray], torch.Tensor
]:  # note: returns both the diffused (one-hot) sequence as well as the noise with which it was diffused
    # Diffuse sequence after encoding it as a collection of zero-centered one-hot vectors.
    assert all(
        [
            key in sequence_feats
            for key in ["is_protein_residue_mask", "is_na_residue_mask", "fixed_mask", "aatype"]
        ]
    ), "Original sequences as well as masks identifying which residues are protein residues, nucleic acid residues, or fixed residues must all be provided."
    assert all(
        [
            len(sequence_feats[key].shape) == 2
            for key in ["is_protein_residue_mask", "is_na_residue_mask", "fixed_mask"]
        ]
    ), "All sequence feature masks must contain two dimensions, one for batch elements and one for nodes (i.e., residues)."
    if eps is not None:
        assert (
            len(eps.shape) == 3
        ), "The provided sequence noise (i.e., `eps`) must contain three dimensions, one for batch elements, one for nodes (i.e., residues), and one for each residue type class."
    if onehot_sequence_input:
        assert (
            len(sequence_feats["aatype"].shape) == 3
        ), "Sequence input was not provided in one-hot format as anticipated."
    else:
        assert (
            len(sequence_feats["aatype"].shape) == 2
        ), "Sequence input was not provided in vocabulary index format as anticipated."

    if training:
        random_seed = int(time.time())
        # Note: PyTorch does not allow you to pass `None` as an argument to denote a random seed.
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    elif random_seed is not None:
        # Use a fixed seed for evaluation.
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    is_protein_residue_mask = sequence_feats["is_protein_residue_mask"].bool()
    is_na_residue_mask = sequence_feats["is_na_residue_mask"].bool()
    device = is_protein_residue_mask.device

    # note: when nucleic acid sequence inputs are present, it is expected that they are already in `aatype9` format
    protein_inputs_present = is_protein_residue_mask.any().item()
    na_inputs_present = is_na_residue_mask.any().item()

    t_bins = torch.linspace(start=min_t, end=1.0, steps=num_t, device=device)
    # ensure discretized `t` is zero-indexed while handling for both +1 and -1 jumps for all valid `t` (during training)
    t_discretized = (
        (torch.bucketize(t.clamp(min=0.02, max=0.99), t_bins, right=True) - 1).clamp(min=0)
        if training
        else (torch.bucketize(t, t_bins, right=True) - 1).clamp(min=0)
    )
    # note: `t_discrete_jump` can be either positive or negative, which will either increment or decrement `t_discretized`
    t_discretized = t_discretized + t_discrete_jump if t_discrete_jump else t_discretized
    assert (t_discretized >= 0).all(), "Discretized `t` cannot be less than zero."
    assert (
        t_discretized < len(t_bins)
    ).all(), "Discretized `t` cannot be greater than the number of discretized bins for `t`."
    fixed_sequence_mask = sequence_feats["fixed_mask"].bool()
    if onehot_sequence_input:
        aatype = sequence_feats["aatype"] if training else sequence_feats["aatype"].clone()
    else:
        # break reference to the input sequence when gradients to it do not need to be preserved
        aatype = sequence_feats["aatype"].clone()
    diffused_onehot_aatype = (
        aatype
        if onehot_sequence_input
        else (
            2
            * torch.nn.functional.one_hot(aatype, num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES)
        )
        - 1
    )
    if protein_inputs_present and not onehot_sequence_input:
        # Diffuse protein sequence separately.
        aatype[is_protein_residue_mask] = sequence_feats["aatype"][is_protein_residue_mask]
        diffused_onehot_aatype[..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES][
            is_protein_residue_mask
        ] = (
            2
            * torch.nn.functional.one_hot(
                aatype[is_protein_residue_mask],
                num_classes=NUM_PROTEIN_ONEHOT_AATYPE_CLASSES,
            )
        ) - 1
    if na_inputs_present and not onehot_sequence_input:
        na_residue_type_adjustment = 0
        if (
            sequence_feats["aatype"][is_na_residue_mask] >= NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
        ).any():
            na_residue_type_adjustment = NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
        # Diffuse nucleic acid sequence separately.
        aatype[is_na_residue_mask] = (
            # Note: This subtraction ensures nucleic acid types start at `0` when being based on ground-truth values
            sequence_feats["aatype"][is_na_residue_mask]
            - na_residue_type_adjustment
        )
        diffused_onehot_aatype[..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:][is_na_residue_mask] = (
            2
            * torch.nn.functional.one_hot(
                aatype[is_na_residue_mask],
                num_classes=NUM_NA_ONEHOT_AATYPE_CLASSES,
            )
        ) - 1
    # Diffuse sequence tokens.
    diffused_onehot_aatype, eps = sequence_ddpm.q_sample(
        diffused_onehot_aatype.float(),
        t_discretized,
        eps=eps,
        mask=fixed_sequence_mask,
        noise_scale=noise_scale,
    )
    if protein_inputs_present and not onehot_sequence_input:
        # Ensure that diffused protein sequences assign minimum likelihood to nucleic acid residues.
        min_protein_onehot_aatype_logit = diffused_onehot_aatype[
            ..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES
        ][is_protein_residue_mask].min()
        diffused_onehot_aatype[..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:][
            is_protein_residue_mask
        ] = (min_protein_onehot_aatype_logit - 0.1)
    if na_inputs_present and not onehot_sequence_input:
        # Ensure that diffused nucleic acid sequences assign minimum likelihood to protein residues.
        min_na_onehot_aatype_logit = diffused_onehot_aatype[
            ..., NUM_PROTEIN_ONEHOT_AATYPE_CLASSES:
        ][is_na_residue_mask].min()
        diffused_onehot_aatype[..., :NUM_PROTEIN_ONEHOT_AATYPE_CLASSES][is_na_residue_mask] = (
            min_na_onehot_aatype_logit - 0.1
        )
    # Determine the most likely tokens.
    diffused_aatype = torch.argmax(diffused_onehot_aatype, axis=-1)
    if protein_inputs_present and not onehot_sequence_input:
        assert (
            diffused_aatype[is_protein_residue_mask].min() >= 0
            and diffused_aatype[is_protein_residue_mask].max() <= 20
        ), "Diffused protein sequences must contain only protein vocabulary tokens."
    if na_inputs_present and not onehot_sequence_input:
        assert (
            diffused_aatype[is_na_residue_mask].min() >= 21
            and diffused_aatype[is_na_residue_mask].max() <= 29
        ), "Diffused nucleic acid sequences must contain only protein vocabulary tokens."

    diffused_atom_deoxy = is_na_residue_mask & (diffused_aatype >= 21) & (diffused_aatype <= 24)
    if na_inputs_present and not onehot_sequence_input:
        na_aatype = diffused_aatype[is_na_residue_mask]
        diffused_na_deoxy = diffused_atom_deoxy[is_na_residue_mask]
        diffused_aatype[is_na_residue_mask] = (
            data_transforms.convert_na_aatype9_to_aatype6(
                na_aatype, deoxy_offset_mask=diffused_na_deoxy
            )
            + na_residue_type_adjustment
        )
    diffused_sequence = {
        "diffused_aatype": diffused_aatype,  # note: for nucleic acids, should be returned in `aatype6` format
        "diffused_onehot_aatype": diffused_onehot_aatype,
        "diffused_atom_deoxy": diffused_atom_deoxy,
    }

    return diffused_sequence, eps


class TrainSampler(Sampler):
    def __init__(
        self,
        data_conf,
        dataset,
        batch_size,
        sample_mode,
    ):
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        if self._sample_mode in ["cluster_length_batch", "cluster_time_batch"]:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            log.info(f"Read {self._max_cluster} clusters.")
            self._missing_pdbs = 0

            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]

            self._data_csv["cluster"] = self._data_csv["pdb_name"].map(cluster_lookup)
            num_clusters = len(set(self._data_csv["cluster"]))
            self.sampler_len = num_clusters * self._batch_size
            log.info(
                f"Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}"
            )

    @beartype
    def _read_clusters(self) -> Dict[str, int]:
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path) as f:
            for i, line in enumerate(f):
                for chain in line.split(" "):
                    pdb = chain.split("_")[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == "length_batch":
            # Each batch contains multiple macromolecules of the same length.
            sampled_order = self._data_csv.groupby("modeled_seq_len").sample(
                self._batch_size, replace=True, random_state=self.epoch
            )
            return iter(sampled_order["index"].tolist())
        elif self._sample_mode == "time_batch":
            # Each batch contains multiple time steps of the same macromolecules.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == "cluster_length_batch":
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby("cluster").sample(1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby("modeled_seq_len").sample(
                self._batch_size, replace=True, random_state=self.epoch
            )
            return iter(sampled_order["index"].tolist())
        elif self._sample_mode == "cluster_time_batch":
            # Each batch contains multiple time steps of a macromolecule from a cluster.
            sampled_clusters = self._data_csv.groupby("cluster").sample(1, random_state=self.epoch)
            dataset_indices = sampled_clusters["index"].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len


class PDBNADataset(Dataset):
    def __init__(
        self,
        data_conf,
        ddpm,
        is_training: bool,
        filter_eval_split: bool = False,
        inference_cfg: Optional[DictConfig] = None,
        sequence_ddpm: Optional[Any] = None,
    ):
        self._data_conf = data_conf
        self._ddpm = ddpm
        self._is_training = is_training
        self._filter_eval_split = filter_eval_split
        self._inference_cfg = inference_cfg
        self._sequence_ddpm = sequence_ddpm
        self._init_metadata()

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def ddpm(self):
        return self._ddpm

    @property
    def is_training(self):
        return self._is_training

    @property
    def filter_eval_split(self):
        return self._filter_eval_split

    @property
    def inference_cfg(self):
        return self._inference_cfg

    @property
    def sequence_ddpm(self):
        return self._sequence_ddpm

    @staticmethod
    @beartype
    def _filter_csv_based_on_holdout(
        csv_df: pd.DataFrame, annot_df: pd.DataFrame, holdout: List[str]
    ) -> pd.DataFrame:
        # Holdout complexes containing proteins belonging to one of the specified `holdout` Pfam families
        assert (
            "pfam_hmm_name" in annot_df.columns
        ), "Pfam annotations must be pre-generated and attached to the metadata CSV to holdout examples based on Pfam families."
        annot_df.fillna({"pfam_hmm_name": ""}, inplace=True)
        holdout_annot_df = annot_df[
            annot_df["pfam_hmm_name"].str.lower().str.contains("|".join(holdout).lower())
        ]
        holdout_complex_pdb_codes = list(set(holdout_annot_df["pdb"]))
        complex_in_holdout = csv_df["pdb_name"].isin(holdout_complex_pdb_codes)
        filtered_df = csv_df[~complex_in_holdout]
        assert not any(
            [code in filtered_df.pdb_name.unique().tolist() for code in holdout_complex_pdb_codes]
        ), "Complexes associated with holdout PDB codes must be removed from the dataset."
        return filtered_df

    @staticmethod
    @beartype
    def _select_pdb_by_criterion(
        pdb: PandasPdb,
        field: str,
        field_values: List[Any],
    ) -> PandasPdb:
        """Filter a PDB using a field selection.

        :param pdb: The PDB object to filter by a field.
        :type pdb: PandasPdb
        :param field: The field by which to filter the PDB.
        :type field: str
        :param field_values: The field values by which to filter
            the PDB.
        :type field_values: List[Any]

        :return: The filtered PDB object.
        :rtype: PandasPdb
        """
        for key in pdb.df:
            if field in pdb.df[key]:
                filtered_pdb = pdb.df[key][pdb.df[key][field].isin(field_values)]
                if "ATOM" in key and len(filtered_pdb) == 0:
                    log.warning(
                        "DataFrame for input PDB does not contain any standard atoms after filtering"
                    )
                pdb.df[key] = filtered_pdb
        return pdb

    @staticmethod
    @beartype
    def _structurally_cluster_and_sample_examples(
        csv_df: pd.DataFrame,
        qtmclust_exec_path: str,
        atom_df_name: str = "ATOM",
        output_file_extension: str = ".pdb",
    ) -> pd.DataFrame:
        assert os.path.exists(
            qtmclust_exec_path
        ), "To structurally cluster examples, a valid qTMclust executable path must be provided."
        warnings.filterwarnings(
            "ignore", message="Column model_id is not an expected column and will be skipped."
        )
        with tempfile.TemporaryDirectory(prefix="structure_clustering_temp_dir_") as temp_dir:
            # record a list of input PDB chains
            chain_list_filepath = os.path.join(temp_dir, "chain_list")
            with open(chain_list_filepath, "w") as f:
                # create a temporary directory of all input PDBs containing only the chains that were selected during dataset preprocessing
                pbar = tqdm(enumerate(csv_df.itertuples(index=False)))
                for row_index, row in pbar:
                    pbar.set_description(
                        f"Processing row #{row_index} for structure-based training dataset clustering"
                    )
                    processed_feats = du.read_pkl(row[1])
                    chain_ids_to_select = [
                        du.chain_int_to_str(id)
                        for id in np.unique(processed_feats["atom_chain_indices"])
                    ]
                    pdb = PandasPdb().read_pdb(row[2]).get_models([1])
                    # work around int-typing bug for `model_id` within version `0.5.0.dev0` of BioPandas -> appears when calling `to_pdb()`
                    cast_pdb_column_to_type(pdb, column_name="model_id", type=str)
                    filtered_pdb = PDBNADataset._select_pdb_by_criterion(
                        pdb, "chain_id", chain_ids_to_select
                    )
                    if len(filtered_pdb.df[atom_df_name]) == 0:
                        # skip empty examples
                        log.info(
                            f"Skipping example #{row_index} as no atoms were contained in it after chain-filtering..."
                        )
                        continue
                    filtered_pdb_path = os.path.join(
                        temp_dir, Path(row[2]).stem + output_file_extension
                    )
                    filtered_pdb.to_pdb(filtered_pdb_path)
                    # record the name of each PDB file in a temporary text file input
                    sample_name_without_extension = os.path.basename(
                        os.path.splitext(filtered_pdb_path)[0]
                    )
                    sample_name_postfix = "" if row_index == (len(csv_df) - 1) else "\n"
                    f.write(f"{sample_name_without_extension}{sample_name_postfix}")
                output_cluster_filepath = os.path.join(temp_dir, "cluster.txt")
            # structurally cluster all input PDB complexes
            complex_output_clusters_df = metrics.run_qtmclust(
                chain_dir=temp_dir,
                chain_list_filepath=chain_list_filepath,
                qtmclust_exec_path=qtmclust_exec_path,
                output_cluster_filepath=output_cluster_filepath,
                tm_cluster_threshold=0.5,  # note: clusters two chains if their TM-score is 0.5 or greater
                chain_ter_mode=0,  # note: reads all chains
                chain_split_mode=0,  # note: parses all chains as a single chain
            )
        # filter the dataset down to only the structural cluster representatives
        rep_names = complex_output_clusters_df.loc[:, 0].unique()
        csv_df["raw_pdb_names"] = csv_df["raw_path"].apply(lambda x: os.path.basename(x))
        csv_df = csv_df[csv_df.raw_pdb_names.isin(rep_names)]
        return csv_df

    @beartype
    def _init_metadata(self, columns_to_plot: Optional[List[str]] = None):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        # Note: Oftentimes, `Mdtraj` cannot parse PDB files comprised of nucleic acid molecules, so in such cases we need to impute its feature values
        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )
        if "oligomeric_detail" not in pdb_csv.columns:
            pdb_csv["oligomeric_detail"] = np.nan
        self.raw_csv = pdb_csv
        self.annot_csv = pd.read_csv(self.data_conf.annot_path)
        if (
            filter_conf.mmcif_allowed_oligomer is not None
            and len(filter_conf.mmcif_allowed_oligomer) > 0
        ):
            # filter complexes derived from mmCIF files
            assert isinstance(filter_conf.mmcif_allowed_oligomer, ListConfig)
            pdb_csv = pdb_csv[
                (
                    pdb_csv.raw_path.str.lower().str.contains("mmcif")
                    & pdb_csv.oligomeric_detail.isin(filter_conf.mmcif_allowed_oligomer)
                )
                | ~pdb_csv.raw_path.str.lower().str.contains("mmcif")
            ]
        if (
            filter_conf.pdb_allowed_oligomer is not None
            and len(filter_conf.pdb_allowed_oligomer) > 0
        ):
            # filter complexes derived from PDB files
            assert isinstance(filter_conf.pdb_allowed_oligomer, ListConfig)
            pdb_csv = pdb_csv[
                (
                    ~pdb_csv.raw_path.str.lower().str.contains("mmcif")
                    & pdb_csv.oligomeric_detail.isin(filter_conf.pdb_allowed_oligomer)
                )
                | pdb_csv.raw_path.str.lower().str.contains("mmcif")
            ]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile is not None and filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, filter_conf.rog_quantile, np.arange(filter_conf.max_len)
            )
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(lambda x: prot_rog_low_pass[x - 1])
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[: filter_conf.subset]
        if filter_conf.allowed_molecule_types is not None:
            if filter_conf.allowed_molecule_types == ["protein", "na"]:
                pass  # Note: This option implies no filtering based on molecule types
            elif filter_conf.allowed_molecule_types == ["protein"]:
                # Note: This option implies that no nucleic acid (NA) chains may be present in the dataset
                pdb_csv = pdb_csv[pdb_csv.num_na_chains == 0]
            elif filter_conf.allowed_molecule_types == ["na"]:
                # Note: This option implies that no protein chains may be present in the dataset
                pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0]
            else:
                raise Exception(
                    "Allowed molecule types must be in `[[protein, na], [protein], [na], null]`"
                )
        if filter_conf.holdout is not None:
            pdb_csv = self._filter_csv_based_on_holdout(
                csv_df=pdb_csv, annot_df=self.annot_csv, holdout=filter_conf.holdout
            )
        if self.data_conf.cluster_examples_by_structure:
            # Note: For a dataset of size ~4,000 examples, structure-based clustering will add ~20 minutes of overhead to the initial data filtering
            pdb_csv = self._structurally_cluster_and_sample_examples(
                pdb_csv, self.data_conf.qtmclust_exec_path
            )
        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
        if columns_to_plot is not None:
            assert len(
                [column for column in columns_to_plot if len(column.strip()) > 0]
            ), "Number of columns provided to plot must be non-zero."
            assert all(
                [column in pdb_csv.columns for column in columns_to_plot]
            ), f"To plot columns {columns_to_plot}, each must be present in `pdb_csv` beforehand."
            sns.pairplot(data=pdb_csv, vars=columns_to_plot)
            plt.savefig(f"{'_'.join(columns_to_plot)}_pairplot.png")

            if self.is_training and len(columns_to_plot) == 1:
                # plot dataset analysis figures one-by-one
                cmap = sns.color_palette("viridis", as_cmap=True)
                plot = sns.histplot(
                    data=pdb_csv, x=columns_to_plot[0], kde=True, hue="num_chains", palette=cmap
                )
                plot.set_title(f"Distribution of `{columns_to_plot[0]}`")
                plot.get_legend().set_title("Number of Chains")
                plt.savefig(f"{columns_to_plot[0]}_histplot.png")
        self._create_split(pdb_csv)

    @beartype
    def _create_split(self, pdb_csv: pd.DataFrame):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            log.info(f"Training: {len(self.csv)} examples")
        else:
            if self.filter_eval_split:
                assert (
                    self.inference_cfg is not None
                ), "Inference config must be provided to filter evaluation split."
                pdb_csv = pdb_csv[
                    (pdb_csv.num_chains >= self.inference_cfg.samples.min_num_chains)
                    & (pdb_csv.num_chains <= self.inference_cfg.samples.max_num_chains)
                ]
                pdb_csv = pdb_csv[
                    (pdb_csv.modeled_seq_len >= self.inference_cfg.samples.min_length)
                    & (pdb_csv.modeled_seq_len <= self.inference_cfg.samples.max_length)
                ]
            pdb_csv = pdb_csv.sort_values(
                ["modeled_na_seq_len", "modeled_protein_seq_len"], ascending=True
            )
            pdb_csv["joint_modeled_seq_len"] = pdb_csv.apply(
                lambda row: f"{row['modeled_na_seq_len']}_{row['modeled_protein_seq_len']}", axis=1
            )
            all_joint_lengths = pdb_csv.joint_modeled_seq_len.unique()
            joint_length_indices = (len(all_joint_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths
            )
            joint_length_indices = joint_length_indices.astype(int)
            eval_joint_lengths = all_joint_lengths[joint_length_indices]
            eval_csv = pdb_csv[pdb_csv.joint_modeled_seq_len.isin(eval_joint_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("joint_modeled_seq_len").sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values(
                ["modeled_na_seq_len", "modeled_protein_seq_len"], ascending=False
            )
            self.csv = eval_csv
            log.info(f"Validation: {len(self.csv)} examples with lengths {eval_joint_lengths}")

    # cache make the same sample in same batch
    @fn.lru_cache(maxsize=100)
    @beartype
    def _process_csv_row(
        self,
        processed_file_path: str,
        t: Optional[TIMESTEP_TYPE] = None,
        random_seed: Optional[int] = None,
    ) -> BATCH_TYPE:
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_complex_feats(processed_feats)

        # Designate which residues to diffuse and which to fix. By default, diffuse all residues.
        diffused_mask = np.ones_like(processed_feats["bb_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        processed_feats["fixed_mask"] = fixed_mask

        # Distinguish between protein residues and nucleic acid residues using corresponding masks.
        processed_feats["is_protein_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 0] == 1
        )
        processed_feats["is_na_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 1] == 1
        ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
        protein_inputs_present = processed_feats["is_protein_residue_mask"].any().item()
        na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

        if self.is_training and self._data_conf.diffuse_sequence:
            sequence_feats = {
                "is_protein_residue_mask": torch.from_numpy(
                    processed_feats["is_protein_residue_mask"]
                ).unsqueeze(0),
                "is_na_residue_mask": torch.from_numpy(
                    processed_feats["is_na_residue_mask"]
                ).unsqueeze(0),
                "fixed_mask": torch.from_numpy(processed_feats["fixed_mask"]).unsqueeze(0),
                "aatype": torch.from_numpy(processed_feats["aatype"]).unsqueeze(0).clone(),
            }
            sequence_feats["aatype"][
                sequence_feats["is_na_residue_mask"]
            ] = data_transforms.convert_na_aatype6_to_aatype9(
                sequence_feats["aatype"][sequence_feats["is_na_residue_mask"]],
                deoxy_offset_mask=processed_feats["atom_deoxy"][None, ...][
                    sequence_feats["is_na_residue_mask"]
                ],
                return_na_within_original_range=True,
            )
            diffused_sequence, eps = diffuse_sequence(
                sequence_feats=copy.deepcopy(sequence_feats),
                t=torch.tensor([t]),
                min_t=self._data_conf.min_t,
                num_t=self._data_conf.num_sequence_t,
                random_seed=random_seed,
                sequence_ddpm=self.sequence_ddpm,
                training=self.is_training,
            )
            diffused_sequence = {
                key: value.squeeze(0).numpy() for key, value in diffused_sequence.items()
            }

            sc_diffused_sequence, _ = diffuse_sequence(
                sequence_feats=copy.deepcopy(sequence_feats),
                t=torch.tensor([t]),
                min_t=self._data_conf.min_t,
                num_t=self._data_conf.num_sequence_t,
                random_seed=random_seed,
                sequence_ddpm=self.sequence_ddpm,
                training=self.is_training,
                eps=eps,
                t_discrete_jump=1,
            )
            sc_diffused_sequence = {
                key: value.squeeze(0).numpy() for key, value in sc_diffused_sequence.items()
            }
            processed_feats["diffused_aatype_eps"] = eps.squeeze(
                0
            ).numpy()  # cache sequence noise for use in the model's KL-divergence loss

        # Find interfaces.
        inter_chain_interacting_residue_mask = torch.zeros(len(diffused_mask), dtype=torch.bool)
        inter_chain_interacting_residue_mask[processed_feats["inter_chain_interacting_idx"]] = True

        # Only take modeled residues.
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats["modeled_idx"]
        if processed_feats["protein_modeled_idx"] is None:
            del processed_feats["protein_modeled_idx"]
        if processed_feats["na_modeled_idx"] is None:
            del processed_feats["na_modeled_idx"]
        processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)
        inter_chain_interacting_residue_mask = inter_chain_interacting_residue_mask[
            min_idx : (max_idx + 1)
        ]

        # Run through OpenFold data transforms.
        chain_feats, protein_chain_feats, na_chain_feats = (
            {
                "aatype": torch.tensor(processed_feats["aatype"]).long(),
                "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
                "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
                "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
            },
            {},
            {},
        )
        if self.is_training and self._data_conf.diffuse_sequence:
            diffused_sequence = tree.map_structure(
                lambda x: x[min_idx : (max_idx + 1)], diffused_sequence
            )
            chain_feats["diffused_aatype"] = torch.tensor(
                diffused_sequence["diffused_aatype"]
            ).long()
            chain_feats["diffused_onehot_aatype"] = torch.tensor(
                diffused_sequence["diffused_onehot_aatype"]
            ).float()
            chain_feats["diffused_atom_deoxy"] = torch.tensor(
                diffused_sequence["diffused_atom_deoxy"]
            ).bool()

            sc_diffused_sequence = tree.map_structure(
                lambda x: x[min_idx : (max_idx + 1)], sc_diffused_sequence
            )
            chain_feats["diffused_aatype_t_plus_1"] = torch.tensor(
                sc_diffused_sequence["diffused_aatype"]
            ).long()
            chain_feats["diffused_onehot_aatype_t_plus_1"] = torch.tensor(
                sc_diffused_sequence["diffused_onehot_aatype"]
            ).float()
            chain_feats["diffused_atom_deoxy_t_plus_1"] = torch.tensor(
                sc_diffused_sequence["diffused_atom_deoxy"]
            ).bool()
            chain_feats["diffused_aatype_eps"] = torch.tensor(
                processed_feats["diffused_aatype_eps"]
            ).float()
        if protein_inputs_present:
            protein_chain_feats = {
                "aatype": chain_feats["aatype"][processed_feats["is_protein_residue_mask"]],
                "all_atom_positions": chain_feats["all_atom_positions"][
                    processed_feats["is_protein_residue_mask"]
                ][:, :NUM_PROTEIN_RESIDUE_ATOMS],
                "all_atom_mask": chain_feats["all_atom_mask"][
                    processed_feats["is_protein_residue_mask"]
                ][:, :NUM_PROTEIN_RESIDUE_ATOMS],
                # note: this should always be empty; will instead be used during complex feature concatenation
                "atom_deoxy": chain_feats["atom_deoxy"][
                    processed_feats["is_protein_residue_mask"]
                ],
            }
            protein_chain_feats["atom14_gt_positions"] = protein_chain_feats[
                "all_atom_positions"
            ]  # cache `atom14` positions
            if self.is_training and self._data_conf.diffuse_sequence:
                protein_chain_feats["diffused_aatype"] = torch.tensor(
                    diffused_sequence["diffused_aatype"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).long()
                protein_chain_feats["diffused_onehot_aatype"] = torch.tensor(
                    diffused_sequence["diffused_onehot_aatype"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).float()
                protein_chain_feats["diffused_atom_deoxy"] = torch.tensor(
                    diffused_sequence["diffused_atom_deoxy"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).bool()

                protein_chain_feats["diffused_aatype_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_aatype"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).long()
                protein_chain_feats["diffused_onehot_aatype_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_onehot_aatype"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).float()
                protein_chain_feats["diffused_atom_deoxy_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_atom_deoxy"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).bool()
                protein_chain_feats["diffused_aatype_eps"] = torch.tensor(
                    processed_feats["diffused_aatype_eps"][
                        processed_feats["is_protein_residue_mask"]
                    ]
                ).float()
        if na_inputs_present:
            na_chain_feats = {
                "aatype": chain_feats["aatype"][processed_feats["is_na_residue_mask"]],
                "all_atom_positions": chain_feats["all_atom_positions"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "all_atom_mask": chain_feats["all_atom_mask"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "atom_deoxy": chain_feats["atom_deoxy"][processed_feats["is_na_residue_mask"]],
            }
            na_chain_feats["atom23_gt_positions"] = na_chain_feats[
                "all_atom_positions"
            ]  # cache `atom23` positions
            if self.is_training and self._data_conf.diffuse_sequence:
                na_chain_feats["diffused_aatype"] = torch.tensor(
                    diffused_sequence["diffused_aatype"][processed_feats["is_na_residue_mask"]]
                ).long()
                na_chain_feats["diffused_onehot_aatype"] = torch.tensor(
                    diffused_sequence["diffused_onehot_aatype"][
                        processed_feats["is_na_residue_mask"]
                    ]
                ).float()
                na_chain_feats["diffused_atom_deoxy"] = torch.tensor(
                    diffused_sequence["diffused_atom_deoxy"][processed_feats["is_na_residue_mask"]]
                ).bool()

                na_chain_feats["diffused_aatype_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_aatype"][processed_feats["is_na_residue_mask"]]
                ).long()
                na_chain_feats["diffused_onehot_aatype_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_onehot_aatype"][
                        processed_feats["is_na_residue_mask"]
                    ]
                ).float()
                na_chain_feats["diffused_atom_deoxy_t_plus_1"] = torch.tensor(
                    sc_diffused_sequence["diffused_atom_deoxy"][
                        processed_feats["is_na_residue_mask"]
                    ]
                ).bool()
                na_chain_feats["diffused_aatype_eps"] = torch.tensor(
                    processed_feats["diffused_aatype_eps"][processed_feats["is_na_residue_mask"]]
                ).float()

        if protein_inputs_present:
            protein_chain_feats = data_transforms.make_atom14_masks(protein_chain_feats)
            data_transforms.atom14_list_to_atom37_list(
                protein_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
            )
            protein_chain_feats = data_transforms.atom37_to_frames(protein_chain_feats)
            protein_chain_feats = data_transforms.atom37_to_torsion_angles()(protein_chain_feats)
        if na_inputs_present:
            na_chain_feats = data_transforms.make_atom23_masks(na_chain_feats)
            data_transforms.atom23_list_to_atom27_list(
                na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
            )
            na_chain_feats = data_transforms.atom27_to_frames(na_chain_feats)
            na_chain_feats = data_transforms.atom27_to_torsion_angles()(na_chain_feats)

        # Merge available protein and nucleic acid features using padding where necessary
        chain_feats = du.concat_complex_torch_features(
            chain_feats,
            protein_chain_feats,
            na_chain_feats,
            feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
            add_batch_dim=False,
        )

        # Collect chain and residue indices
        chain_idx = processed_feats["atom_chain_indices"]
        asym_id = processed_feats["asym_id"]
        sym_id = processed_feats["sym_id"]
        entity_id = processed_feats["entity_id"]
        res_idx = np.arange(1, len(chain_idx) + 1)  # Start residue indices from `1`

        # To speed up processing, only take necessary features
        final_feats = {
            "aatype": chain_feats["aatype"],
            "chain_idx": chain_idx,
            "asym_id": asym_id,
            "sym_id": sym_id,
            "entity_id": entity_id,
            "residue_index": res_idx,
            "residx_atom23_to_atom37": chain_feats["residx_atom23_to_atom37"],
            "res_mask": processed_feats["bb_mask"],
            "atom37_pos": chain_feats["all_atom_positions"],
            "atom37_mask": chain_feats["all_atom_mask"],
            "atom23_pos": chain_feats["atom23_gt_positions"],
            "atom_deoxy": chain_feats["atom_deoxy"],
            "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
            "molecule_type_encoding": processed_feats["molecule_type_encoding"],
            "is_protein_residue_mask": processed_feats["is_protein_residue_mask"],
            "is_na_residue_mask": processed_feats["is_na_residue_mask"],
            "fixed_mask": processed_feats["fixed_mask"],
            "inter_chain_interacting_residue_mask": inter_chain_interacting_residue_mask,
        }
        if self.is_training and self._data_conf.diffuse_sequence:
            final_feats["diffused_aatype"] = chain_feats["diffused_aatype"]
            final_feats["diffused_onehot_aatype"] = chain_feats["diffused_onehot_aatype"]
            final_feats["diffused_atom_deoxy"] = chain_feats["diffused_atom_deoxy"]
            # Fix ground-truth residue types when scaffolding.
            sequence_fixed_mask = processed_feats["fixed_mask"].astype(bool)
            final_feats["diffused_aatype"][sequence_fixed_mask] = final_feats["aatype"][
                sequence_fixed_mask
            ]
            final_feats["diffused_onehot_aatype"][sequence_fixed_mask] = (
                (
                    2
                    * torch.nn.functional.one_hot(
                        final_feats["aatype"][sequence_fixed_mask],
                        num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
                    )
                )
                - 1
            ).float()
            final_feats["diffused_atom_deoxy"][sequence_fixed_mask] = final_feats["atom_deoxy"][
                sequence_fixed_mask
            ]

            final_feats["diffused_aatype_t_plus_1"] = chain_feats["diffused_aatype_t_plus_1"]
            final_feats["diffused_onehot_aatype_t_plus_1"] = chain_feats[
                "diffused_onehot_aatype_t_plus_1"
            ]
            final_feats["diffused_atom_deoxy_t_plus_1"] = chain_feats[
                "diffused_atom_deoxy_t_plus_1"
            ]
            # Fix ground-truth residue types when scaffolding and self-conditioning.
            final_feats["diffused_aatype_t_plus_1"][sequence_fixed_mask] = final_feats["aatype"][
                sequence_fixed_mask
            ]
            final_feats["diffused_onehot_aatype_t_plus_1"][sequence_fixed_mask] = (
                (
                    2
                    * torch.nn.functional.one_hot(
                        final_feats["aatype"][sequence_fixed_mask],
                        num_classes=NUM_PROTEIN_NA_ONEHOT_AATYPE_CLASSES,
                    )
                )
                - 1
            ).float()
            final_feats["diffused_atom_deoxy_t_plus_1"][sequence_fixed_mask] = final_feats[
                "atom_deoxy"
            ][sequence_fixed_mask]
            final_feats["diffused_aatype_eps"] = chain_feats["diffused_aatype_eps"]

        return final_feats

    @beartype
    def _create_diffused_masks(self, atom37_pos: np.ndarray, rng: Any, row: Any) -> np.ndarray:
        bb_pos = atom37_pos[:, protein_constants.atom_order["CA"]]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min,
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min, high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(f"Unable to generate diffusion mask for {row}")
        return diff_mask

    def __len__(self):
        return len(self.csv)

    @beartype
    def convert_dict_float64_items_to_float32(self, dictionary: BATCH_TYPE) -> BATCH_TYPE:
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                converted_dict[key] = value  # For non-NumPy array and non-PyTorch tensor types
        return converted_dict

    @beartype
    def __getitem__(self, idx: INDEX_TYPE) -> Union[BATCH_TYPE, Tuple[BATCH_TYPE, str, str]]:
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if "pdb_name" in csv_row:
            pdb_name = csv_row["pdb_name"]
        elif "chain_name" in csv_row:
            pdb_name = csv_row["chain_name"]
        else:
            raise ValueError("Need chain identifier.")

        # Sample `t`, and use a fixed seed for evaluation.
        if self.is_training:
            random_seed = None
            rng = np.random.default_rng(random_seed)
            t = rng.uniform(self._data_conf.min_t, 1.0)
        else:
            random_seed = idx
            rng = np.random.default_rng(random_seed)
            t = 1.0

        pdb_file_path = csv_row["raw_path"]
        processed_file_path = csv_row["processed_path"]
        chain_feats = self._process_csv_row(processed_file_path, t=t, random_seed=random_seed)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, 0]
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_pos_t"] = torch.zeros_like(gt_bb_rigid.get_trans())
        if self.is_training and self._data_conf.diffuse_sequence:
            # TODO: investigate ability to bias the model's generative sequence distribution by initializing `sc_aatype_t` in a specific way
            chain_feats["sc_aatype_t"] = torch.zeros_like(chain_feats["diffused_onehot_aatype"])

        # Diffuse structure according to `t`.
        if self.is_training:
            diff_feats_t = self._ddpm.forward_marginal(
                rigids_0=gt_bb_rigid, t=t, diffuse_mask=None
            )
        else:
            diff_feats_t = self.ddpm.sample_ref(
                num_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_seq_len"])
        final_feats = self.convert_dict_float64_items_to_float32(final_feats)
        final_feats["molecule_type_encoding"] = final_feats["molecule_type_encoding"].float()
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name, pdb_file_path


class SamplingDataset(Dataset):
    def __init__(
        self,
        min_length: int,
        max_length: int,
        length_step: int,
        samples_per_length: int,
        residue_molecule_type_mappings: List[str],
        residue_chain_mappings: List[str],
        eval_dataset: Optional[Any] = None,
    ):
        self.samples = []
        if eval_dataset is None:
            sample_lengths = list(range(min_length, max_length + 1, length_step))
            assert (
                len(sample_lengths)
                == len(residue_molecule_type_mappings)
                == len(residue_chain_mappings)
            ), "Number of inputs (per sample) must match number of requested samples."
            for length_index, length in enumerate(sample_lengths):
                sample_molecule_type_encoding = self.construct_molecule_type_encoding_from_mapping(
                    molecule_type_mapping=residue_molecule_type_mappings[length_index],
                    target_length=length,
                    length_index=length_index,
                )
                (
                    sample_is_protein_residue_mask,
                    sample_is_na_residue_mask,
                ) = self.construct_is_residue_masks_from_molecule_type_encoding(
                    molecule_type_encoding=sample_molecule_type_encoding
                )
                (
                    sample_asym_id,
                    sample_sym_id,
                    sample_entity_id,
                    sample_chain_idx,
                ) = self.construct_chain_masks_from_chain_mapping(
                    chain_mapping=residue_chain_mappings[length_index],
                    target_length=length,
                    length_index=length_index,
                )
                for sample_length_index in range(samples_per_length):
                    sample = {
                        "sample_length": length,
                        "sample_length_index": sample_length_index,
                        "sample_chain_idx": sample_chain_idx,
                        "sample_molecule_type_encoding": sample_molecule_type_encoding.astype(
                            np.float32
                        ),
                        "sample_is_protein_residue_mask": sample_is_protein_residue_mask,
                        "sample_is_na_residue_mask": sample_is_na_residue_mask,
                        "sample_asym_id": sample_asym_id,
                        "sample_sym_id": sample_sym_id,
                        "sample_entity_id": sample_entity_id,
                    }
                    self.samples.append(sample)
        else:
            # evaluate sampling using a data-driven chain length and count stratification
            global_row_index = 0
            for sample_group in eval_dataset.csv.groupby("joint_modeled_seq_len"):
                for sample_length_index in range(len(sample_group[-1])):
                    row = eval_dataset.__getitem__(global_row_index)[0]
                    sample = {
                        "sample_length": len(row["aatype"]),
                        "sample_length_index": sample_length_index,
                        "sample_chain_idx": row["chain_idx"],
                        "sample_molecule_type_encoding": row["molecule_type_encoding"],
                        "sample_is_protein_residue_mask": row["is_protein_residue_mask"],
                        "sample_is_na_residue_mask": row["is_na_residue_mask"],
                        "sample_asym_id": row["asym_id"],
                        "sample_sym_id": row["sym_id"],
                        "sample_entity_id": row["entity_id"],
                    }
                    global_row_index += 1
                    self.samples.append(sample)

    @staticmethod
    @beartype
    def construct_molecule_type_encoding_from_mapping(
        molecule_type_mapping: str,
        target_length: int,
        length_index: int,
        molecule_type_index_mapping: Dict[str, int] = MOLECULE_TYPE_INDEX_MAPPING,
    ) -> np.ndarray:
        encodings = []
        for mapping in molecule_type_mapping.split(","):
            mapping_items = mapping.split(":")
            assert (
                len(mapping_items) == 2
            ), "Molecule type mappings must conform to the standard syntax of '`molecule_type`:`residue_count`'."
            molecule_type, molecule_type_residue_count = mapping_items
            assert (
                molecule_type in molecule_type_index_mapping
            ), "Allowed molecule types are amino acids (`A`), deoxyribonucleic acids (`D`), and ribonucleic acids (`R`)."
            assert is_integer(
                molecule_type_residue_count
            ), "Residue count must be specified as an integer value."
            molecule_type_index = molecule_type_index_mapping[molecule_type]
            molecule_type_residue_count = int(molecule_type_residue_count)
            encodings_ = []
            for _ in range(molecule_type_residue_count):
                encoding = [0, 0, 0, 0]
                encoding[molecule_type_index] = 1
                encodings_.append(encoding)
            encodings.append(encodings_)
        molecule_type_encoding = np.concatenate(encodings)
        assert (
            len(molecule_type_encoding) == target_length
        ), f"Error: The annotations provided by the molecule type mapping at index {length_index} do not match the corresponding target sequence length of {target_length}. Please re-run with a valid mapping at this index."
        return molecule_type_encoding

    @staticmethod
    @beartype
    def construct_is_residue_masks_from_molecule_type_encoding(
        molecule_type_encoding: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            molecule_type_encoding[:, 0] == 1,
            np.logical_or(molecule_type_encoding[:, 1] == 1, molecule_type_encoding[:, 2] == 1),
        )

    @staticmethod
    @beartype
    def construct_chain_masks_from_chain_mapping(
        chain_mapping: str,
        target_length: int,
        length_index: int,
        valid_pdb_chain_ids: List[str] = PDB_CHAIN_IDS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chain_indices = []
        for mapping in chain_mapping.split(","):
            mapping_items = mapping.split(":")
            assert (
                len(mapping_items) == 2
            ), "Chain identifier (ID) mappings must conform to the standard syntax of '`chain_id`:`residue_count`'."
            chain_id, chain_id_residue_count = mapping_items
            assert (
                chain_id in valid_pdb_chain_ids
            ), "Allowed chain IDs are only the alphanumeric chain IDs permitted by the PDB."
            assert is_integer(
                chain_id_residue_count
            ), "Residue count must be specified as an integer value."
            pdb_chain_index = valid_pdb_chain_ids.index(chain_id)
            chain_id_residue_count = int(chain_id_residue_count)
            chain_indices.append([pdb_chain_index for _ in range(chain_id_residue_count)])
        chain_indices_dict_list = [
            {"chain_indices": chain_indices_list} for chain_indices_list in chain_indices
        ]
        total_num_residues = sum(
            [
                len(chain_indices_dict["chain_indices"])
                for chain_indices_dict in chain_indices_dict_list
            ]
        )
        assert (
            total_num_residues == target_length
        ), f"The annotations provided by the chain mapping at index {length_index} do not match the corresponding target sequence length of {target_length}. Please re-run with a valid mapping at this index."
        du.add_alphafold_multimer_chain_assembly_features(
            chain_feat_dicts=chain_indices_dict_list,
            chain_feat_name="chain_indices",
        )
        asym_ids, sym_ids, entity_ids = [], [], []
        for chain_dict in chain_indices_dict_list:
            asym_ids.append(chain_dict["asym_id"])
            sym_ids.append(chain_dict["sym_id"])
            entity_ids.append(chain_dict["entity_id"])
        asym_id, sym_id, entity_id = (
            np.concatenate(asym_ids, dtype=np.float32),
            np.concatenate(sym_ids, dtype=np.float32),
            np.concatenate(entity_ids, dtype=np.float32),
        )
        chain_idx = np.concatenate(chain_indices, dtype=np.int64)
        return asym_id, sym_id, entity_id, chain_idx

    def __len__(self):
        return len(self.samples)

    @beartype
    def __getitem__(self, index: INDEX_TYPE) -> SAMPLING_ARGS:
        return self.samples[index]
