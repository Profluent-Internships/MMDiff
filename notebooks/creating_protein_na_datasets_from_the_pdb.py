# %% [markdown]
# # Creating Protein-Nucleic Acid (NA) Datasets from the PDB
# 
# Graphein provides a utility for curating and splitting datasets from the [RCSB PDB](https://www.rcsb.org/).
# 
# 
# Initialising a PDBManager will download PDB Metadata which we can use to make detailed selections of protein complex structures.
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profluent-Internships/MMDiff/blob/main/notebooks/creating_protein_na_datasets_from_the_pdb.ipynb) [![GitHub](https://img.shields.io/badge/-View%20on%20GitHub-181717?logo=github&logoColor=ffffff)](https://github.com/Profluent-Internships/MMDiff/blob/main/notebooks/creating_protein_na_datasets_from_the_pdb.ipynb)

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

unique_id = "08_17_2023"
root_dir = Path(f"/export/share/amorehead/Data/Datasets/PDB-NA/{unique_id}/unzipped_pdbs")
pdb_dir = root_dir / "pdb"

# %% [markdown]
# ## Download Metadata

# %%
from graphein.ml.datasets import PDBManager

pdb_manager = PDBManager(root_dir=str(root_dir))

# %% [markdown]
# ## Make Selections

# %%
import matplotlib.pyplot as plt

MIN_RESOLUTION = 4.5
EXPERIMENT_TYPES = ["NMR", "diffraction", "EM"]
MOLECULE_TYPES_TO_GROUP = ["protein", "na"]
PROTEIN_STANDARD_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
NA_STANDARD_ALPHABET = "ATGCU"
MIN_PROTEIN_LENGTH = 40
MIN_NA_LENGTH = 3
MAX_PROTEIN_LENGTH = 3000
MAX_NA_LENGTH = 3000
MAX_NUM_PROTEIN_CHAINS = 100
MAX_NUM_NA_CHAINS = 100

pdb_manager.reset()
pdb_manager.remove_unavailable_pdbs(update=True)
pdb_manager.resolution_better_than_or_equal_to(MIN_RESOLUTION, update=True)
pdb_manager.experiment_types(types=EXPERIMENT_TYPES, update=True)
pdb_manager.select_complexes_with_grouped_molecule_types(
    molecule_types_to_group=MOLECULE_TYPES_TO_GROUP, update=True
)
pdb_manager.df = pdb_manager.df.loc[
    # select only protein chains or nucleic acid chains that contain standard residue types
    (
        (pdb_manager.df.molecule_type.eq("protein"))
        & (pdb_manager.df.sequence.map(lambda x: set(x).issubset(set(PROTEIN_STANDARD_ALPHABET))))
    )
    | (
        (pdb_manager.df.molecule_type.eq("na"))
        & (pdb_manager.df.sequence.map(lambda x: set(x).issubset(set(NA_STANDARD_ALPHABET))))
    )
]
# filter based on minimum chain length
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("protein")].length,
    bins=100,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MIN_PROTEIN_LENGTH, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Minimum Protein Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('protein')].length < MIN_PROTEIN_LENGTH)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("na")].length,
    bins=250,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MIN_NA_LENGTH, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Minimum Nucleic Acid Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('na')].length < MIN_NA_LENGTH)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
pdb_manager.df = pdb_manager.df[
    # select only protein chains of at least e.g., length 40
    ~((pdb_manager.df.molecule_type.eq("protein")) & (pdb_manager.df.length < MIN_PROTEIN_LENGTH))
]
pdb_manager.df = pdb_manager.df[
    # select only nucleic acid chains of at least e.g., length 10
    ~((pdb_manager.df.molecule_type.eq("na")) & (pdb_manager.df.length < MIN_NA_LENGTH))
]
# filter based on maximum chain length
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("protein")].length,
    bins=100,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MAX_PROTEIN_LENGTH, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Maximum Protein Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('protein')].length > MAX_PROTEIN_LENGTH)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("na")].length,
    bins=250,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MAX_NA_LENGTH, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Maximum Nucleic Acid Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('na')].length > MAX_NA_LENGTH)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
pdb_manager.df = pdb_manager.df[
    # select only protein chains of no greater than e.g., length 256
    ~((pdb_manager.df.molecule_type.eq("protein")) & (pdb_manager.df.length > MAX_PROTEIN_LENGTH))
]
pdb_manager.df = pdb_manager.df[
    # select only nucleic acid chains of no greater than e.g., length 128
    ~((pdb_manager.df.molecule_type.eq("na")) & (pdb_manager.df.length > MAX_NA_LENGTH))
]
# filter based on maximum number of protein and nucleic acid chains in each complex
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("protein")].groupby("pdb").size(),
    bins=100,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MAX_NUM_PROTEIN_CHAINS, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Maximum Protein Chains Per Complex")
plt.xlabel("Number of Protein Chains")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('protein')].groupby('pdb').size() > MAX_NUM_PROTEIN_CHAINS)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.hist(
    pdb_manager.df[pdb_manager.df.molecule_type.eq("na")].groupby("pdb").size(),
    bins=100,
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=MAX_NUM_NA_CHAINS, color="red", linestyle="--", label="Cutoff Length")
plt.title("Distribution of Maximum Nucleic Acid Chains Per Complex")
plt.xlabel("Number of Nucleic Acid Chains")
plt.ylabel("Frequency")
plt.text(
    0.7,
    0.85,
    f"Examples Removed: {sum(pdb_manager.df[pdb_manager.df.molecule_type.eq('na')].groupby('pdb').size() > MAX_NUM_NA_CHAINS)}",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
pdb_manager.df = pdb_manager.df.groupby("pdb").filter(
    lambda x: (x["molecule_type"].eq("protein").sum() <= MAX_NUM_PROTEIN_CHAINS)
    and (x["molecule_type"].eq("na").sum() <= MAX_NUM_NA_CHAINS)
)
# view selected complexes
print(pdb_manager.df)
print(f"Number of unique PDB complexes remaining: {pdb_manager.get_num_unique_pdbs()}")
print(
    f"Number of unique PDB complexes containing only protein chains: {pdb_manager.df.groupby('pdb').filter(lambda x: (x['molecule_type'].eq('protein').sum() >= 1) and (x['molecule_type'].eq('na').sum() == 0)).pdb.unique().tolist().__len__()}"
)
print(
    f"Number of unique PDB complexes containing only nucleic acid chains: {pdb_manager.df.groupby('pdb').filter(lambda x: (x['molecule_type'].eq('protein').sum() == 0) and (x['molecule_type'].eq('na').sum() >= 1)).pdb.unique().tolist().__len__()}"
)
print(
    f"Number of unique PDB complexes containing both protein and nucleic acid chains: {pdb_manager.df.groupby('pdb').filter(lambda x: (x['molecule_type'].eq('protein').sum() >= 1) and (x['molecule_type'].eq('na').sum() >= 1)).pdb.unique().tolist().__len__()}"
)

# %% [markdown]
# ## View Selection Properties

# %%
print("Number of PDB chains: ", pdb_manager.get_num_chains())
print("Number of unique PDB complexes: ", pdb_manager.get_num_unique_pdbs())
print("Longest chain: ", pdb_manager.get_longest_chain())
print("Shortest chain: ", pdb_manager.get_shortest_chain())
print("Best chain resolution: ", pdb_manager.get_best_resolution())
print("Worst chain resolution: ", pdb_manager.get_worst_resolution())
print("Experiment types: ", pdb_manager.get_experiment_types())
print("Molecule types: ", pdb_manager.get_molecule_types())

# %% [markdown]
# ## Export

# %%
pdb_manager.download_pdbs(str(pdb_dir), max_workers=24)
pdb_manager.export_pdbs(pdb_dir=str(pdb_dir), max_num_chains_per_pdb_code=-1, models=[1], filter_for_interface_contacts=True)

# %% [markdown]
# # I/O
# 
# We can write our selections as FASTA files or download and write the relevant PDBs in our selection to disk:
# 
# ## CSV

# %%
import os

import pandas as pd

os.makedirs("tmp/", exist_ok=True)

# write selection to disk
pdb_manager.to_csv("tmp/test.csv")

# read selection from disk
sel = pd.read_csv("tmp/test.csv")

# %% [markdown]
# ## FASTA

# %%
from graphein.protein.utils import read_fasta

# write selection to a fasta file
pdb_manager.to_fasta("tmp/test.fasta")

# load selection from a fasta file
fs = read_fasta("tmp/test.fasta")


