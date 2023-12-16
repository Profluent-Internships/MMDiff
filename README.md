<div align="center">

# Towards Joint Sequence-Structure Generation of Nucleic Acid and Protein Complexes with SE(3)-Discrete Diffusion

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![Checkpoints DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8247932.svg)](https://doi.org/10.5281/zenodo.8247932)

</div>

## Description

Official PyTorch implementation of "Towards Joint Sequence-Structure Generation of Nucleic Acid and Protein Complexes with SE(3)-Discrete Diffusion".

<div align="center">

![Animation of a diffusion model-generated 3D nucleic acid molecule visualized successively](img/Nucleic_Acid_Diffusion.gif)

</div>

<details open><summary><b>Table of contents</b></summary>

- [Creating a Virtual Environment](#virtual-environment-creation)
- [Installing RoseTTAFold2NA](#rf2na-installation)
- [Preparing Datasets](#dataset-preparation)
  - [Preparing Protein Data](#protein-dataset-preparation)
  - [Clustering Protein Data](#protein-dataset-clustering)
  - [Batching Data](#dataset-batching)
  - [Downloading Nucleic Acid Data](#na-dataset-downloading)
  - [Analyzing Nucleic Acid Data](#na-dataset-analysis)
  - [Preparing Nucleic Acid Data](#na-dataset-preparation)
  - [Combining Protein and Nucleic Acid Data](#protein-na-dataset-combination)
- [Training New Models](#model-training)
- [Sampling and Evaluation with Trained Models](#model-sampling-and-eval)
- [Acknowledgements](#acknowledgements)
- [Citations](#citations)

</details>

## How to install <a name="virtual-environment-creation"></a>

Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies

```bash
# clone project
git clone https://github.com/Profluent-Internships/MMDiff
cd MMDiff

# create conda environment
mamba env create -f environment.yaml
conda activate pdb-na-se3-diffusion  # note: one still needs to use `conda` to (de)activate environments

# install local project as package
pip3 install -e .
```

Download pre-trained checkpoints

**Note**: Make sure to be located in the project's root directory beforehand (e.g., `~/MMDiff/`)

```bash
# fetch and extract model checkpoints directory
wget https://zenodo.org/record/8247932/files/MMDiff_Checkpoints.tar.gz
tar -xzf MMDiff_Checkpoints.tar.gz
rm MMDiff_Checkpoints.tar.gz
```

Install US-align and qTMclust to cluster generated structures

```bash
cd $MY_PROGRAMS_DIR  # download US-align to your choice of directory (e.g., `~/Programs/`)
git clone https://github.com/pylelab/USalign.git && cd USalign/ && git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
g++ -static -O3 -ffast-math -lm -o qTMclust qTMclust.cpp
```

**Note**: Make sure to update the `usalign_exec_path` and `qtmclust_exec_path` values in e.g., `configs/paths/default.yaml` to reflect where you have placed the US-align and qTMclust executables on your machine.

## How to install RoseTTAFold2NA to score generated sequences <a name="rf2na-installation"></a>

```bash
cd forks/RoseTTAFold2NA/

# create conda environment for RoseTTAFold2NA
conda deactivate
mamba env create -f RF2na-linux.yml --prefix RF2NA/
conda activate RF2NA/

# install SE(3)-Transformer locally to run RoseTTAFold2NA #
cd SE3Transformer/
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ..

# download pre-trained weights locally #
cd network
wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz
tar xvfz RF2NA_apr23.tgz
du -sh weights/ # note: this directory should contain a 1.1GB weights file
cd ..

# download sequence and structure databases #
# note: downloading these databases is unnecessary if one will only be using RF2NA's single-sequence mode #
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz

# RNA databases
mkdir -p RNA
cd RNA

# Rfam [300M]
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full_region.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
gunzip Rfam.cm.gz
cmpress Rfam.cm

# RNAcentral [12G]
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/rfam/rfam_annotations.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/id_mapping/id_mapping.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz
../input_prep/reprocess_rnac.pl id_mapping.tsv.gz rfam_annotations.tsv.gz   # ~8 minutes
gunzip -c rnacentral_species_specific_ids.fasta.gz | makeblastdb -in - -dbtype nucl  -parse_seqids -out rnacentral.fasta -title "RNACentral"

# nt [151G]
update_blastdb.pl --decompress nt
cd ../../../../  # return to the root directory of the project
```

## How to prepare input data <a name="dataset-preparation"></a>

### How to download protein input data <a name="protein-dataset-preparation"></a>

To get the protein portion of the training dataset, first download the PDB and then preprocess it using our provided scripts. The PDB can be downloaded from the RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb. Note that our scripts assume you have downloaded the PDB in the mmCIF format. Navigate down to "Download Protocols" and follow the download instructions depending on your location.

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/

```bash
00/
01/
02/
..
zz/
```

In this directory, unzip all the files:

```bash
find . -name "*.gz" -exec gzip -d {} +
```

Then run the following with \<mmcif_dir> replaced with the location
of the files downloaded from the PDB.

```bash
python src/data/components/pdb/process_pdb_mmcif_files.py --mmcif_dir <mmcif_dir> --write_dir <processed_dir> --num_processes 50 --skip_existing --verbose
```

See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `protein_metadata.csv` will be saved
that contains the pickle path of each protein example as well as additional information
about each example for faster filtering.

### How to enable PDB clustering <a name="protein-dataset-clustering"></a>

To use clustered training data, download the clusters at 30% sequence identity
via the [RCSB](https://www.rcsb.org/docs/programmatic-access/file-download-services#sequence-clusters-data).
This download link also works at the time of writing:

```
https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt
```

Place this file in `data/processed_pdb` or anywhere in your file system.
Update your config to point to the clustered data:

```yaml
data:
  cluster_path: data/processed_pdb/clusters-by-entity-30.txt
```

To use clustered data, set `sample_mode` to either `cluster_time_batch` or `cluster_length_batch`.
See next section for details.

### Setting the batching mode <a name="dataset-batching"></a>

```yaml
experiment:
  # Use one of the following:

  # Each batch contains multiple time steps of the same protein.
  sample_mode: time_batch

  # Each batch contains multiple proteins of the same length.
  sample_mode: length_batch

  # Each batch contains multiple time steps of a protein from a cluster.
  sample_mode: cluster_time_batch

  # Each batch contains multiple clusters of the same length.
  sample_mode: cluster_length_batch
```

### How to download nucleic acid input data <a name="na-dataset-downloading"></a>

```bash
# download complexes from the PDB for training
python notebooks/creating_protein_na_datasets_from_the_pdb.py
```

### How to analyze input data <a name="na-dataset-analysis"></a>

```bash
# download Pfam databases and related metadata
pfam_db_dir=~/Databases/Pfam
mkdir -p $pfam_db_dir

curl ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz | gunzip > $pfam_db_dir/Pfam-A.hmm
curl ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz | gunzip > $pfam_db_dir/Pfam-A.hmm.dat
curl ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/active_site.dat.gz | gunzip > $pfam_db_dir/active_site.dat

cd $pfam_db_dir
hmmpress Pfam-A.hmm

# execute analysis script to attach Pfam annotations to existing dataset metadata
python notebooks/analyze_dataset_characteristics.py
```

### How to process nucleic acid input data <a name="na-dataset-preparation"></a>

First, group each of the downloaded PDB files by their two middle alphanumeric characters,
with \<pdb_dir> replaced with the location of the PDB files downloaded.

```bash
cd <pdb_dir> && find . -maxdepth 1 -type f -name "*.pdb" -exec sh -c 'dir=$(basename {} | awk -F"[._]" "{print substr(\$1, 2, 2)}"); mkdir -p "$dir"; mv {} "$dir"' \;
```

Then, run the following with \<pdb_dir> replaced with the location of the PDB files downloaded.

```bash
python src/data/components/pdb/process_pdb_na_files.py --pdb_dir <pdb_dir> --write_dir <processed_dir> --num_processes 50 --skip_existing --verbose
```

See the script for more options. Each PDB will be written as a pickle file that
we read and process in the data loading pipeline. A `na_metadata.csv` will be saved
that contains the pickle path of each nucleic acid example as well as additional information
about each example for faster filtering.

**Note:** The `process_pdb_na_files.py` script must be run *after* first running the `process_pdb_mmcif_files.py` script, to ensure that the nucleic acid complexes curated by the `creating_protein_na_datasets_from_the_pdb.py` script overwrite those collected by `process_pdb_mmcif_files.py`. This results in de-duplication of approximately 2,500 PDB complexes that are assembled by both `process_pdb_mmcif_files.py` (first) and `creating_protein_na_datasets_from_the_pdb.py` (second, and preferred since nucleic acid molecules are included in this collection).

## How to combine protein and nucleic acid metadata for dataloading <a name="protein-na-dataset-combination"></a>

Lastly, when building the dataset from scratch, we need to combine
the `protein_metadata.csv` and `na_metadata.csv` files by joining the two by
their `pdb_name` columns and dropping duplicate rows contained in `protein_metadata.csv`
(i.e., preferring to keep entries in `na_metadata.csv` over those in `protein_metadata.csv`).
One can do so as follows:

```bash
python src/data/components/pdb/join_pdb_metadata.py na_metadata_csv_path=<processed_dir>/na_metadata.csv protein_metadata_csv_path=<processed_dir>/protein_metadata.csv metadata_output_csv_path=<processed_dir>/metadata.csv
```

After the script above is finished, the resulting `metadata.csv` file should also be placed
in `metadata/` as `PDB_NA_Dataset.csv` to finalize the information
required for the model's dataloading pipeline, as shown below:

```bash
cp <processed_dir>/metadata.csv metadata/PDB_NA_Dataset.csv
```

## How to train <a name="model-training"></a>

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/). For example:

```bash
python src/train.py experiment=pdb_prot_na_gen_se3
```

**Note**: You can override any parameter from the command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## How to sample <a name="model-sampling-and-eval"></a>

`sample.py` is the inference script, which can be run as follows.

```bash
python src/sample.py
```

For example, to reproduce the sampling experiments in our manuscript,
one can run the inference script as follows, for protein-only,
nucleic acid-only, and protein-nucleic acid sequence-structure diffusion,
respectively:

```bash
# protein-only sampling and evaluation
python src/sample.py inference.name='protein_only_sequence_structure_se3_discrete_diffusion_stratified_eval_${now:%Y-%m-%d}_${now:%H-%M-%S}' inference.seed=123 inference.run_statified_eval=true inference.filter_eval_split=true inference.run_self_consistency_eval=true inference.run_diversity_eval=true inference.run_novelty_eval=true inference.output_dir=\'./inference_eval_outputs/\' inference.samples.min_length=10 inference.samples.max_length=120 inference.samples.num_length_steps=10 inference.samples.min_num_chains=1 inference.samples.max_num_chains=4 paths.usalign_exec_path=$HOME/Programs/USalign/USalign paths.qtmclust_exec_path=$HOME/Programs/USalign/qTMclust data.data_cfg.filtering.mmcif_allowed_oligomer=[notype] data.data_cfg.filtering.allowed_molecule_types=[protein] ckpt_path=checkpoints/protein_na_sequence_structure_model_c4dddeef121c4ff0969f8e50ea442ab7_no_monomers_with_torsion_sup_epoch_78.ckpt
```

```bash
# nucleic acid-only sampling and evaluation
python src/sample.py inference.name='na_only_sequence_structure_se3_discrete_diffusion_stratified_eval_${now:%Y-%m-%d}_${now:%H-%M-%S}' inference.seed=456 inference.run_statified_eval=true inference.filter_eval_split=true inference.run_self_consistency_eval=true inference.run_diversity_eval=true inference.run_novelty_eval=true inference.output_dir=\'./inference_eval_outputs/\' inference.samples.min_length=10 inference.samples.max_length=120 inference.samples.num_length_steps=10 inference.samples.min_num_chains=1 inference.samples.max_num_chains=4 paths.usalign_exec_path=$HOME/Programs/USalign/USalign paths.qtmclust_exec_path=$HOME/Programs/USalign/qTMclust data.data_cfg.filtering.mmcif_allowed_oligomer=[notype] data.data_cfg.filtering.allowed_molecule_types=[na] inference.measure_auxiliary_na_metrics=true ckpt_path=checkpoints/na_only_sequence_structure_model_51d3f5f38c0c4354af562b8cfac3e486_epoch_127.ckpt
```

```bash
# protein-nucleic acid sampling and evaluation
python src/sample.py inference.name='protein_na_sequence_structure_se3_discrete_diffusion_stratified_eval_${now:%Y-%m-%d}_${now:%H-%M-%S}' inference.seed=789 inference.run_statified_eval=true inference.filter_eval_split=true inference.run_self_consistency_eval=true inference.run_diversity_eval=true inference.run_novelty_eval=true inference.output_dir=\'./inference_eval_outputs/\' inference.samples.min_length=10 inference.samples.max_length=120 inference.samples.num_length_steps=10 inference.samples.min_num_chains=1 inference.samples.max_num_chains=4 paths.usalign_exec_path=$HOME/Programs/USalign/USalign paths.qtmclust_exec_path=$HOME/Programs/USalign/qTMclust data.data_cfg.filtering.mmcif_allowed_oligomer=[notype] data.data_cfg.filtering.allowed_molecule_types=[protein,na] ckpt_path=checkpoints/protein_na_sequence_structure_model_c4dddeef121c4ff0969f8e50ea442ab7_no_monomers_with_torsion_sup_epoch_78.ckpt
```

The config for inference is in `configs/sample.yaml`.
See the config for different inference options.
By default, inference will use the model weights
`checkpoints/na_only_sequence_structure_model_51d3f5f38c0c4354af562b8cfac3e486_epoch_127.ckpt`
for nucleic acid joint generation of sequence and structure.
Simply change the `ckpt_path` to use your custom weights
(e.g., for structure-only generation) as desired.

**Note**: Evaluation with RoseTTAFold2NA for assessing
designability does not allow semicolons (`:`) in the
names of any FASTA file inputs. Keep this in mind when
choosing your value for `inference.name` or `inference.output_dir`.

```yaml
ckpt_path: <path>
```

Samples will be saved to `inference.output_dir` in the `sample.yaml`. By default it is
set to `./inference_outputs/`. Sample outputs will be saved as follows:

```bash
inference_outputs
└── 10D_07M_2023Y_20h_46m_13s                                         # date time of inference
    └── length_100                                                    # sampled length
        ├── sample_0                                                  # sample ID for length
        │   ├── self_consistency                                      # self-consistency results
        │   │   ├── rf2na                                             # RoseTTAFold2NA prediction using generated chain sequences
        │   │   │   ├── sample_1.pdb
        │   │   ├── {protein_,na_,}sample_1.pdb
        │   │   ├── sc_results.csv                                    # summary metrics CSV
        │   │   └── seqs
        │   │       ├── {P:,S:,R:,PR:,}sample_0_chain_a.fa            # generated sequence for each generated chain
        │   │       ├── {P:,S:,R:,PR:,}sample_0_chain_b.fa            # note: `P:` -> protein; `S:` -> single-stranded DNA; `R` -> RNA; and `PR:` -> protein-RNA
        │   │       ├── {P:,S:,R:,PR:,}sample_0_chain_a.fasta         # note: MMDiff sequence when `inference.generate_protein_sequences_using_pmpnn` is `true` for protein-only generation
        │   │       ├── {P:,S:,R:,PR:,}sample_0_chain_b.fasta
        │   │       └── ...
        │   ├── {protein_,na_,}bb_traj_1.pdb                          # `x_{t-1}` diffusion trajectory
        │   ├── {protein_,na_,}sample_1.pdb                           # final sample
        │   └── x0_traj_1.pdb                                         # `x_0` model prediction trajectory
        └── sample_1                                                  # next sample
```

## Acknowledgements <a name="acknowledgements"></a>

`MMDiff` builds upon the source code and data from the following projects:

- [se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [OpenComplex](https://github.com/baaihealth/OpenComplex)
- [RoseTTAFold2NA](https://github.com/uw-ipd/RoseTTAFold2NA)

We thank all their contributors and maintainers!

## Citing this work <a name="citations"></a>

If you use the code or data associated with this package or otherwise find such work useful, please cite:

```bibtex
@inproceedings{
  morehead2023towards,
  title={Towards Joint Sequence-Structure Generation of Nucleic Acid and Protein Complexes with SE(3)-Discrete Diffusion},
  author={Morehead, Alex and Bhatnagar, Aadyot and Ruffolo, Jeffrey A. and Madani, Ali},
  booktitle={NeurIPS Machine Learning in Structural Biology Workshop},
  year={2023}
}
```
