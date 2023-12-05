# %% [markdown]
# # Analyzing Protein-Nucleic Acid (NA) Data from the PDB

import matplotlib.pyplot as plt

# %%
import pandas as pd

# Load the CSV file
df = pd.read_csv("/export/home/Repositories/MMDiff/metadata/PDB_NA_Dataset.csv")

# Calculate word frequencies in the 'name' column
name_frequencies = df["name"].str.lower().str.split().explode().value_counts().head(25)

# Plot word frequencies
plt.figure(figsize=(12, 6))
name_frequencies.plot(kind="bar")
plt.title("Top 25 Word Frequencies in Names")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Process and group the 'source' column
df["source"] = df["source"].str.split(";")
source_counts = df.explode("source").groupby("source").size().nlargest(25)

# Plot source counts
plt.figure(figsize=(10, 6))
source_counts.plot(kind="bar")
plt.title("Top 25 Word Frequencies in Sources")
plt.xlabel("Source")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
import tempfile


def write_string_to_fasta(string):
    # Generate a random temporary file path
    _, file_path = tempfile.mkstemp(suffix=".fasta")

    # Write the string as a FASTA sequence
    with open(file_path, "w") as temp_file:
        temp_file.write(">sequence\n")
        temp_file.write(string + "\n")

    # Return the file path
    return file_path


# %%
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Declare constants
verbose = False
dataset_metadata_csv_filepath = (
    "/export/home/Repositories/MMDiff/metadata/PDB_NA_Dataset.csv"
)
pfam_db_dir = "/export/home/Data/Databases/Pfam"
outputs_filepath = "/tmp/pfam_output.txt"  # nosec

# Load your dataset's metadata from a file (assuming the metadata is stored in CSV format)
metadata_df = pd.read_csv(dataset_metadata_csv_filepath)

# Initialize new Pfam result_fields
pfam_columns = [
    "pfam_alignment_start",
    "pfam_alignment_end",
    "pfam_envelope_start",
    "pfam_envelope_end",
    "pfam_hmm_acc",
    "pfam_hmm_name",
    "pfam_type",
    "pfam_hmm_start",
    "pfam_hmm_end",
    "pfam_hmm_length",
    "pfam_bit_score",
    "pfam_e_value",
    "pfam_significance",
    "pfam_clan",
]
for column in pfam_columns:
    metadata_df[column] = np.nan

# Iterate over each sequence
for row_index, row in tqdm(metadata_df.iterrows()):
    # Get the sequence ID
    sequence_id = row.pdb

    # Convert the sequence to a string
    sequence_str = str(row.sequence)

    # Check if the sequence is sourced from a protein
    if row.molecule_type not in ["protein"]:
        if verbose:
            print(
                "Skipping updates for sequence",
                sequence_id,
                "as it is not derived from a protein.",
            )
        continue

    # Print the sequence ID
    if verbose:
        print("Sequence ID:", sequence_id)

    # Fetch PFAM annotations for the protein sequence using HMMER
    input_fasta_filepath = write_string_to_fasta(sequence_str)
    os.remove(outputs_filepath) if os.path.exists(outputs_filepath) else None
    cmd = f"pfam_scan.pl -fasta {input_fasta_filepath} -dir {pfam_db_dir} -outfile {outputs_filepath}"
    os.system(cmd)  # nosec

    # Read the PFAM output file
    with open(outputs_filepath) as f:
        pfam_lines = f.readlines()

    # Parse and record the PFAM annotations
    assert len(pfam_lines) > 0, f"Pfam must generate an output for sequence {sequence_id}"
    results_line = pfam_lines[-1]
    if results_line == "\n":
        if verbose:
            print(f"Warning: Pfam did not generate an output for sequence {sequence_id}")
        continue  # note: this means Pfam did not generate an output for this protein sequence
    result_fields = results_line.strip().split()
    assert (
        len(result_fields) == 15
    ), f"Exactly 15 result result_fields must be provided by `pfam_scan.pl` for sequence {sequence_id}"

    metadata_df.at[row_index, "pfam_alignment_start"] = int(result_fields[1])
    metadata_df.at[row_index, "pfam_alignment_end"] = int(result_fields[2])
    metadata_df.at[row_index, "pfam_envelope_start"] = int(result_fields[3])
    metadata_df.at[row_index, "pfam_envelope_end"] = int(result_fields[4])
    metadata_df.at[row_index, "pfam_hmm_acc"] = str(result_fields[5])
    metadata_df.at[row_index, "pfam_hmm_name"] = str(result_fields[6])
    metadata_df.at[row_index, "pfam_type"] = str(result_fields[7])
    metadata_df.at[row_index, "pfam_hmm_start"] = int(result_fields[8])
    metadata_df.at[row_index, "pfam_hmm_end"] = int(result_fields[9])
    metadata_df.at[row_index, "pfam_hmm_length"] = int(result_fields[10])
    metadata_df.at[row_index, "pfam_bit_score"] = float(result_fields[11])
    metadata_df.at[row_index, "pfam_e_value"] = float(result_fields[12])
    metadata_df.at[row_index, "pfam_significance"] = int(result_fields[13])
    metadata_df.at[row_index, "pfam_clan"] = str(result_fields[14])

    # Print the PFAM annotation details
    if verbose:
        for column in pfam_columns:
            print(f"{column}: {metadata_df.at[row_index, column]}")
        print()

    # Periodically store updated metadata on local device
    metadata_df.to_csv(dataset_metadata_csv_filepath + ".new")
