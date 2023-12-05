# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3", config_path="../../../../configs/paths", config_name="pdb_metadata.yaml"
)
def main(cfg: DictConfig):
    na_df = pd.read_csv(cfg.na_metadata_csv_path)
    protein_df = pd.read_csv(cfg.protein_metadata_csv_path)
    # impute missing columns for the nucleic acid DataFrame
    na_df["oligomeric_count"] = na_df["num_chains"].astype(str)
    na_df["oligomeric_detail"] = na_df["num_chains"].apply(
        lambda x: "heteromeric" if x > 1 else "monomeric"
    )
    # note: we can reasonably assume the following two column values due to our initial filtering for nucleic acid molecules
    na_df["resolution"] = 0.0
    na_df["structure_method"] = "x-ray diffraction"
    output_df = pd.concat([na_df, protein_df]).drop_duplicates(
        subset=["pdb_name"], keep="first", ignore_index=True
    )
    output_df.to_csv(cfg.metadata_output_csv_path, index=False)


if __name__ == "__main__":
    main()
