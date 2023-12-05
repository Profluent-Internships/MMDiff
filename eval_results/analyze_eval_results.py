import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def main(eval_results_csv_dir: str):
    csv_filepaths = sorted(glob.glob(os.path.join(eval_results_csv_dir, "*.csv")))
    for csv_filepath in csv_filepaths:
        results_df = pd.read_csv(csv_filepath)
        results_df["sequence_length"] = results_df["sequence"].apply(lambda x: len(x))
        results_df["num_chains"] = results_df["sequence"].apply(lambda x: len(x.split(",")) + 1)

        rmsd_legend_plotted = False
        tm_score_legend_plotted = False

        print(f"For the evaluation results within `{csv_filepath}`:")
        if "rmsd" in results_df.columns:
            print(
                f"Percent where scRMSD < 2.0 Angstrom is {(results_df['rmsd'] < 2.0).mean() * 100}%"
            )
            print(
                f"Percent where scRMSD < 5.0 Angstrom is {(results_df['rmsd'] < 5.0).mean() * 100}%"
            )
            print(
                f"scRMSD of mean {results_df['rmsd'].mean()} and stdev {results_df['rmsd'].std()}"
            )
        if "tm_score" in results_df.columns:
            print(
                f"Percent where scTM > 0.2 Angstrom is {(results_df['tm_score'] > 0.2).mean() * 100}%"
            )
            print(
                f"Percent where scTM > 0.5 Angstrom is {(results_df['tm_score'] > 0.5).mean() * 100}%"
            )
            print(
                f"scTM of mean {results_df['tm_score'].mean()} and stdev {results_df['tm_score'].std()}"
            )
        if "chain_diversity" in results_df.columns:
            print(
                f"Chain diversity of mean {results_df['chain_diversity'].mean()} and stdev {results_df['chain_diversity'].std()}"
            )
        if "complex_diversity" in results_df.columns:
            print(
                f"Complex diversity of mean {results_df['complex_diversity'].mean()} and stdev {results_df['complex_diversity'].std()}"
            )
        if "single_chain_complex_diversity" in results_df.columns:
            print(
                f"Single-chain diversity of mean {results_df['single_chain_complex_diversity'].mean()} and stdev {results_df['single_chain_complex_diversity'].std()}"
            )
        if "all_chain_complex_diversity" in results_df.columns:
            print(
                f"All-chain diversity of mean {results_df['all_chain_complex_diversity'].mean()} and stdev {results_df['all_chain_complex_diversity'].std()}"
            )
        if "trainTM" in results_df.columns:
            results_df["novelty"] = 1.0 - results_df["trainTM"]
            print(
                f"Novelty of mean {results_df['novelty'].mean()} and stdev {results_df['novelty'].std()}"
            )
        # Log nucleic acid-only statistics
        if (
            "num_rf2na_base_pairs" in results_df.columns
            and not results_df["num_rf2na_base_pairs"].isnull().all()
        ):
            print(
                f"Generated samples number of RF2NA-generated base pairs of mean {results_df['num_rf2na_base_pairs'].mean()} and stddev {results_df['num_rf2na_base_pairs'].std()}"
            )
        if (
            "radius_gyration" in results_df.columns
            and not results_df["radius_gyration"].isnull().all()
        ):
            print(
                f"Generated samples radius of gyration of mean {results_df['radius_gyration'].mean()} and stddev {results_df['radius_gyration'].std()}"
            )
            # Plot radius of gyration analysis figure
            # results_df["num_chains"] = results_df["sequence"].apply(lambda x: len(x.split(",")) + 1)
            # cmap = sns.color_palette("viridis", as_cmap=True)
            # plot = sns.histplot(
            #     data=results_df, x="radius_gyration", kde=True, hue="num_chains", palette=cmap
            # )
            # plot.set_title(f"Nucleic Acid Radius of Gyration Generated Distribution")
            # plt.savefig(
            #     f"{os.path.join(Path(csv_filepath).parent, f'{Path(csv_filepath).stem}_radius_gyration')}.png"
            # )
            # plot.clear()
        print()

        csv_type_header = "Protein-Nucleic Acid"
        csv_filepath_stem = Path(csv_filepath).stem
        if "protein_only" in csv_filepath_stem:
            csv_type_header = "Protein"
        elif "na_only" in csv_filepath_stem:
            csv_type_header = "Nucleic Acid"

        # Normalize the `num_chains` values to range [0, 1]
        norm = Normalize(vmin=results_df["num_chains"].min(), vmax=results_df["num_chains"].max())

        # Use the `ScalarMappable` class to map the `num_chains` values to colors
        smap = ScalarMappable(norm=norm, cmap="rainbow")
        colors = smap.to_rgba(results_df["num_chains"])

        # Increase font size for the entire figure
        plt.rcParams.update(
            {"font.size": 24, "font.weight": "bold"}
        )  # Note: Adjust the font size as needed

        plt.figure(figsize=(12, 8))  # Increase figure size for readability

        # Note: Assuming `novelty` is the column containing the novelty values
        if "novelty" in results_df.columns:
            novelty_values = results_df["novelty"]
            for i, (x, y, c, novelty) in enumerate(
                zip(results_df["sequence_length"], results_df["rmsd"], colors, novelty_values)
            ):
                if not rmsd_legend_plotted and i == 0:
                    # Add custom legend entry for star marker
                    custom_legend_entry_rmsd_1 = Line2D(
                        [], [], marker="*", color="k", markersize=8, label="Novelty > 0.7"
                    )
                    custom_legend_entry_rmsd_2 = Line2D(
                        [], [], marker="o", color="k", markersize=5, label="Novelty ≤ 0.7"
                    )
                    threshold_legend_entry_rmsd_1 = Line2D(
                        [],
                        [],
                        color="red",
                        linestyle="solid",
                        label="Designability Threshold of 2 Å",
                    )
                    threshold_legend_entry_rmsd_2 = Line2D(
                        [],
                        [],
                        color="salmon",
                        linestyle="dashed",
                        label="Designability Threshold of 5 Å",
                    )
                    plt.legend(
                        handles=[
                            custom_legend_entry_rmsd_1,
                            custom_legend_entry_rmsd_2,
                            threshold_legend_entry_rmsd_1,
                            threshold_legend_entry_rmsd_2,
                        ]
                    )
                    rmsd_legend_plotted = True
                if novelty > 0.7:
                    # Plot a star marker for high novelty values
                    plt.scatter(x, y, c=[c], label="Novelty > 0.7", marker="*", s=30, alpha=0.7)
                else:
                    plt.scatter(
                        x, y, c=[c], label="Novelty ≤ 0.7", marker="o", s=5, alpha=0.7
                    )  # Show label only for the first point and adjust transparency
        else:
            for i, (x, y, c) in enumerate(
                zip(results_df["sequence_length"], results_df["rmsd"], colors)
            ):
                plt.scatter(
                    x,
                    y,
                    c=[c],
                    label=None,
                    marker="o",
                    s=5,
                    alpha=0.7,
                )  # Show label only for the first point and adjust transparency

        plt.xlabel("Sequence Length", fontdict={"weight": "bold"})
        plt.ylabel("scRMSD", fontdict={"weight": "bold"})
        plt.axhline(
            y=2.0, color="red", linestyle="solid", label="Designability Threshold of 2 Å"
        )  # Add red horizontal line
        plt.axhline(
            y=5.0, color="red", linestyle="dashed", label="Designability Threshold of 5 Å"
        )  # Add red horizontal line
        # plt.legend()
        plt.grid(True)
        color_bar = plt.colorbar(smap, label="Number of Chains")
        color_bar.set_label(
            label="Number of Chains", fontweight="bold"
        )  # Set font weight to bold for color bar label
        # plt.title(f"{csv_type_header} scRMSD vs Sequence Length")
        plt.show()
        plt.savefig(
            f"{os.path.join(Path(csv_filepath).parent, f'{Path(csv_filepath).stem}_rmsd_vs_seq_len')}.png"
        )

        plt.figure(figsize=(12, 8))  # Increase figure size for readability

        # Note: Assuming `novelty` is the column containing the novelty values
        if "novelty" in results_df.columns:
            novelty_values = results_df["novelty"]
            for i, (x, y, c, novelty) in enumerate(
                zip(results_df["sequence_length"], results_df["tm_score"], colors, novelty_values)
            ):
                if not tm_score_legend_plotted and i == 0:
                    # Add custom legend entry for star marker
                    custom_legend_entry_tm_score_1 = Line2D(
                        [], [], marker="*", color="k", markersize=8, label="Novelty > 0.7"
                    )
                    custom_legend_entry_tm_score_2 = Line2D(
                        [], [], marker="o", color="k", markersize=5, label="Novelty ≤ 0.7"
                    )
                    threshold_legend_entry_tm_score_1 = Line2D(
                        [],
                        [],
                        color="red",
                        linestyle="solid",
                        label="Designability Threshold of 0.5 TM",
                    )
                    threshold_legend_entry_tm_score_2 = Line2D(
                        [],
                        [],
                        color="salmon",
                        linestyle="dashed",
                        label="Designability Threshold of 0.2 TM",
                    )
                    plt.legend(
                        handles=[
                            custom_legend_entry_tm_score_1,
                            custom_legend_entry_tm_score_2,
                            threshold_legend_entry_tm_score_1,
                            threshold_legend_entry_tm_score_2,
                        ]
                    )
                    tm_score_legend_plotted = True
                if novelty > 0.7:
                    # Plot a star marker for high novelty values
                    plt.scatter(x, y, c=[c], label="Novelty > 0.7", marker="*", s=30, alpha=0.7)
                else:
                    plt.scatter(x, y, c=[c], label="Novelty ≤ 0.7", marker="o", s=5, alpha=0.7)
        else:
            for i, (x, y, c) in enumerate(
                zip(results_df["sequence_length"], results_df["tm_score"], colors)
            ):
                plt.scatter(
                    x,
                    y,
                    c=[c],
                    label=None,
                    marker="o",
                    s=5,
                    alpha=0.7,
                )

        plt.xlabel("Sequence Length", fontdict={"weight": "bold"})
        plt.ylabel("scTM", fontdict={"weight": "bold"})
        plt.axhline(
            y=0.5, color="red", linestyle="solid", label="Designability Threshold of 0.5 TM"
        )  # Add red horizontal line
        plt.axhline(
            y=0.2, color="red", linestyle="dashed", label="Designability Threshold of 0.2 TM"
        )  # Add red horizontal line
        # plt.legend()
        plt.grid(True)
        color_bar = plt.colorbar(smap, label="Number of Chains")
        color_bar.set_label(
            label="Number of Chains", fontweight="bold"
        )  # Set font weight to bold for color bar label
        # plt.title(f"{csv_type_header} scTM vs Sequence Length")
        plt.show()
        plt.savefig(
            f"{os.path.join(Path(csv_filepath).parent, f'{Path(csv_filepath).stem}_tm_score_vs_seq_len')}.png"
        )


if __name__ == "__main__":
    main(eval_results_csv_dir="eval_results/")
