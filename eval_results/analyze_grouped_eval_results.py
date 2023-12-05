import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

PLOT_ROW_GROUPS = {
    "eval_results/protein_only_sequence_structure_rb_all_rank_predictions.csv": 0,
    "eval_results/protein_only_sequence_structure_all_rank_predictions.csv": 0,
    "eval_results/protein_only_sequence_structure_specialist_all_rank_predictions.csv": 0,
    "eval_results/na_only_sequence_structure_rb_all_rank_predictions.csv": 1,
    "eval_results/na_only_sequence_structure_all_rank_predictions.csv": 1,
    "eval_results/na_only_sequence_structure_specialist_all_rank_predictions.csv": 1,
    "eval_results/protein_na_sequence_structure_rb_all_rank_predictions.csv": 2,
    "eval_results/protein_na_sequence_structure_all_rank_predictions.csv": 2,
    "eval_results/protein_na_sequence_structure_specialist_all_rank_predictions.csv": 2,
}
PLOT_ROW_GROUPS_LIST = list(PLOT_ROW_GROUPS.keys())
NUM_PLOT_ROW_GROUPS = len(set(PLOT_ROW_GROUPS.values()))


def main(eval_results_csv_dir: str):
    fig_rmsd, axes_rmsd = plt.subplots(
        nrows=NUM_PLOT_ROW_GROUPS,
        ncols=NUM_PLOT_ROW_GROUPS,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )
    fig_tm_score, axes_tm_score = plt.subplots(
        nrows=NUM_PLOT_ROW_GROUPS,
        ncols=NUM_PLOT_ROW_GROUPS,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )
    rmsd_legend_plotted = False
    tm_score_legend_plotted = False
    colorbar_groups_plotted = set()
    axes_to_label = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]

    # Increase font size for the entire figure
    plt.rcParams.update({"font.weight": "bold"})  # Note: Adjust the font size as needed

    for csv_filepath, group in PLOT_ROW_GROUPS.items():
        group_axes_rmsd = axes_rmsd[group]
        group_axes_tm_score = axes_tm_score[group]

        csv_filepaths = sorted(glob.glob(os.path.join(eval_results_csv_dir, "*.csv")))
        for csv_filepath in csv_filepaths:
            csv_plot_row_group = PLOT_ROW_GROUPS.get(csv_filepath, None)
            if csv_plot_row_group is not None and csv_plot_row_group == group:
                results_df = pd.read_csv(csv_filepath)
                results_df["sequence_length"] = results_df["sequence"].apply(lambda x: len(x))
                results_df["num_chains"] = results_df["sequence"].apply(
                    lambda x: len(x.split(",")) + 1
                )

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
                print()

                csv_type_header = "Protein-Nucleic Acid"
                csv_filepath_stem = Path(csv_filepath).stem
                if "protein_only" in csv_filepath_stem:
                    csv_type_header = "Protein"
                elif "na_only" in csv_filepath_stem:
                    csv_type_header = "Nucleic Acid"

                # Normalize the `num_chains` values to range [0, 1]
                norm = Normalize(
                    vmin=results_df["num_chains"].min(), vmax=results_df["num_chains"].max()
                )

                # Use the `ScalarMappable` class to map the `num_chains` values to colors
                smap = ScalarMappable(norm=norm, cmap="rainbow")
                colors = smap.to_rgba(results_df["num_chains"])

                # Select separate axes for scRMSD and scTM
                csv_filepath_group_axes_index = PLOT_ROW_GROUPS_LIST.index(csv_filepath) % 3
                ax_rmsd, ax_tm_score = (
                    group_axes_rmsd[csv_filepath_group_axes_index],
                    group_axes_tm_score[csv_filepath_group_axes_index],
                )
                axis = [group, csv_filepath_group_axes_index]

                if csv_filepath_group_axes_index == 2 and group not in colorbar_groups_plotted:
                    # Only plot legend for first plot of each row group
                    rmsd_colorbar = plt.colorbar(smap, ax=ax_rmsd)
                    rmsd_colorbar.set_label("Number of Chains", fontweight="bold")

                # Note: Assuming `novelty` is the column containing the novelty values
                if "novelty" in results_df.columns:
                    novelty_values = results_df["novelty"]
                    for i, (x, y, c, novelty) in enumerate(
                        zip(
                            results_df["sequence_length"],
                            results_df["rmsd"],
                            colors,
                            novelty_values,
                        )
                    ):
                        if (
                            not rmsd_legend_plotted
                            and group == 0
                            and csv_filepath_group_axes_index == 0
                            and i == 0
                        ):
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
                            ax_rmsd.legend(
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
                            ax_rmsd.scatter(
                                x, y, c=[c], label="Novelty > 0.7", marker="*", s=30, alpha=0.7
                            )
                        else:
                            ax_rmsd.scatter(
                                x, y, c=[c], label="Novelty ≤ 0.7", marker="o", s=5, alpha=0.7
                            )  # Show label only for the first point and adjust transparency
                else:
                    for i, (x, y, c) in enumerate(
                        zip(results_df["sequence_length"], results_df["rmsd"], colors)
                    ):
                        ax_rmsd.scatter(
                            x,
                            y,
                            c=[c],
                            label=None,
                            marker="o",
                            s=5,
                            alpha=0.7,
                        )  # Show label only for the first point and adjust transparency

                if axis in axes_to_label:
                    ax_rmsd.set_xlabel("Sequence Length", fontdict={"weight": "bold"})
                    if axis != [2, 2]:
                        ax_rmsd.set_ylabel("scRMSD", fontdict={"weight": "bold"})
                ax_rmsd.axhline(
                    y=2.0, color="red", linestyle="solid", label="Designability Threshold of 2 Å"
                )
                ax_rmsd.axhline(
                    y=5.0, color="red", linestyle="dashed", label="Designability Threshold of 5 Å"
                )
                ax_rmsd.grid(True)

                if csv_filepath_group_axes_index == 2 and group not in colorbar_groups_plotted:
                    # Only plot legend for first plot of each row group
                    tm_score_colorbar = plt.colorbar(smap, ax=ax_tm_score)
                    tm_score_colorbar.set_label("Number of Chains", fontweight="bold")
                    colorbar_groups_plotted.add(group)

                # Note: Assuming `novelty` is the column containing the novelty values
                if "novelty" in results_df.columns:
                    novelty_values = results_df["novelty"]
                    for i, (x, y, c, novelty) in enumerate(
                        zip(
                            results_df["sequence_length"],
                            results_df["tm_score"],
                            colors,
                            novelty_values,
                        )
                    ):
                        if (
                            not tm_score_legend_plotted
                            and group == 0
                            and csv_filepath_group_axes_index == 0
                            and i == 0
                        ):
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
                            ax_tm_score.legend(
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
                            ax_tm_score.scatter(
                                x, y, c=[c], label="Novelty > 0.7", marker="*", s=30, alpha=0.7
                            )
                        else:
                            ax_tm_score.scatter(
                                x, y, c=[c], label="Novelty ≤ 0.7", marker="o", s=5, alpha=0.7
                            )  # Show label only for the first point and adjust transparency
                else:
                    for i, (x, y, c) in enumerate(
                        zip(results_df["sequence_length"], results_df["tm_score"], colors)
                    ):
                        ax_tm_score.scatter(
                            x,
                            y,
                            c=[c],
                            label=None,
                            marker="o",
                            s=5,
                            alpha=0.7,
                        )  # Show label only for the first point and adjust transparency

                if axis in axes_to_label:
                    ax_tm_score.set_xlabel("Sequence Length", fontdict={"weight": "bold"})
                    if axis != [2, 2]:
                        ax_tm_score.set_ylabel("scTM", fontdict={"weight": "bold"})
                ax_tm_score.axhline(
                    y=0.5,
                    color="red",
                    linestyle="solid",
                    label="Designability Threshold of 0.5 TM",
                )
                ax_tm_score.axhline(
                    y=0.2,
                    color="red",
                    linestyle="dashed",
                    label="Designability Threshold of 0.2 TM",
                )
                ax_tm_score.grid(True)

    for ax in fig_rmsd.get_axes():
        ax.label_outer()
    for ax in fig_tm_score.get_axes():
        ax.label_outer()

    fig_rmsd.tight_layout()
    fig_tm_score.tight_layout()

    fig_rmsd.savefig(
        os.path.join(Path(csv_filepath).parent, "all_rank_predictions_rmsd_vs_seq_len.png")
    )
    fig_tm_score.savefig(
        os.path.join(Path(csv_filepath).parent, "all_rank_predictions_tm_score_vs_seq_len.png")
    )

    plt.close(fig_rmsd)
    plt.close(fig_tm_score)


if __name__ == "__main__":
    main(eval_results_csv_dir="eval_results/")
