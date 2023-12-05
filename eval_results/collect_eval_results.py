import os
import shutil
from pathlib import Path


def main(eval_outputs_dir: str, eval_results_dir: str):
    for item in os.listdir(eval_outputs_dir):
        eval_outputs_subdir = os.path.join(eval_outputs_dir, item)
        if os.path.isdir(eval_outputs_subdir):
            all_rank_predictions_csv_path = os.path.join(
                eval_outputs_subdir, "all_rank_predictions.csv"
            )
            if os.path.exists(all_rank_predictions_csv_path):
                eval_results_run_name = Path(eval_outputs_subdir).stem
                eval_results_run_type = "_".join(
                    [
                        s.replace("naive", "rb")
                        for s in eval_results_run_name.split("_se3_discrete_diffusion_stratified")[
                            0
                        ].split("_")
                    ]
                )
                eval_results_run_category = "_".join(
                    eval_results_run_name.split("_se3_discrete_diffusion_stratified")[1]
                    .split("eval")[0]
                    .split("_")
                )
                eval_results_csv_name = os.path.join(
                    eval_results_dir,
                    f"{eval_results_run_type}{eval_results_run_category}all_rank_predictions.csv",
                )
                shutil.copyfile(all_rank_predictions_csv_path, eval_results_csv_name)


if __name__ == "__main__":
    main(eval_outputs_dir="inference_eval_outputs/", eval_results_dir="eval_results/")
