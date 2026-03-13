from matplotlib.ticker import MaxNLocator
from logging_output_scripts.utils import get_csv_df, get_normalized_df, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from utils import datasets_map

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"


def create_pareto_front_plots():
    """
    Creates scatter plots (MSE vs Complexity) for all datasets in a single row,
    colored by fold, and saves them into ONE combined PDF.
    """
    # --- Theme and Style ---
    sns.set_style("whitegrid")
    sns.set_theme(style="whitegrid",
                  font="Times New Roman",
                  font_scale=1.7,
                  rc={"pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams['figure.dpi'] = 200

    # --- Load Config ---
    config_path = 'logging_output_scripts/config.json' if os.path.exists(
        'logging_output_scripts/config.json') else 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    final_output_dir = config['output_directory']
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir, exist_ok=True)

    all_datasets_results = []

    # --- 1. Data Collection ---
    # We aggregate all data first; no plotting happens in this loop.
    for problem_key, problem_display_name in config['datasets'].items():
        res_var = pd.DataFrame()

        for heuristic_id, renamed_heuristic in config['heuristics'].items():
            if config["normalize_datasets"]:
                fold_df = get_normalized_df(heuristic_id, "../mlruns_csv/MIX")
            else:
                if config["data_directory"] == "mlruns":
                    fold_df = get_df(heuristic_id, problem_key)
                else:
                    fold_df = get_csv_df(heuristic_id, problem_key)

            if fold_df is not None and not fold_df.empty:
                fold_df['Used_Representation'] = renamed_heuristic
                res_var = pd.concat([res_var, fold_df], ignore_index=True)

        if not res_var.empty:
            # Invert MSE if necessary
            if not config["normalize_datasets"] and config["data_directory"] == "mlruns":
                res_var[mse] *= -1

            # Ensure 'fold' is treated as a string for discrete coloring
            if 'fold' in res_var.columns:
                res_var['fold'] = res_var['fold'].astype(str)

            all_datasets_results.append((problem_display_name, res_var.copy()))

    # --- 2. Generate the SINGLE Combined Row Plot ---
    num_plots = len(all_datasets_results)
    if num_plots > 0:
        # Create one figure with 'num_plots' subplots in a single row
        # squeeze=False ensures 'axes' is an array even if there is only 1 plot
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), dpi=400, squeeze=False)

        for i, (dataset_title, data) in enumerate(all_datasets_results):
            ax = axes[0, i]

            is_last = (i == num_plots - 1)

            sns.scatterplot(
                data=data,
                x=mse,
                y=complexity,
                hue='fold' if 'fold' in data.columns else None,
                style='Used_Representation',
                palette='tab10',
                s=100,
                legend=is_last,  # Legend only on the last plot
                ax=ax
            )

            if is_last:
                ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)


            ax.set_title(dataset_title, style="italic", fontsize=20)
            ax.set_xlabel("MSE", weight="bold")
            # Only show the y-axis label on the very first plot
            ax.set_ylabel("Complexity" if i == 0 else "", weight="bold")

            # Ensure complexity axis uses integer steps
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Save command: ensures only one file is created with the full row
        output_filename = os.path.join(final_output_dir, "Scatter_Row_MSE_vs_Complexity.pdf")
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Combined plot successfully saved to: {output_filename}")
    else:
        print("No data found to plot.")


if __name__ == '__main__':
    create_pareto_front_plots()