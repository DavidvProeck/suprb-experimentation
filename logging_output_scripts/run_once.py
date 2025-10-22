import json
import os
import time
import numpy as np
import mlflow

from logging_output_scripts.violin_and_swarm_plots import create_plots
from logging_output_scripts.summary_csv import create_summary_csv
from logging_output_scripts.stat_analysis import calvo, ttest
from logging_output_scripts.utils import filter_runs

DATASETS = {
    "airfoil_self_noise": "Airfoil Self-Noise",
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "concrete_strength": "Concrete Strength",
    "energy_cool": "Energy Efficiency Cooling",
    # "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    # "parkinson_total": "Parkinson's Telemonitoring"
}

SAGA_DATASETS = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    "parkinson_total": "Parkinson's Telemonitoring",
}

MIX_DATASETS = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    "energy_cool": "Energy Efficiency Cooling",
}


SOLUTION_COMPOSITION = {
    "GeneticAlgorithm": "GA",
    "RandomSearch": "RS",
    "ArtificialBeeColonyAlgorithm": "ABC",
    "AntColonyOptimization": "ACO",
    "GreyWolfOptimizer": "GWO",
    "ParticleSwarmOptimization": "PSO",
}

SC_MIX_RD = {
    "ES Tuning": "GA",
    "RandomSearch": "RS",
    "ArtificialBeeColonyAlgorithm": "ABC",
    "AntColonyOptimization": "ACO",
    "GreyWolfOptimizer": "GWO",
    "ParticleSwarmOptimization": "PSO",
}

RULE_DISCOVERY = {
    "ES Tuning": "ES",
    "RS Tuning": "RS",
    " NS True": "NS-P",
    "MCNS True": "MCNS-P",
    "NSLC True": "NSLC-P",
    " NS False": "NS-G",
    "MCNS False": "MCNS-G",
    "NSLC False": "NSLC-G",
}

MOORD = {
    # "MOO-RD-InfoGain-tuning": "IG",
    # "MOO-RD-Volume-tuned": "Vol",
    # "MOO-RD-Support-tuned": "Sup",
    # "MOO-RD-Novelty-P-tuned": "N-P",
    # "MOO-RD-Novelty-G-tuned": "N-G",
    # "ES-Baseline-tuning": "ES",
    "NSGA2+InfoGain-tuned": "IG",
    "NSGA2+Volume-tuned": "Vol",
    "NSGA2+Support-tuned": "Sup",
    "NSGA2+Novelty-tuned": "Nov",
}

ASOC = {
    "ES Tuning": "SupRB",
    "XCSF": "XCSF",
    "Decision Tree": "DT",
    "Random Forest": "RF",
}

SAGA = {
    "s:ga": "GA",
    "s:saga1": "SAGA1",
    "s:saga2": "SAGA2",
    "s:saga3": "SAGA3",
    "s:sas": "SAGA4",
}

ADEL = {"SupRB": "SupRB", "Random Forest": "RF", "Decision Tree": "DT"}


def load_config(path="logging_output_scripts/config.json"):
    with open(path, "r") as f:
        return json.load(f)


def save_config(config, path="logging_output_scripts/config.json"):
    with open(path, "w") as f:
        json.dump(config, f)


def mlruns_to_csv(datasets, subdir, normalize):
    all_runs_df = mlflow.search_runs(search_all_experiments=True)
    os.makedirs(f"mlruns_csv/{subdir}", exist_ok=True)

    mse_col = "metrics.test_neg_mean_squared_error"
    complexity_col = "metrics.elitist_complexity"

    for dataset in datasets:
        df = all_runs_df[
            all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False)
            & (all_runs_df["tags.fold"] == "True")
        ][["tags.mlflow.runName", mse_col, complexity_col]]

        print(dataset, np.min(df[mse_col]), np.max(df[mse_col]),
              np.min(df[complexity_col]), np.max(df[complexity_col]))

        df[mse_col] *= -1

        if normalize:
            df[mse_col] = (df[mse_col] - df[mse_col].min()) / (df[mse_col].max() - df[mse_col].min())
            df[complexity_col] = (df[complexity_col] - df[complexity_col].min()) / (
                df[complexity_col].max() - df[complexity_col].min()
            )

        df.to_csv(f"mlruns_csv/{subdir}/{dataset}_all.csv", index=False)


def run_ttests(setting):
    """Run all relevant t-tests based on the current setting."""
    name = setting[0]

    if name.endswith("/RBML"):
        # Comparison between rule-based and ensemble learners
        comparisons = [
            ("XCSF", "ES Tuning", "XCSF", "SupRB"),
            ("Decision Tree", "ES Tuning", "Decision Tree", "SupRB"),
            ("Random Forest", "ES Tuning", "Random Forest", "SupRB"),
        ]

    elif name.endswith("/RD"):
        # Extensive rule discovery comparisons
        comparisons = [
            # vs ES baseline
            ("NSLC True", "ES Tuning", "NSLC-P", "ES"),
            ("NSLC False", "ES Tuning", "NSLC-G", "ES"),
            ("MCNS True", "ES Tuning", "MCNS-P", "ES"),
            ("MCNS False", "ES Tuning", "MCNS-G", "ES"),
            (" NS True", "ES Tuning", "NS-P", "ES"),
            (" NS False", "ES Tuning", "NS-G", "ES"),

            # Within methods
            ("NSLC False", "NSLC True", "NSLC-G", "NSLC-P"),
            ("MCNS True", "NSLC True", "MCNS-P", "NSLC-P"),
            ("MCNS False", "NSLC True", "MCNS-G", "NSLC-P"),
            (" NS True", "NSLC True", "NS-P", "NSLC-P"),
            (" NS False", "NSLC True", "NS-G", "NSLC-P"),

            ("MCNS True", "NSLC False", "MCNS-P", "NSLC-G"),
            ("MCNS False", "NSLC False", "MCNS-G", "NSLC-G"),
            (" NS True", "NSLC False", "NS-P", "NSLC-G"),
            (" NS False", "NSLC False", "NS-G", "NSLC-G"),

            ("MCNS False", "MCNS True", "MCNS-G", "MCNS-P"),
            (" NS True", "MCNS True", "NS-P", "MCNS-P"),
            (" NS False", "MCNS True", "NS-G", "MCNS-P"),

            (" NS True", "MCNS False", "NS-P", "MCNS-G"),
            (" NS False", "MCNS False", "NS-G", "MCNS-G"),

            (" NS False", " NS True", "NS-G", "NS-P"),
        ]

    elif name.endswith("/SC"):
        # Solution Composition (GA vs others)
        ga = "ES Tuning"
        comparisons = [
            ("RandomSearch", ga, "RS", "GA"),
            ("ArtificialBeeColonyAlgorithm", ga, "ABC", "GA"),
            ("AntColonyOptimization", ga, "ACO", "GA"),
            ("GreyWolfOptimizer", ga, "GWO", "GA"),
            ("ParticleSwarmOptimization", ga, "PSO", "GA"),
        ]

    elif name.endswith("/MIX"):
        # Mixing strategy comparisons
        comparisons = []
        if setting[4] != "mlruns_csv/MIX/subset_":
            comparisons.append((
                "r:3; f:NBestFitness; -e:ExperienceCalculation",
                "r:3; f:FilterSubpopulation; -e:ExperienceCalculation",
                r"$l$ Best", "Base"
            ))
        comparisons += [
            ("r:3; f:FilterSubpopulation; -e:CapExperience/",
             "r:3; f:FilterSubpopulation; -e:ExperienceCalculation", "Experience Cap", "Base"),
            ("r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality",
             "r:3; f:FilterSubpopulation; -e:ExperienceCalculation", "Experience Cap (dim)", "Base"),
            ("r:3; f:FilterSubpopulation; -e:CapExperience/",
             "r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality",
             "Experience Cap", "Experience Cap (dim)"),
        ]

    elif name.endswith("/SAGA"):
        # SAGA algorithm comparisons
        comparisons = [
            ("s:saga1", "s:ga", "SAGA1", "GA"),
            ("s:saga2", "s:ga", "SAGA2", "GA"),
            ("s:saga3", "s:ga", "SAGA3", "GA"),
            ("s:sas", "s:ga", "SAGA4", "GA"),

            ("s:saga1", "s:saga2", "SAGA1", "SAGA2"),
            ("s:saga1", "s:saga3", "SAGA1", "SAGA3"),
            ("s:saga1", "s:sas", "SAGA1", "SAGA4"),
            ("s:saga2", "s:saga3", "SAGA2", "SAGA3"),
            ("s:saga2", "s:sas", "SAGA2", "SAGA4"),
            ("s:saga3", "s:sas", "SAGA3", "SAGA4"),
        ]


    elif name.endswith("/NSGA2"):
        comparisons = [
            # --- Baseline comparisons ---
            # ("NSGA2+InfoGain-tuned", "NSGA2+Baseline-tuned", "IG", "ES"),
            # ("NSGA2+Volume-tuned", "NSGA2+Baseline-tuned", "Vol", "ES"),
            # ("NSGA2+Support-tuned", "NSGA2+Baseline-tuned", "Sup", "ES"),
            # ("NSGA2+Novelty-tuned", "NSGA2+Baseline-tuned", "Nov", "ES"),
            # --- Cross-objective comparisons ---
            # ("NSGA2+InfoGain-tuned", "NSGA2+Volume-tuned", "IG", "Vol"),
            # ("NSGA2+InfoGain-tuned", "NSGA2+Support-tuned", "IG", "Sup"),
            # ("NSGA2+InfoGain-tuned", "NSGA2+Novelty-tuned", "IG", "Nov"),
            ("NSGA2+Volume-tuned", "NSGA2+Support-tuned", "Vol", "Sup"),
            ("NSGA2+Volume-tuned", "NSGA2+Novelty-tuned", "Vol", "Nov"),
            ("NSGA2+Support-tuned", "NSGA2+Novelty-tuned", "Sup", "Nov"),

        ]

    else:
        comparisons = []

    for c1, c2, n1, n2 in comparisons:
        ttest(latex=False, cand1=c1, cand2=c2, cand1_name=n1, cand2_name=n2)


def run_main(setting):
    """Orchestrates a full pipeline for one experimental setting."""
    config = load_config()

    # Select datasets depending on experiment type
    if setting[0].endswith("/SAGA"):
        config["datasets"] = SAGA_DATASETS
    elif setting[0].endswith("/MIX") or setting[0].endswith("/RBML"):
        config["datasets"] = MIX_DATASETS
    else:
        config["datasets"] = DATASETS

    config.update({
        "output_directory": setting[0],
        "normalize_datasets": setting[3],
        "heuristics": setting[1],
        "data_directory": setting[4],
    })

    os.makedirs("diss-graphs/graphs", exist_ok=True)
    os.makedirs(config["output_directory"], exist_ok=True)

    save_config(config)
    time.sleep(10)

    # Filter mlflow runs if using live mlruns data
    if config["data_directory"] == "mlruns":
        all_runs_df = mlflow.search_runs(search_all_experiments=True)
        filter_runs(all_runs_df)

    # Generate graphs and analyses
    create_plots()
    calvo(ylabel=setting[2])
    run_ttests(setting)


if __name__ == "__main__":
    test = ["diss-graphs/graphs/NSGA2", MOORD, "Rule Discovery", False, "mlruns_csv/MIX"]
    rd = ["diss-graphs/graphs/RD", RULE_DISCOVERY, "Rule Discovery", False, "mlruns_csv/RD"]
    sc = ["diss-graphs/graphs/SC_only_GA", SOLUTION_COMPOSITION, "Solution Composition", False, "mlruns_csv/SC_only_GA"]
    xcsf = ["diss-graphs/graphs/RBML", ASOC, "Estimator", False, "mlruns_csv/RBML"]
    adeles = ["diss-graphs/graphs/ADELES", ADEL, "Rule Discovery", False, "mlruns"]
    mix_calvo = ["diss-graphs/graphs/MIX", {}, "Mixing Variant", True, "mlruns_csv/MIX"]
    mix_calvo_sub = ["diss-graphs/graphs/MIX/subset", {}, "Mixing Variant", True, "mlruns_csv/MIX"]
    sagas = ["diss-graphs/graphs/SAGA", SAGA, "Solution Composition", False, "mlruns_csv/SAGA"]
    sc_rd = ["diss-graphs/graphs/SC", SC_MIX_RD, "Solution Composition", False, "mlruns_csv/SC"]

    # Example execution
    mlruns_to_csv(DATASETS, "MIX", True)

    setting = test  # change to desired experiment
    run_main(setting)