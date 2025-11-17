import os
import re
import yaml
import csv
from pathlib import Path
import statistics
from collections import defaultdict

# Path to your mlruns directory
MLRUNS_DIR = Path("/mlruns")
OUTPUT_CSV = "elitist_metrics_summary_by_algorithm_dataset.csv"

def extract_meta_info(meta_path: Path):
    """Extract run_id and run_name from a meta.yaml file."""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        run_id = meta.get("run_id", "N/A")
        run_name = meta.get("run_name", "N/A")
        return run_id, run_name
    except Exception as e:
        print(f"[WARNING] Could not read meta.yaml at {meta_path}: {e}")
        return "N/A", "N/A"

def read_metric_file(metric_path: Path):
    """Read MLflow metric file and return the average of its values."""
    try:
        with metric_path.open("r", encoding="utf-8") as f:
            values = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1]))
                    except ValueError:
                        pass
            if values:
                return statistics.mean(values)
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARNING] Could not read metric file at {metric_path}: {e}")
        return None

def parse_run_name(run_name: str):
    """
    Extract algorithm and dataset names from a run_name string.

    Examples:
        "/MOO-RD-Volume-tuning j:739648 p:airfoil_self_noise; ..."
            -> ("Volume", "airfoil_self_noise")
        "/MOO-RD-Novelty-G-tuning j:739721 p:combined_cycle_power_plant; ..."
            -> ("Novelty-G", "combined_cycle_power_plant")
        "/ES-Baseline-tuning j:739642 p:concrete_strength; ..."
            -> ("Baseline", "concrete_strength")
    """
    if not run_name or run_name == "N/A":
        return ("Unknown", "Unknown")

    # Algorithm extraction:
    # Matches either "MOO-RD-" or "ES-" followed by the algorithm name (letters, digits, _, or -)
    algo_match = re.search(r'(?:MOO-RD-|ES-)([\w-]+)-tuning', run_name)
    algorithm = algo_match.group(1) if algo_match else "Unknown"

    # Dataset extraction: after 'p:' and before ';'
    dataset_match = re.search(r'p:([^;]+);', run_name)
    dataset = dataset_match.group(1) if dataset_match else "Unknown"

    return (algorithm, dataset)


def main():
    # Structure: {(algorithm, dataset): {"complexity": [...], "error": [...]}}
    grouped_metrics = defaultdict(lambda: {"complexity": [], "error": []})

    for root, dirs, files in os.walk(MLRUNS_DIR):
        root_path = Path(root)

        if root_path.name == "metrics":
            complexity_path = root_path / "elitist_complexity"
            error_path = root_path / "elitist_error"

            if complexity_path.exists() or error_path.exists():
                run_dir = root_path.parent
                meta_path = run_dir / "meta.yaml"
                _, run_name = extract_meta_info(meta_path)

                algorithm, dataset = parse_run_name(run_name)

                avg_complexity = read_metric_file(complexity_path) if complexity_path.exists() else None
                avg_error = read_metric_file(error_path) if error_path.exists() else None

                key = (algorithm, dataset)

                if avg_complexity is not None:
                    grouped_metrics[key]["complexity"].append(avg_complexity)
                if avg_error is not None:
                    grouped_metrics[key]["error"].append(avg_error)

    data_rows = []
    for (algorithm, dataset), metrics in grouped_metrics.items():
        avg_complexity = statistics.mean(metrics["complexity"]) if metrics["complexity"] else "N/A"
        avg_error = statistics.mean(metrics["error"]) if metrics["error"] else "N/A"

        data_rows.append({
            "algorithm": algorithm,
            "dataset": dataset,
            "avg_elitist_complexity": avg_complexity,
            "avg_elitist_error": avg_error,
            "num_runs": len(metrics["complexity"])
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["algorithm", "dataset", "avg_elitist_complexity", "avg_elitist_error", "num_runs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

    print(f"CSV created successfully: {OUTPUT_CSV}")
    print(f"Total (algorithm, dataset) pairs summarized: {len(data_rows)}")

if __name__ == "__main__":
    main()
