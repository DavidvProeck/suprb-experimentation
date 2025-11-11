import os
import yaml
import csv
from pathlib import Path

# Path to your mlruns directory
MLRUNS_DIR = Path("/home/vonproda/Desktop/BA/suprb-experimentation/mlruns")
OUTPUT_CSV = "tuned_params_summary.csv"

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

def read_file_content(file_path: Path):
    """Read the content of a file as a string."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "N/A"
    except Exception as e:
        print(f"[WARNING] Could not read file at {file_path}: {e}")
        return "N/A"

def main():
    data_rows = []

    for root, dirs, files in os.walk(MLRUNS_DIR):
        root_path = Path(root)

        # Look specifically for tuned_params inside params folders
        if "tuned_params" in files and root_path.name == "params":
            tuned_params_path = root_path / "tuned_params"
            tuning_n_calls_path = root_path / "tuning_n_calls"

            # meta.yaml is one directory above the params folder
            run_dir = root_path.parent
            meta_path = run_dir / "meta.yaml"

            # The experiment folder is one level above the run directory
            experiment_id = run_dir.parent.name

            run_id, run_name = extract_meta_info(meta_path)
            tuned_params_content = read_file_content(tuned_params_path)
            tuning_n_calls_content = read_file_content(tuning_n_calls_path)

            data_rows.append({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_name": run_name,
                "tuning_n_calls": tuning_n_calls_content,
                "tuned_params": tuned_params_content
            })

    # Write results to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["experiment_id", "run_id", "run_name", "tuning_n_calls", "tuned_params"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

    print(f"CSV created successfully: {OUTPUT_CSV}")
    print(f"Total tuned_params files found: {len(data_rows)}")

if __name__ == "__main__":
    main()
