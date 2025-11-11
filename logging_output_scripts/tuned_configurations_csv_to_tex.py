import csv
import ast
import re
from collections import defaultdict

INPUT_CSV = "tuned_params_summary.csv"
OUTPUT_TEX = "tuned_params_multirow_table.tex"

CONFIG_ORDER = ["BL", "IG", "VR", "S", "V", "N-P", "N-G", "Other"]

def extract_dataset(run_name: str):
    """Extract dataset name from run_name (after 'p:')."""
    m = re.search(r"p:([a-zA-Z0-9_]+)", run_name)
    return m.group(1) if m else "unknown"

def simplify_run_name(run_name: str):
    """Simplify run name to configuration label."""
    if "InfoGain" in run_name:
        return "IG"
    if "VarianceReduction" in run_name:
        return "VR"
    if "Support" in run_name:
        return "S"
    if "Novelty-P" in run_name:
        return "N-P"
    if "Novelty-G" in run_name:
        return "N-G"
    if "Baseline" in run_name:
        return "BL"
    if "Volume" in run_name:
        return "V"
    return "Other"

def sanitize_params_string(s: str):
    """Remove Python object references like <... object at 0x...>."""
    return re.sub(r"<[^>]+object at 0x[0-9a-fA-F]+>", "'<object>'", s)

def parse_numeric_params(s: str):
    """Parse tuned_params string and keep only numeric values."""
    try:
        d = ast.literal_eval(sanitize_params_string(s))
        return {k: v for k, v in d.items() if isinstance(v, (int, float))}
    except Exception:
        return {}

def latex_escape(s: str):
    """Escape underscores for LaTeX."""
    return s.replace("_", "\\_")

def format_number(v):
    """Format: integers stay integers, floats to 4 decimals."""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return ""

def nice_param_order(keys):
    """Order columns logically."""
    preferred = [
        "rule_discovery__mu",
        "rule_discovery__lmbda",
        "rule_discovery__n_iter",
        "rule_discovery__mutation__sigma",
        "rule_discovery__min_experience",
        "rule_discovery__init__fitness__alpha",  # BL only
        "solution_composition__init__mixing__experience_calculation__upper_bound",
        "solution_composition__init__mixing__experience_weight",
        "solution_composition__init__mixing__filter_subpopulation__rule_amount",
    ]
    present = set(keys)
    ordered = [k for k in preferred if k in present]
    ordered += [k for k in sorted(keys) if k not in ordered]
    return ordered

def main():
    cfg_runs = defaultdict(list)

    # --- Read and group by configuration
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = simplify_run_name(row["run_name"])
            dataset = extract_dataset(row["run_name"])
            params = parse_numeric_params(row["tuned_params"])

            # Keep alpha only for BL
            if config != "BL":
                params.pop("rule_discovery__init__fitness__alpha", None)

            cfg_runs[config].append({"dataset": dataset, "params": params})

    # --- Collect all parameter names
    all_keys = set()
    for runs in cfg_runs.values():
        for r in runs:
            all_keys.update(r["params"].keys())
    ordered_keys = nice_param_order(all_keys)

    # --- Write one unified multirow table
    with open(OUTPUT_TEX, "w", encoding="utf-8") as tex:
        tex.write("% Requires \\usepackage{booktabs,multirow}\n\n")
        tex.write("\\begin{table*}[h!]\n")
        tex.write("\\centering\n")
        tex.write("\\caption{Tuned parameters grouped by configuration}\n")
        tex.write("\\begin{tabular}{l l " + " ".join(["c" for _ in ordered_keys]) + "}\n")
        tex.write("\\toprule\n")

        headers = ["Configuration", "Dataset"] + [latex_escape(k) for k in ordered_keys]
        tex.write(" & ".join(headers) + " \\\\\n")
        tex.write("\\midrule\n")

        for config in CONFIG_ORDER:
            runs = sorted(cfg_runs.get(config, []), key=lambda r: r["dataset"])
            if not runs:
                continue

            tex.write(f"\\multirow{{{len(runs)}}}{{*}}{{{config}}}")
            first = True
            for r in runs:
                params = r["params"]
                dataset = latex_escape(r["dataset"])
                values = [format_number(params.get(k, "")) for k in ordered_keys]
                row = " & ".join(values)
                if first:
                    tex.write(f" & {dataset} & {row} \\\\\n")
                    first = False
                else:
                    tex.write(f"& {dataset} & {row} \\\\\n")
            tex.write("\\midrule\n")

        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table*}\n")

    print(f"Created single multirow LaTeX table: {OUTPUT_TEX}")

if __name__ == "__main__":
    main()
