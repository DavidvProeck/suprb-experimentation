# ────────────────────────────────────────────────────────────────
# Coverage Helpers
# ────────────────────────────────────────────────────────────────
import csv
from typing import Any, Dict, List, Optional

from pathlib import Path
import numpy as np


def rule_masks(rules: List[Any], X: np.ndarray) -> np.array:
    if not rules:
        return np.zeros((0, len(X)), dtype=bool)
    masks = [np.asarray(r.match(X), dtype=bool).ravel() for r in rules]
    return np.vstack(masks)

def global_coverage(masks: np.ndarray, return_counts: bool = False):
    if masks.size == 0:
        return (0.0, np.zeros((0,), dtype=int)) if return_counts else 0.0
    covered_any = masks.any(axis=0)
    frac = float(covered_any.mean())
    if return_counts:
        counts = masks.sum(axis=0).astype(int)
        return frac, counts
    return frac


# ────────────────────────────────────────────────────────────────
# Overlap Metrics
# ────────────────────────────────────────────────────────────────
def overlap_metrics(masks: np.ndarray) -> Dict[str, float]:
    """
    Overlap metrics:
    - Jaccard Distance in [0, 1]: avg pairwise Jaccard. Check for overlap where rules are active. Ignore uncovered samples as they would dominate the metric. Based on Luciano da F Costa. “Further generalizations of the Jaccard index”. In: arXiv preprint
    arXiv:2110.09619 (2021).
    - redundancy in >= 0: avg extra hits per covered sample (0 = no overlap, 1≈each covered sample hit by 2 rules) Based on Negar Koochakzadeh, Vahid Garuslu, and Frank Maurer. “Test Redundancy Mea-
    surement Based on Coverage Information: Evaluations and Lessons Learned”. In:
    2009, pp. 220–229. doi: 10.1109/ICST.2009.8.
    - gini_coverage in [0, 1]: Inequality of coverage across samples (0=even, 1=concentrated) Based on Corrado Gini. Variabilità e mutabilità: contributo allo studio delle distribuzioni e delle
    relazioni statistiche.[Fasc. I.] Tipogr. di P. Cuppini, 1912.
    """

    if masks.size == 0:
        return {"mean_jaccard": 0.0, "redundancy": 0.0, "gini_coverage": 0.0}
    n_rules, _ = masks.shape

    j_vals = []
    for i in range(n_rules):
        Ai = masks[i]
        for j in range(i + 1, n_rules):
            Aj = masks[j]
            inter = np.logical_and(Ai, Aj).sum()
            union = np.logical_or(Ai, Aj).sum()
            jacc = (inter / union) if union > 0  else 0.0
            j_vals.append(jacc)
    
    mean_jaccard = float(np.mean(j_vals)) if j_vals else 0.0


    counts = masks.sum(axis=0).astype(float)
    covered = counts > 0
    redundancy = float((counts[covered] - 1).mean()) if covered.any() else 0.0

    if counts.sum() <= 0:
        gini_coverage = 0.0
    else:
        x = np.sort(counts)
        n = len(x)
        cumx = np.cumsum(x)
        gini_coverage = 1.0 + 1.0 / n - 2.0 * (cumx.sum() / (n * cumx[-1]))


    return {
        "mean_jaccard": mean_jaccard,
        "redundancy": redundancy,
        "gini_coverage": gini_coverage,
    }

# ────────────────────────────────────────────────────────────────
# Average Crowding Distance (uses rule.crowding_distance_)
# ────────────────────────────────────────────────────────────────

def average_crowding_distance(rules: List[Any], ignore_inf: bool = True) -> float:
    vals = []
    for r in rules:
        v = getattr(r, "crowding_distance_", None)
        if v is None:
            continue
        if ignore_inf and np.isinf(v):
            continue
        vals.append(float(v))
    
    return float(np.mean(vals)) if vals else 0.0

# ────────────────────────────────────────────────────────────────
# Average Accuracy (uses rule.error_)
# ────────────────────────────────────────────────────────────────
def average_mse(
        rules: List[Any],
        masks: Optional[np.ndarray] = None,
        aggregation: str = "coverage_weighted",
) -> Dict[str, float]:

    per_rule_mse = []
    per_rule_sizes = []

    if masks is not None:
        sizes = masks.sum(axis=1).astype(float)
    else:
        sizes = None
    
    for i, r in enumerate(rules):
        mse = float(getattr(r, "error_", np.nan))
        if not np.isfinite(mse):
            continue
        per_rule_mse.append(mse)
        per_rule_sizes.append(1.0 if sizes is None else sizes[i])

    if not per_rule_mse:
        return {
            "mean": 0.0,
            "mean_uniform": 0.0,
            "mean_coverage_weighted": 0.0,
            "n_effective_rules": 0.0
        }
    
    vals = np.asarray(per_rule_mse, dtype=float)
    sizes = np.asarray(per_rule_sizes, dtype=float)

    mean_uniform = float(vals.mean())
    if sizes.sum() > 0:
        weights = sizes / sizes.sum()
        mean_weighted = float(np.sum(vals * weights))
    else:
        mean_weighted = mean_uniform
    
    return {
        "mean": mean_weighted if aggregation == "coverage_weighted" else mean_uniform,
        "mean_uniform": mean_uniform,
        "mean_coverage_weighted": mean_weighted,
        "n_effective_rules": int(len(vals))
    }


# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
def summarize_rule_set(
        rules: List[Any],
        X: np.ndarray,
        y: np.ndarray,
        aggregation: str = "coverage_weighted",
) -> Dict[str, Any]:
    M = rule_masks(rules, X)
    cov, counts = global_coverage(M, return_counts=True)
    overlaps = overlap_metrics(M)
    crowd = average_crowding_distance(rules)
    acc_attr = average_mse(rules, masks=M, aggregation=aggregation)

    return {
        "global_coverage": cov,
        "overlap": overlaps,
        "average_crowding_distance": crowd,
        "average_error": acc_attr["mean"],
        "average_error_details": acc_attr,
        "n_rules": int(len(rules)),
        "coverage_counts_summary": {
            "mean": float(counts.mean()) if counts.size else 0.0,
            "p50": float(np.percentile(counts, 50)) if counts.size else 0.0,
            "p90": float(np.percentile(counts, 90)) if counts.size else 0.0,
            "max": int(counts.max()) if counts.size else 0,
        },
    }

# ────────────────────────────────────────────────────────────────
# CSV saving
# ────────────────────────────────────────────────────────────────
def save_metrics_to_csv(
        summary: Dict,
        filepath: str,
        extra_info: Dict,
):
    filepath = Path(filepath)
    row = {
        "global_coverage": summary["global_coverage"],
        "n_rules": summary["n_rules"],
        "average_crowding_distance": summary["average_crowding_distance"],
        "average_error": summary["average_error"],
    }

    for k, v in summary["overlap"].items():
        row[f"overlap_{k}"] = v
    for k, v in summary["coverage_counts_summary"].items():
        row[f"coverage_{k}"] = v
    if extra_info:
        row.update(extra_info)

    write_header = not filepath.exists()
    with filepath.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)