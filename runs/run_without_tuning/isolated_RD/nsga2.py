from datetime import timedelta
from time import time
from sklearn.linear_model import Ridge

import suprb
from suprb import rule
from suprb.optimizer.rule import origin, mutation
from suprb.optimizer.rule.nsga2 import NSGA2

from helpers import estimate_and_set_bounds, load_eggholder, init_rule_discovery_env, build_filename
from metrics import summarize_rule_set, save_metrics_to_csv


def run(metrics_config):
    t0 = time()
    random_state = 42
    X, y = load_eggholder(n_samples=250, noise=0.5, random_state=random_state)

    mu = 16
    rule_discovery = NSGA2(
        n_iter=10,
        mu=mu,
        lmbda=64,
        origin_generation=origin.SquaredError(),
        mutation=mutation.Normal(sigma=1.22,
                                 matching_type=rule.matching.OrderedBound([-1, 1])
                                 ),
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.PlaceholderFitness(),
            model=Ridge(alpha=0.01, random_state=random_state),
            matching_type=rule.matching.OrderedBound([-1, 1])),

        n_jobs=4,
        metrics=metrics_config
    )
    init_rule_discovery_env(rule_discovery)

    n_rules = mu
    estimate_and_set_bounds(rule_discovery, X)

    generated_rules = rule_discovery.optimize(X, y, n_rules=n_rules)

    runtime = timedelta(seconds=time() - t0)
    print(f"Runtime: {runtime}")

    filename = build_filename(rule_discovery)

    summary = summarize_rule_set(generated_rules, X, y)
    print("Average error (MSE):", summary["average_error"])

    save_metrics_to_csv(
        summary,
        "results/results_metrics.csv",
        extra_info={
            "graph_filename": filename,
            "metrics": ", ".join(metrics_config),
        }
    )


if __name__ == "__main__":
    metric_sets = [
        ['error', 'volume'],
        ['error', 'support'],
        ['error', 'information_gain'],
        ['error', 'variance_reduction'],
    ]

    for metrics in metric_sets:
        print(f"\n====================================================================================================")
        print(f" Starting 100 runs with metrics: {metrics}")
        print(f"====================================================================================================")

        for i in range(100):
            print(f"\n--- Run {i + 1}/100 ---")
            run(metrics)