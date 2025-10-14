from datetime import timedelta
from time import time
from sklearn.linear_model import Ridge

import suprb
from suprb import rule
from suprb.optimizer.rule import origin, mutation
from suprb.optimizer.rule.nsga2 import NSGA2Novelty_G_P
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation

from helpers import estimate_and_set_bounds, load_eggholder, init_rule_discovery_env, build_filename, visualize_rule_predictions
from metrics import summarize_rule_set, save_metrics_to_csv

def run():
    t0 = time()
    random_state = 42
    X, y = load_eggholder(n_samples=250, noise=0.2, random_state=random_state)

    mu = 20
    rule_discovery = NSGA2Novelty_G_P(
        n_iter=10,
        mu=mu,
        lmbda=64,
        origin_generation=origin.SquaredError(),
        mutation=mutation.Normal(sigma=1.22,
                                           matching_type=rule.matching.OrderedBound([-1, 1])),
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.MooFitness(),
            model=Ridge(alpha=0.01, random_state=random_state),
            matching_type=rule.matching.OrderedBound([-1, 1])
        ),
        fitness_objs=[
            lambda r: r.error_,
        ],
        fitness_objs_labels=[
            "Error",
        ],
        novelty_mode="P",
        min_experience=2,
        max_restarts=5,
        keep_archive_across_restarts=True,
    )
    init_rule_discovery_env(rule_discovery)

    n_rules = mu
    estimate_and_set_bounds(rule_discovery, X)

    generated_rules = rule_discovery.optimize(X, y, n_rules=n_rules)

    runtime = timedelta(seconds=time() - t0)
    print(f"\nTotal runtime: {runtime}")

    filename = build_filename(rule_discovery)
    visualize_rule_predictions(
        X,
        y,
        generated_rules,
        runtime,
        rule_discovery,
        filename,
        show_params=False,
        subtitle=" - N-P"
    )

    summary = summarize_rule_set(generated_rules, X, y)
    print("\nAverage error(MSE):", summary["average_error"])

    save_metrics_to_csv(
        summary,
        "results/results_metrics.csv",
        extra_info={
            "graph_filename": filename,
        }
    )


if __name__ == "__main__":
    run()