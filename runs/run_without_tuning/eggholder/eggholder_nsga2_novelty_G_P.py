from datetime import timedelta
from time import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from runs.run_without_tuning.eggholder.load_eggholder import load_eggholder_1D
from runs.run_without_tuning.eggholder.metrics_rules import save_metrics_to_csv, summarize_rule_set
from helpers.visualize_rule_predictions import build_filename, visualize_rule_predictions_1D

import suprb
from suprb import rule
from suprb.optimizer.rule import origin, mutation
from suprb.solution.base import Solution
from suprb.solution.fitness import ComplexityWu
from suprb.solution.mixing_model import ErrorExperienceHeuristic
from suprb.optimizer.rule.nsga2 import NSGA2Novelty_G_P
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.novelty_search_type import MinimalCriteria
from suprb.utils import estimate_bounds

import sys
import os

sys.path.insert(0, os.path.abspath("/home/vonproda/Desktop/BA/ba_suprb-experimentation"))

#pprint(sys.path)

def estimate_and_set_bounds(rule_discovery, X):
    bounds = estimate_bounds(X)
    for key, value in rule_discovery.get_params().items():
        if key.endswith("bounds") and value is None:
            print(f"Setting bounds for {key} based on data")
            rule_discovery.set_params(**{key: bounds})


def run():
    t0 = time()
    random_state = 42
    
    X, y = load_eggholder_1D(n_samples=250, noise=0.2, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    mu = 20
    nov_calc = NoveltyCalculation(
        k_neighbor=15,
        #novelty_search_type=MinimalCriteria(min_examples_matched=2)
    )
    rule_discovery = NSGA2Novelty_G_P(
        n_iter=10,
        mu=mu,
        lmbda=64,
        origin_generation=origin.SquaredError(),
        mutation=mutation.HalfnormIncrease(sigma=1.22,
                                           matching_type=rule.matching.OrderedBound([-1, 1])),
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.MooFitness(),
            model=Ridge(alpha=0.01, random_state=random_state),
            matching_type=rule.matching.OrderedBound([-1, 1])
        ),
        fitness_objs = [
            lambda r: r.error_,
        ],
        fitness_objs_labels = [
            "Error",
        ],
        novelty_mode="P",
        min_experience=2,
        max_restarts=5,
        keep_archive_across_restarts=True,
    )

    rule_discovery.pool_ = []
    rule_pool = []

    rule_discovery.elitist_ = Solution([0,0,0], [0,0,0], ErrorExperienceHeuristic(), ComplexityWu())

    n_rules = mu

    estimate_and_set_bounds(rule_discovery, X)
    new_rules = rule_discovery.optimize(X, y, n_rules=n_rules)

    rule_pool.extend(new_rules)  

    print("Generated rules:")
    for i, rule1 in enumerate(rule_pool, 1):
       print(f"Rule {i}: {rule1}. Volume: {rule1.volume_}. Error: {rule1.error_}")

    print("-" * 100)

    runtime = timedelta(seconds=time() - t0)
    print(f"\nTotal runtime: {runtime}")

    filename = build_filename(rule_discovery)
    visualize_rule_predictions_1D(X, y, rule_pool, runtime, rule_discovery, filename, show_params=False, subtitle=" - N-P")

    summary = summarize_rule_set(rule_pool, X, y)
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