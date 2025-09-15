from datetime import timedelta, datetime
from time import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from helpers.project_root import get_project_root
from runs.run_without_tuning.eggholder.load_eggholder import load_eggholder_1D
from helpers.visualize_rule_predictions import build_filename, visualize_rule_predictions_1D

import suprb
from suprb import rule
from suprb.optimizer.rule import origin, mutation
from suprb.solution.base import Solution
from suprb.solution.fitness import PseudoBIC, ComplexityWu
from suprb.solution.mixing_model import ErrorExperienceHeuristic
from suprb.optimizer.rule.origin import RuleOriginGeneration
from suprb.optimizer.rule.mutation import Normal, Halfnorm, HalfnormIncrease, Uniform, UniformIncrease
from suprb.optimizer.rule.nsga2 import NSGA2
from suprb.utils import estimate_bounds

import sys
import os
sys.path.insert(0, os.path.abspath("/home/david/Desktop/BA/ba_suprb-experimentation"))


# ────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────
def estimate_and_set_bounds(rule_discovery, X):
    bounds = estimate_bounds(X)
    for key, value in rule_discovery.get_params().items():
        if key.endswith("bounds") and value is None:
            print(f"Setting bounds for {key} based on data")
            rule_discovery.set_params(**{key: bounds})


def make_experiment_dir():
    project_root = get_project_root()
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = project_root / f"RD_NSGA2_Mutation_Variants_{tstamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created experiment folder: {exp_dir}")
    return exp_dir


def base_rule_discovery(mutation, random_state=42):
    n_iter = 64
    mu = 16
    lmbda = 128
    n_jobs = 4

    origin_generation = origin.SquaredError()

    init = rule.initialization.MeanInit(
        fitness=rule.fitness.VolumeWu(),
        model=Ridge(alpha=0.01, random_state=random_state),
        matching_type=rule.matching.OrderedBound([-1, 1]))

    fitness_objs = [
        lambda r: r.error_,
        lambda r: -r.volume_,
    ]
    fitness_objs_labels = [
        "Error",
        "Volume",
    ]

    rule_discovery = NSGA2(
        n_iter=n_iter,
        mu=mu,
        lmbda=lmbda,
        origin_generation=origin_generation,
        mutation=mutation,
        init=init,
        n_jobs=n_jobs,
        fitness_objs=fitness_objs,
        fitness_objs_labels=fitness_objs_labels,
    )

    rule_discovery.pool_ = []
    rule_discovery.elitist_ = Solution([0, 0, 0], [0, 0, 0], ErrorExperienceHeuristic(), ComplexityWu())

    return rule_discovery


def run():
    t0 = time()
    random_state = 42

    X, y = load_eggholder_1D(n_samples=200, noise=0.2, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    exp_dir = make_experiment_dir()

    #TODO: Try out different sigma values
    sigma = 1.22

    #TODO: Look at different matching_types
    matching_type = rule.matching.OrderedBound([-1, 1])

    #TODO: Experiment with the wrapper class SigmaRange
    mutation_variants = {
        "Normal": Normal(matching_type=matching_type, sigma=sigma),
        "Halfnorm": Halfnorm(matching_type=matching_type, sigma=sigma),
        "HalfnormIncrease": HalfnormIncrease(matching_type=matching_type, sigma=sigma),
        "Uniform": Uniform(matching_type=matching_type, sigma=sigma),
        "UniformIncrease": UniformIncrease(matching_type=matching_type, sigma=sigma),
    }
    
    for mut_name, mut in mutation_variants.items():
        print("-" * 100)
        print(f"Running RD with mutation: {mut_name}")
        rule_discovery = base_rule_discovery(mut, random_state=random_state)

        estimate_and_set_bounds(rule_discovery, X)

        n_rules = rule_discovery.get_params().get("mu", 16)
        rule_pool = rule_discovery.optimize(X, y, n_rules=n_rules)

        print("Generated rules:")
        for i, r in enumerate(rule_pool, 1):
            print(f"Rule {i}: {r}. Volume: {r.volume_}. Error: {r.error_}")

        runtime = timedelta(seconds=time() - t0)
        print(f"Runtime: {runtime}")

        base_name = build_filename(rule_discovery)
        out_stem = f"{base_name}__{mut_name}"
        out_path_stem = exp_dir / f"{out_stem}.png"

        subtitle = " - " + mut_name + " Mutation"
        visualize_rule_predictions_1D(X, y, rule_pool, runtime, rule_discovery, str(out_path_stem), show_params=False, subtitle=subtitle)

    print("-" * 100)
    print("All mutation variants completed.")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    run()


