from datetime import timedelta
from time import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from runs.run_without_tuning.eggholder.load_eggholder import load_eggholder_1D
from helpers.create_manual_rule import create_manual_rule
from helpers.visualize_rule_predictions import build_filename, visualize_rule_predictions_1D

import suprb
from suprb import rule
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule import origin, mutation
from suprb.solution.base import Solution
from suprb.solution.fitness import PseudoBIC, ComplexityWu
from suprb.solution.mixing_model import ErrorExperienceHeuristic
from suprb.optimizer.rule.origin import RuleOriginGeneration
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.utils import estimate_bounds


def estimate_and_set_bounds(rule_discovery, X):
    bounds = estimate_bounds(X)
    for key, value in rule_discovery.get_params().items():
        if key.endswith("bounds") and value is None:
            print(f"Setting bounds for {key} based on data")
            rule_discovery.set_params(**{key: bounds})


def run():
    t0 = time()
    random_state = 42
    
    #X, y = load_smooth_eggholder_nd(n_samples=500, n_dims=1,noise=0.01,
    # random_state=random_state)
    #X, y = load_linear_nd(n_samples=10000, n_dims=1, weights=None,
    #                      intercept=0.0, noise=0.1, random_state=random_state)
    X, y = load_eggholder_1D(n_samples=250, noise=0.2, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    rule_discovery = ES1xLambda(
        n_iter=500,
        operator='&',
        origin_generation=origin.SquaredError(),
        mutation=mutation.HalfnormIncrease(sigma=1.2,
                                           matching_type=rule.matching.OrderedBound([-1, 1])),
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.VolumeWu(),
            model=Ridge(alpha=0.01, random_state=random_state),
            matching_type=rule.matching.OrderedBound([-1, 1])),

            #constraint=suprb.optimizer.rule.constraint.MinRange(),
            )
    
    rule_discovery.pool_ = []
    rule_pool = []
    elitist_rule = create_manual_rule(X, y, -0.5, 0.5)
    #rule_pool.append(elitist_rule)

    #rule_discovery.elitist_ = Solution([0,0,0], [0,0,0], ErrorExperienceHeuristic(), PseudoBIC())
    rule_discovery.elitist_ = Solution([0,0,0], [0,0,0], ErrorExperienceHeuristic(), ComplexityWu())
    #elitist_rule = create_manual_rule(X, y, -0.066, 1.5)
    #rule_discovery.elitist_ = Solution([1], [elitist_rule], ErrorExperienceHeuristic(), PseudoBIC())
    #rule_discovery.elitist_ = None
    
    n_rules = 16

    estimate_and_set_bounds(rule_discovery, X)
    new_rules = rule_discovery.optimize(X, y, n_rules=n_rules)
    #new_rules = rule_discovery._optimize(X, y, initial_rule=elitist_rule, random_state=random_state)
    #rule_pool.append(new_rules)
    rule_pool.extend(new_rules)  
    #rule_pool.append(create_manual_rule(X, y, -0.066, 0.15))

    print("Generated rules:")
    for i, rule1 in enumerate(rule_pool, 1):
       print(f"Rule {i}: {rule1}. Volume: {rule1.volume_}. Error: {rule1.error_}")

    print("-" * 100)
    runtime = timedelta(seconds=time() - t0)
    filename = build_filename(rule_discovery)
    filename = f"{filename}" + ".png"
    visualize_rule_predictions_1D(X, y, rule_pool, runtime, rule_discovery, filename, show_params=False, subtitle=" - ES")


if __name__ == "__main__":
    run()