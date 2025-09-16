import cProfile
import pstats

import numpy as np
from typing import Optional, List, Callable, Tuple
from joblib import Parallel, delayed

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.utils import RandomState
from ..base import MultiRuleDiscovery
from ..origin import Matching, SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, HalfnormIncrease
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint
from .nsga2_helpers import visualize_pareto_front

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance, FunctionalDiversity


class NSGA2(MultiRuleDiscovery):
    """
    Adapted from: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II by Kalyanmoy Deb et al.
    Uses Crowding Distance and NonDominatedSorting from pymoo package.
    """
    def __init__(
            self,
            n_iter: int = 32,
            mu: int = 50,
            lmbda: int = 100,
            origin_generation: RuleOriginGeneration = SquaredError(),
            init: RuleInit = MeanInit(),
            mutation: RuleMutation = HalfnormIncrease(sigma=1.22),
            constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
            acceptance: RuleAcceptance = Variance(),
            random_state: int = None,
            n_jobs: int = 1,
            fitness_objs: Optional[List[Callable[[Rule], float]]] = None,
            fitness_objs_labels: Optional[List[str]] = None,
            profile: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            origin_generation=origin_generation,
            init=init,
            acceptance=acceptance,
            constraint=constraint,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.mu = mu
        self.lmbda = lmbda
        self.mutation = mutation
        self.constraint = constraint

        if fitness_objs is None:
            fitness_objs = [
                lambda r: r.error_,
                lambda r: -r.volume_,
            ]

        # Change fitness objectives List to tuple avoid accidental changes.
        # Extra objectives in subclasses are added during runtime to avoid sklearn.clone errors.
        self.fitness_objs: Tuple[Callable[[Rule], float], ...] = tuple(fitness_objs)

        if fitness_objs_labels is None:
            fitness_objs_labels = [f"obj_{i}" for i in range(len(self.fitness_objs))]
        self.fitness_objs_labels: Tuple[str, ...] = tuple(fitness_objs_labels)

        self.profile = profile

    def _optimize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            initial_rule: Rule,
            random_state: RandomState
    ) -> Optional[List[Rule]]:
        profiler = cProfile.Profile() if self.profile else None
        if profiler:
            profiler.enable()

        population = []

        origins = self.origin_generation(
            n_rules=self.mu,
            X=X,
            y=y,
            pool=self.pool_,
            elitist=self.elitist_,
            random_state=random_state,
        )

        population = Parallel(n_jobs=self.n_jobs)(
            delayed(self._init_valid_origin)(origin, X, y, random_state)
            for origin in origins
        )

        population = [p for p in population if p is not None]

        for _ in range(self.n_iter):
            parents = random_state.choice(population, size=self.lmbda, replace=True)

            children = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_valid_child)(parent, X, y, random_state)
                for parent in parents
            )

            children = [c for c in children if c is not None]

            population_combined = population + children

            pareto_fronts = self._fast_nondominated_sort(population_combined)
            population = self._build_next_population(pareto_fronts)

        pareto_front = pareto_fronts[0] if pareto_fronts else []
        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)

        visualize_pareto_front(self, pareto_front)
        return pareto_front

# ────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────
    def _fast_nondominated_sort(self, population: List[Rule]) -> List[List[Rule]]:
        if not population:
            return []
        objs = self._fitness_objs_runtime()
        obj_matrix = np.vstack(
            [[obj(rule) for obj in objs] for rule in population]
        )

        fronts = NonDominatedSorting().do(obj_matrix, only_non_dominated_front=False)
        return [[population[i] for i in front] for front in fronts]


    def _assign_crowding_distance(
            self,
            front: List[Rule],
            cd_func,
    ):
        if not front:
            return

        objs = self._fitness_objs_runtime()
        obj_matrix = np.vstack(
            [[obj(rule) for obj in objs] for rule in front]
        )
        crowding_distances = cd_func.do(obj_matrix)

        for rule, dist in zip(front, crowding_distances):
            rule.crowding_distance_ = dist


    def _init_valid_origin(
            self,
            origin,
            X,
            y,
            random_state,
    ):
        rule = self.constraint(self.init(mean=origin, random_state=random_state)).fit(X, y)
        return rule if rule.is_fitted_ and rule.experience_> 0 else None


    def _generate_valid_child(
        self,
        parent,
        X,
        y,
        random_state,
    ):
        child = self.constraint(self.mutation(parent, random_state)).fit(X,y)
        return child if child.is_fitted_ and child.experience_ > 0 else None


    def _build_next_population(
            self,
            pareto_fronts,
    ):
        population_new = []

        cd_func = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=True)

        for front in pareto_fronts:
            self._assign_crowding_distance(front, cd_func)
            if len(population_new) + len(front) <= self.mu:
                population_new.extend(front)
            else:
                front_sorted = sorted(front, key=lambda rule: -rule.crowding_distance_)
                population_new.extend(front_sorted[:self.mu - len(population_new)])
                break

        return population_new


    def _fitness_objs_runtime(self) -> List[Callable[[Rule], float]]:
        return list(self.fitness_objs)


    def _fitness_labels_runtime(self) -> List[str]:
        return list(self.fitness_objs_labels)