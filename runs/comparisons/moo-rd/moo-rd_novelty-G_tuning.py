import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import origin, mutation
import suprb.solution.mixing_model as mixing_model

from suprb.optimizer.rule import nsga2
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.novelty_search_type import MinimalCriteria

from suprb.optimizer.rule.nsga2 import NSGA2Novelty_G_P

random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
@click.option('-r', '--rule_amount', type=click.INT, default=1)
@click.option('-f', '--filter_subpopulation', type=click.STRING, default='FilterSubpopulation')
@click.option('-e', '--experience_calculation', type=click.STRING, default='ExperienceCalculation')
@click.option('-w', '--experience_weight', type=click.INT, default=1)
@click.option('-n', '--study_name', type=click.STRING, default=None)
def run(problem: str, job_id: str, rule_amount: int, filter_subpopulation: str,
        experience_calculation: str, experience_weight: int, study_name: str):
    print(f"Problem is {problem}, with job id {job_id}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_discovery=NSGA2Novelty_G_P(
            n_iter=16,     # <- tuned
            mu=16,         # <- tuned
            lmbda=64,      # <- tuned
            origin_generation=origin.SquaredError(),
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.MooFitness(),  # dummy fitness for MOO
                model=Ridge(alpha=0.01, random_state=random_state),
            ),
            mutation=mutation.Normal(
                sigma=1.22  # <- tuned
            ),
            fitness_objs=[lambda r: r.error_],
            fitness_objs_labels=["Error"], # novelty objective is added internally
            novelty_calc=NoveltyCalculation(
                k_neighbor=15,
                #novelty_search_type=MinimalCriteria(min_examples_matched=15)  # <- tuned #TODO: Leads to warnings in crowding distance calculation.
            ),
            novelty_mode="G",
            profile=False,
            min_experience=2, #Rules that match only one sample are considered trivial, so min_experience >= 2
            max_restarts=4,
            keep_archive_across_restarts=True,
        ),
        solution_composition=ga.GeneticAlgorithm(
            n_iter=32,
            population_size=32,
            selection=ga.selection.Tournament()
        ),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=60*60*24*3,  # 72 hours
        # scoring='neg_mean_squared_error', #TODO mention in discussion
        scoring='fitness',
        verbose=10
    )

    @param_space()
    def suprb_NS_GA_space(trial: Trial, params: Bunch):
        params.rule_discovery__mu = trial.suggest_int('rule_discovery__mu', 8, 64, step=4)

        lam_min = max(32, params.rule_discovery__mu)
        params.rule_discovery__lmbda = trial.suggest_int('rule_discovery__lmbda', lam_min, 256, step=16)

        params.rule_discovery__n_iter = trial.suggest_int('rule_discovery__n_iter', 8, 128, step=4)

        sigma_hi = max(1.5, float(np.sqrt(X.shape[1])) * 2.0)
        params.rule_discovery__mutation__sigma = trial.suggest_float(
            'rule_discovery__mutation__sigma', 0.05, sigma_hi
        )

        # params.rule_discovery__novelty_calc__novelty_search_type__min_examples_matched = trial.suggest_int(
        #     'rule_discovery__min_examples_matched', 0, 30
        # )

        # params.rule_discovery__novelty_mode = trial.suggest_categorical(
        #     'rule_discovery__novelty_mode', ['G', 'P']
        # )

        params.rule_discovery__min_experience = trial.suggest_int(
            'rule_discovery__min_experience', 2, 32
        )

        # GA is fixed

        # Mixing
        params.solution_composition__init__mixing__filter_subpopulation__rule_amount = rule_amount
        params.solution_composition__init__mixing__experience_weight = experience_weight
        params.solution_composition__init__mixing__filter_subpopulation = getattr(
            mixing_model, filter_subpopulation)()
        params.solution_composition__init__mixing__experience_calculation = getattr(
            mixing_model, experience_calculation)()

        if isinstance(params.solution_composition__init__mixing__experience_calculation,
                      mixing_model.CapExperienceWithDimensionality):
            params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_float(
                'solution_composition__init__mixing__experience_calculation__upper_bound', 2, 5)
        else:
            params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_int(
                'solution_composition__init__mixing__experience_calculation__upper_bound', 20, 50)


    experiment_name = f'MOO-RD-Novelty-G-tuning j:{job_id} p:{problem}; r:{rule_amount}; f:{filter_subpopulation}; -e:{experience_calculation}' or study_name
    print(experiment_name)
    experiment = Experiment(name=experiment_name, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_NS_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(
        evaluation,
        cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state),
        n_jobs=8
    )

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
