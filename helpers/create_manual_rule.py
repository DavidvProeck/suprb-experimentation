import numpy as np
from suprb import rule
from sklearn.linear_model import Ridge
from suprb.rule.base import Rule



def create_manual_rule(X, y, low, high):
    n_dims = X.shape[1]

    input_space = np.array([[low, high]] * n_dims)
    
    matching = rule.matching.OrderedBound(input_space)
    fitness = rule.fitness.VolumeWu()
    model = Ridge(alpha=0.01)

    manual_rule = Rule(match=matching, input_space=input_space, model=model, fitness=fitness)

    manual_rule.fit(X, y)

    return manual_rule