from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import algorithms


def asym_error(S_size_, T_size_, S_, T_):
    return len(S_[S_ >= S_size_]) + len(T_[T_ < S_size_])


def count_misses(S_size, T_size, S: np.ndarray, T: np.ndarray):
    return min(asym_error(S_size, T_size, S, T), asym_error(T_size, S_size, T, S))


def test_spectral(S_size, T_size, p, q, trials=10):
    """
    TODO will be deleted eventually
    :param n: number of vertices in a block
    :param p: in group probability
    :param q: cross group probability
    :param trials: number of trials to run
    :return:
    """

    algs = {
        # "$I-D^{-1/2}AD^{1/2}$": algorithms.NormalizedLaplacianMethod(),
        "$D-A$": algorithms.StandardLaplacianMethod(),
        # "$I-2A$": algorithms.AugmentedAdjacencyMethod(-1, 1),
        # "Energy": algorithms.EnergyMethod(),
        # "$A$": algorithms.AugmentedAdjacencyMethod(1, 0)
    }

    misses_ts = {}
    for trial in tqdm(range(trials)):
        num_blocks = 2
        Gp = [[p if i == j else q for j in range(num_blocks)]
              for i in range(num_blocks)]
        G = nx.stochastic_block_model([S_size, T_size], Gp)

        for alg_name in algs:
            solver = algs[alg_name]
            S, T, weights = solver.solve_sbm(G)
            if alg_name not in misses_ts:
                misses_ts[alg_name] = []

            misses_ts[alg_name].append(count_misses(S_size, T_size, S, T))

            if trial == trials-1 and weights is not None:
                plt.plot(weights, label=alg_name)
                if asym_error
                plt.scatter(S[S >= S_size], weights[S[S >= S_size]], label=alg_name)
                plt.scatter(T[T < S_size], weights[T[T < S_size]], label=alg_name)

    plt.title('Solver Weights')
    plt.xlabel('Vertex')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    for alg_name in algs:
        plt.plot(np.arange(1, trials+1), misses_ts[alg_name], label=f'{alg_name}, avg={np.mean(misses_ts[alg_name])}')
    plt.title('Spectral Error')
    plt.xlabel('Trial')
    plt.ylabel('Misses')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_spectral(10, 10, .53, .47)
