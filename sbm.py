import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import algorithms


def count_misses(S: np.ndarray, T: np.ndarray, n):
    le1 = S[S >= n]
    re1 = T[T < n]
    error1 = len(le1) + len(re1)
    le2 = S[S > n]
    re2 = T[T <= n]
    error2 = len(le2) + len(re2)
    return min(error1, error2)


def test_spectral(n, p, q, trials=50):
    """
    TODO will be deleted eventually
    :param n: number of vertices in a block
    :param p: in group probability
    :param q: cross group probability
    :param trials: number of trials to run
    :return:
    """

    algs = {
        "Normalized Laplacian": algorithms.NormalizedLaplacianMethod(),
        "AugAdj{-1,1}": algorithms.AugmentedAdjacencyMethod(-1, 1),
        "DefAdj": algorithms.AugmentedAdjacencyMethod(1, 0)
    }

    misses_ts = {}
    for trial in tqdm(range(trials)):
        num_blocks = 2
        Gp = [[p if i == j else q for j in range(num_blocks)]
              for i in range(num_blocks)]
        G = nx.stochastic_block_model([n, n], Gp)

        for alg_name in algs:
            solver = algs[alg_name]
            S, T, weights = solver.solve_sbm(G)
            if alg_name not in misses_ts:
                misses_ts[alg_name] = []
            misses_ts[alg_name].append(count_misses(S, T, n))

            if trial == trials-1 and weights is not None:
                plt.plot(weights, label=alg_name)

    plt.title('Solver Weights')
    plt.xlabel('Vertex')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    for alg_name in algs:
        plt.plot(np.arange(1, trials+1), misses_ts[alg_name], label=alg_name)
    plt.title('Spectral Error')
    plt.xlabel('Trial')
    plt.ylabel('Misses')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_spectral(100, 0.53, 0.47)
