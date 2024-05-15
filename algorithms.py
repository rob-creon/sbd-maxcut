import numpy as np
# import tensorflow as tf
import scipy as sp
import networkx as nx

from abc import ABC, abstractmethod


class Algorithm(ABC):

    @abstractmethod
    def solve_max_cut(self, G: nx.Graph):
        """
        Approximates the max-cut of the graph.
        :return: a vertex cut (S,T)
        """
        pass

    @abstractmethod
    def solve_sbm(self, G: nx.Graph):
        """
        Attempts to recovers the two communities from a 2-SBM (Stochastic block
        model).
        :return: communities (S,T), [optional] weights
        """
        pass


class AugmentedAdjacencyMethod(Algorithm):
    def __init__(self, one_val: float, zero_val: float):
        self.one = one_val
        self.zero = zero_val

    def solve_max_cut(self, G: nx.Graph):
        pass

    def solve_sbm(self, G: nx.Graph):
        # Augment adjacency matrix
        A = nx.adjacency_matrix(G).toarray()
        A[A == 1] = self.one
        A[A == 0] = self.zero

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        idx = np.argsort(eigenvalues)[0]  # Index of the smallest eigenvalue
        vec = eigenvectors[:, idx]

        # Recover communities
        S = np.squeeze(np.where(vec >= 0))
        T = np.squeeze(np.where(vec < 0))

        return S, T, vec


class NormalizedLaplacianMethod(Algorithm):

    def solve_max_cut(self, G: nx.Graph):
        pass

    def solve_sbm(self, G: nx.Graph):

        # Compute matrices
        A = nx.adjacency_matrix(G)
        D = np.diag(np.array(G.degree(weight=None))[:, 1])
        L_norm = np.eye(len(G)) - np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        fiedler_index = np.argsort(eigenvalues)[1]  # Index of the second smallest eigenvalue
        fiedler_vector = eigenvectors[:, fiedler_index]

        # Recover communities
        S = np.squeeze(np.where(fiedler_vector >= 0))
        T = np.squeeze(np.where(fiedler_vector < 0))

        return S, T, fiedler_vector
