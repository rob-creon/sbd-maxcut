import numpy as np
import scipy.cluster.vq
import tensorflow as tf
import scipy as sp
from sklearn.cluster import KMeans
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


class StandardLaplacianMethod(Algorithm):

    def solve_max_cut(self, G: nx.Graph):
        pass

    def solve_sbm(self, G: nx.Graph):

        # Compute matrices
        A = nx.adjacency_matrix(G)
        D = np.diag(np.array(G.degree(weight=None))[:, 1])
        L_norm = D - A

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        fiedler_index = np.argsort(eigenvalues)[1]  # Index of the second smallest eigenvalue
        fiedler_vector = eigenvectors[:, fiedler_index]

        # Recover communities
        S = np.squeeze(np.where(fiedler_vector >= 0))
        T = np.squeeze(np.where(fiedler_vector < 0))

        return S, T, fiedler_vector


class EnergyMethod(Algorithm):

    def solve_max_cut(self, G: nx.Graph):
        pass

    def init_vecs(self, num_vectors):
        angles = np.random.uniform(0, 2 * np.pi, num_vectors)
        x = np.cos(angles)
        y = np.cos(angles)
        return np.column_stack((x, y))

    def energy_loss(self, A, vectors):
        return tf.reduce_sum(tf.matmul(A, vectors) * vectors)

    def gradient_descent(self, A, num_vectors, lr=5e-4, iterations=3000, tolerance=1e-12):
        vectors = tf.Variable(self.init_vecs(num_vectors), dtype=tf.float64)
        A_tensor = tf.constant(A, dtype=tf.float64)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        for _ in range(iterations):
            with tf.GradientTape() as tape:
                loss = self.energy_loss(A_tensor, vectors)
            gradients = tape.gradient(loss, vectors)
            opt.apply_gradients([(gradients, vectors)])
            vectors.assign_sub(lr * gradients)
            vectors.assign(vectors)
            if tf.reduce_all(tf.abs(gradients) < tolerance):
                break
        return vectors.numpy()

    def solve_sbm(self, G: nx.Graph):
        A = nx.adjacency_matrix(G).toarray()
        opt_vectors = self.gradient_descent(A, len(A))

        # Calculate angles
        angles = np.arctan2(opt_vectors[:, 1], opt_vectors[:, 0]).reshape(-1, 1)
        angles_mod = np.mod(angles, 2*np.pi)

        cluster_labels = KMeans(n_clusters=2).fit_predict(angles_mod)
        S = np.where(cluster_labels[cluster_labels == 0])[0]
        T = np.where(cluster_labels[cluster_labels == 1])[0]

        return S, T, np.zeros(len(A))
