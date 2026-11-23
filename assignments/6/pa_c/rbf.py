"""
Radial-Basis-Function Network (RBF-Net), implemented from scratch (numpy only
for array arithmetic -- no RBF/neural-network library).

N-K-M network: N inputs, K RBF neurons (Gaussian), M linear output neurons,
matching the lecture's structure (assignments 27-32):

    r_k(X) = exp(-||X - C_k||^2 / (2 s_k^2))      RBF-layer output
    y_m    = sum_k w_km * r_k(X)                   linear output layer

Centers/widths C_k, s_k: set via the "Input Data Driven" philosophy
(assignment 28) using k-means clustering (own implementation) on the
training inputs; each width is set from the distance to its nearest
neighboring center, a standard heuristic that gives overlapping RBF bumps.

Weights w_km: trained with online gradient descent (lecture's own rule,
sheet 6 slide "Case P>K: Learn weights"):

    Delta w_km = eta * (yhat_m - y_m) * r_k

(no transfer-function derivative needed -- the output layer is linear.)
"""

import numpy as np


class RBFNet:
    def __init__(self, n_inputs, n_rbf, n_outputs, seed):
        self.N, self.K, self.M = n_inputs, n_rbf, n_outputs
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.uniform(-0.5, 0.5, size=(self.K, self.M))  # w_km, as required
        self.C = None  # K x N RBF centers
        self.s = None  # K RBF widths

    def _kmeans(self, X, n_iters=100):
        """Own from-scratch k-means: find K centers among/near the input patterns."""
        idx = self.rng.choice(len(X), size=self.K, replace=False)
        C = X[idx].copy()
        for _ in range(n_iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # P x K
            assign = d2.argmin(axis=1)
            new_C = np.array(
                [X[assign == k].mean(axis=0) if np.any(assign == k) else C[k] for k in range(self.K)]
            )
            if np.allclose(new_C, C):
                break
            C = new_C
        return C

    def set_centers_input_driven(self, X):
        """'Input Data Driven' philosophy (assignment 28): k-means on the input
        data for the centers; width = distance to the nearest neighboring center."""
        self.C = self._kmeans(np.asarray(X, dtype=float))
        d = np.sqrt(((self.C[:, None, :] - self.C[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        nearest[~np.isfinite(nearest)] = 1.0  # fallback if K==1
        self.s = np.maximum(nearest, 1e-3)

    def _rbf_outputs(self, x):
        """r_k for one input pattern x -> length-K vector."""
        d2 = ((self.C - x) ** 2).sum(axis=1)
        return np.exp(-d2 / (2.0 * self.s ** 2))

    def forward(self, x):
        r = self._rbf_outputs(x)
        y = self.W.T @ r
        return r, y

    def train_step(self, x, y_hat, eta):
        """One online gradient-descent step on a single pattern (x, y_hat)."""
        r, y = self.forward(x)
        delta = y_hat - y                   # length M, output layer is linear (f'=1)
        self.W += eta * np.outer(r, delta)  # Delta w_km = eta * delta_m * r_k
        return 0.5 * np.sum(delta ** 2)

    def predict(self, x):
        return self.forward(x)[1]

    def fit_weights_pinv(self, X, Y):
        """EXTRA: Moore-Penrose-Pseudo-Inverse weight calculation (assignment 29),
        R+ = (R^T R)^-1 R^T, W = R+ Y. Uses numpy's pinv, as explicitly permitted
        for this extra part."""
        R = np.array([self._rbf_outputs(x) for x in X])  # P x K
        R_plus = np.linalg.pinv(R)                         # K x P
        self.W = R_plus @ np.asarray(Y, dtype=float)        # K x M
