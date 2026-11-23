"""
PA-C EXTRA: compares gradient-descent-trained RBF-Net weights against weights
computed directly via the Moore-Penrose-Pseudo-Inverse (assignment 29),
RBFNet.fit_weights_pinv, for three regimes: P=K, P slightly > K, P >> K.
Reports accuracy (test error) and time for both methods.
"""

import time

import numpy as np

from rbf import RBFNet


def load_patterns(path, n_inputs):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    return data[:, :n_inputs], data[:, n_inputs:]


N_INPUTS, N_OUTPUTS = 2, 1
K = 20
ETA = 0.05
N_STEPS = 60000
SEED = 0

X_all, Y_all = load_patterns("training_data.txt", N_INPUTS)
X_test, Y_test = load_patterns("test_data.txt", N_INPUTS)

cases = {"P = K": K, "P slightly > K": K + 5, "P >> K": len(X_all)}

lines = [f"RBF-Net {N_INPUTS}-{K}-{N_OUTPUTS}: gradient descent vs. Moore-Penrose-Pseudo-Inverse\n"]
for label, P in cases.items():
    X, Y = X_all[:P], Y_all[:P]

    # gradient descent
    net_gd = RBFNet(N_INPUTS, K, N_OUTPUTS, SEED)
    net_gd.set_centers_input_driven(X)
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        p = net_gd.rng.integers(0, len(X))
        net_gd.train_step(X[p], Y[p], ETA)
    t_gd = time.perf_counter() - t0
    err_gd = np.mean([0.5 * np.sum((y_hat - net_gd.predict(x)) ** 2) for x, y_hat in zip(X_test, Y_test)])

    # Moore-Penrose-Pseudo-Inverse (same centers, for a fair comparison)
    net_pi = RBFNet(N_INPUTS, K, N_OUTPUTS, SEED)
    net_pi.C, net_pi.s = net_gd.C, net_gd.s
    t0 = time.perf_counter()
    net_pi.fit_weights_pinv(X, Y)
    t_pi = time.perf_counter() - t0
    err_pi = np.mean([0.5 * np.sum((y_hat - net_pi.predict(x)) ** 2) for x, y_hat in zip(X_test, Y_test)])

    lines.append(f"{label}  (P={P}, K={K}):")
    lines.append(f"  gradient descent ({N_STEPS} steps): mean test error = {err_gd:.6f}, time = {t_gd:.3f}s")
    lines.append(f"  Moore-Penrose-Pseudo-Inverse:        mean test error = {err_pi:.6f}, time = {t_pi:.5f}s")
    lines.append("")

with open("pinv_compare.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
