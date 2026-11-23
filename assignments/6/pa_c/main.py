"""
PA-C: trains an RBF-Net (rbf.RBFNet) on training_data.txt and evaluates it on
test_data.txt (same data format as PA-A/PA-B: two '#' header lines, then
'x_1 ... x_N  yhat_1 ... yhat_M' per line).

Demo task: reused from PA-A, dataset 04 -- a 2D continuous regression task
(350 patterns, 80/20 train/test split) -- so the RBF-Net's performance can be
compared directly against the MLP trained on the very same data in PA-A.

Usage: python3 main.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


N_INPUTS, N_RBF, N_OUTPUTS = 2, 20, 1
ETA = 0.05
N_STEPS = 60000
SEED = 0

X_train, Y_train = load_patterns("training_data.txt", N_INPUTS)
X_test, Y_test = load_patterns("test_data.txt", N_INPUTS)

net = RBFNet(N_INPUTS, N_RBF, N_OUTPUTS, SEED)
net.set_centers_input_driven(X_train)  # "Input Data Driven" philosophy (assignment 28)

learning_curve = []
for step in range(N_STEPS):
    p = net.rng.integers(0, len(X_train))
    error = net.train_step(X_train[p], Y_train[p], ETA)
    learning_curve.append(error)

with open("learning_curve.txt", "w") as f:
    f.write("# training step    quadratic error E\n")
    for step, error in enumerate(learning_curve):
        f.write(f"{step} {error:.6f}\n")

plt.figure(figsize=(7, 4))
plt.plot(learning_curve, linewidth=0.5)
plt.xlabel("training step")
plt.ylabel("quadratic error E")
plt.title(f"RBF-Net learning curve ({N_INPUTS}-{N_RBF}-{N_OUTPUTS}, gradient descent)")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=130)
plt.close()

test_errors = [0.5 * np.sum((y_hat - net.predict(x)) ** 2) for x, y_hat in zip(X_test, Y_test)]
print(f"RBF-Net {N_INPUTS}-{N_RBF}-{N_OUTPUTS}, {N_STEPS} gradient-descent steps, eta={ETA}")
print(f"mean test error: {np.mean(test_errors):.6f}   max test error: {np.max(test_errors):.6f}")

with open("results.txt", "w") as f:
    f.write(f"RBF-Net {N_INPUTS}-{N_RBF}-{N_OUTPUTS}, {N_STEPS} gradient-descent steps, eta={ETA}\n")
    f.write(f"mean test error: {np.mean(test_errors):.6f}\n")
    f.write(f"max test error: {np.max(test_errors):.6f}\n\n")
    f.write("x1 x2 teacher output\n")
    for x, y_hat in zip(X_test, Y_test):
        y = net.predict(x)
        f.write(f"{x[0]:.4f} {x[1]:.4f}  {y_hat[0]:.4f}  {y[0]:.4f}\n")

# depict: teacher vs. network output over the test set, and the RBF centers
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc0 = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:, 0], cmap="coolwarm", s=25)
axes[0].scatter(net.C[:, 0], net.C[:, 1], c="k", marker="x", s=60, label="RBF centers")
axes[0].set_title("teacher (test set)")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].legend()
fig.colorbar(sc0, ax=axes[0])
preds = np.array([net.predict(x)[0] for x in X_test])
sc1 = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=preds, cmap="coolwarm", s=25)
axes[1].scatter(net.C[:, 0], net.C[:, 1], c="k", marker="x", s=60)
axes[1].set_title("RBF-Net output (test set)")
axes[1].set_xlabel("x1")
fig.colorbar(sc1, ax=axes[1])
fig.suptitle("Teacher vs. RBF-Net output, with RBF centers")
fig.tight_layout()
fig.savefig("prediction_vs_teacher.png", dpi=130)
plt.close(fig)
print("saved learning_curve.txt/.png, results.txt, prediction_vs_teacher.png")
