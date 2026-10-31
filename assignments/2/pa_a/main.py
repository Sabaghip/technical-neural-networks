"""
PA-A: trains a Multi-Layer-Perceptron with Backpropagation of Error on
training_data.txt and evaluates it on test_data.txt.

Demo task: XOR. Chosen because Assignment 9 proves a single-layer Perceptron
cannot solve it -- this demonstrates a 2-4-1 MLP (one hidden layer) can.

Usage:  python3 main.py
Needs:  numpy, matplotlib
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: just save a PNG, no display needed
import matplotlib.pyplot as plt

from mlp import MLP


def load_patterns(path):
    """Read a data file in the format required by the assignment: two header
    lines starting with '#', then P lines of 'x_1 ... x_N  yhat_1 ... yhat_M'
    separated by one or more blanks."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    return np.array(rows)


# ---- Network structure & hyperparameters (set directly in source, as allowed) ----
LAYER_SIZES = [2, 4, 1]                 # N-H-M = 2-4-1
TRANSFER_NAMES = ["tanh", "logistic"]   # hidden layer: tanh, output layer: logistic
LEARNING_RATES = [0.5, 0.5]             # one eta per weight layer
SEED = 42
N_TRAINING_STEPS = 5000

N_INPUTS = LAYER_SIZES[0]

train = load_patterns("training_data.txt")
test = load_patterns("test_data.txt")
X_train, Y_train = train[:, :N_INPUTS], train[:, N_INPUTS:]
X_test, Y_test = test[:, :N_INPUTS], test[:, N_INPUTS:]

net = MLP(LAYER_SIZES, TRANSFER_NAMES, LEARNING_RATES, SEED)  # step 1: initialization

learning_curve = []
for step in range(N_TRAINING_STEPS):
    p = net.rng.integers(0, len(X_train))         # step 2: pick a pattern p (reproducible)
    error = net.train_step(X_train[p], Y_train[p])  # steps 3-6: forward/compare/backward/update
    learning_curve.append(error)
    # step 7: ready? -- stop once N_TRAINING_STEPS is reached (loop end)

with open("learning_curve.txt", "w") as f:
    f.write("# training step    quadratic error E\n")
    for step, error in enumerate(learning_curve):
        f.write(f"{step} {error:.6f}\n")

plt.figure(figsize=(7, 4))
plt.plot(learning_curve)
plt.xlabel("training step")
plt.ylabel("quadratic error E")
plt.title("Learning curve (XOR, 2-4-1 MLP)")
plt.tight_layout()
plt.savefig("learning_curve.png")
print("learning curve written to learning_curve.txt / learning_curve.png")

print("\nTesting (no further weight changes):")
test_errors = []
for x, y_hat in zip(X_test, Y_test):
    y = net.predict(x)
    e = 0.5 * np.sum((y_hat - y) ** 2)
    test_errors.append(e)
    print(f"  input={x}  teacher={y_hat}  output={np.round(y, 4)}  error={e:.6f}")
print(f"mean test error: {np.mean(test_errors):.6f}")
