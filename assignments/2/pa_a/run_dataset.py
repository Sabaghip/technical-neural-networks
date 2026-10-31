"""
Runs the PA-A MLP (mlp.MLP) on one of the officially provided
PA-A_training_data_0X.txt files and writes all results into the same
directory as the data file: the actual train/test split used, the
learning curve (.txt + .png), and the test predictions (results.txt).

Usage:
    python3 run_dataset.py data/01/PA-A_training_data_01.txt \
        --layers 2 4 1 --transfer tanh tanh --lr 0.5 0.5 \
        --steps 5000 --seed 10 --test-fraction 0
"""

import argparse
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp import MLP


def load_patterns(path, n_inputs):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    return data[:, :n_inputs], data[:, n_inputs:]


def write_patterns(path, source_name, X, Y):
    with open(path, "w") as f:
        f.write(f"# derived from {source_name}\n")
        f.write(f"# P={len(X)}    N={X.shape[1]}    M={Y.shape[1]}\n")
        for x, y in zip(X, Y):
            f.write(" ".join(f"{v:g}" for v in x) + "   " + " ".join(f"{v:g}" for v in y) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--transfer", nargs="+", required=True)
    ap.add_argument("--lr", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="held-out fraction for testing; 0 = test on the full pattern set (used for the small, exhaustive datasets)",
    )
    args = ap.parse_args()

    data_path = pathlib.Path(args.data_file)
    outdir = data_path.parent
    n_inputs = args.layers[0]

    X, Y = load_patterns(data_path, n_inputs)

    rng = np.random.default_rng(args.seed)
    if args.test_fraction > 0:
        idx = rng.permutation(len(X))
        n_test = max(1, int(len(X) * args.test_fraction))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    else:
        train_idx = test_idx = np.arange(len(X))

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    write_patterns(outdir / "training_data.txt", data_path.name, X_train, Y_train)
    write_patterns(outdir / "test_data.txt", data_path.name, X_test, Y_test)

    net = MLP(args.layers, args.transfer, args.lr, args.seed)  # step 1: initialization

    learning_curve = []
    for step in range(args.steps):
        p = net.rng.integers(0, len(X_train))           # step 2: pick a pattern p
        error = net.train_step(X_train[p], Y_train[p])  # steps 3-6
        learning_curve.append(error)

    with open(outdir / "learning_curve.txt", "w") as f:
        f.write("# training step    quadratic error E\n")
        for step, error in enumerate(learning_curve):
            f.write(f"{step} {error:.6f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(learning_curve)
    plt.xlabel("training step")
    plt.ylabel("quadratic error E")
    plt.title(f"Learning curve ({data_path.stem})")
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve.png")
    plt.close()

    lines = [
        f"dataset: {data_path.name}",
        f"architecture: {'-'.join(map(str, args.layers))}   transfer={args.transfer}   "
        f"lr={args.lr}   steps={args.steps}   seed={args.seed}",
        f"train patterns: {len(X_train)}   test patterns: {len(X_test)}",
        "",
        "Testing (no further weight changes):",
    ]
    test_errors = []
    for x, y_hat in zip(X_test, Y_test):
        y = net.predict(x)
        e = 0.5 * np.sum((y_hat - y) ** 2)
        test_errors.append(e)
        lines.append(f"  input={np.round(x, 4)}  teacher={np.round(y_hat, 4)}  output={np.round(y, 4)}  error={e:.6f}")
    lines.append(f"mean test error: {np.mean(test_errors):.6f}")

    with open(outdir / "results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
