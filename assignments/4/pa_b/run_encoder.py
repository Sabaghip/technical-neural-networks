"""
PA-B: trains the MLP (mlp.MLP, reused unmodified from PA-A) on an N-H-N
encoder-decoder (autoencoder) task: N one-hot input patterns, teacher =
input, compressed through a bottleneck hidden layer of H neurons.

Both hidden and output layer use the logistic transfer function, since
inputs/teachers are 0.0/1.0 -- logistic's range (0,1) matches directly.

Writes into the given output directory:
  - learning_curve.txt / .png   -- error over training
  - results.txt                  -- final output for every pattern
  - io_heatmap.png               -- input matrix vs. reconstructed output matrix
  - hidden_states.txt            -- the H hidden activations for every pattern
  - hidden_states.png            -- 2D scatter (H=2), 3D scatter (H=3), or heatmap (H>3)

Usage:
    python3 run_encoder.py --n 8 --h 3 --outdir 8-3-8 --lr 0.5 0.5 --steps 20000 --seed 0
"""

import argparse
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from mlp import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of input/output neurons (and patterns)")
    ap.add_argument("--h", type=int, required=True, help="number of hidden (bottleneck) neurons")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lr", type=float, nargs=2, default=[0.5, 0.5])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N, H = args.n, args.h
    X = np.eye(N)   # N one-hot input patterns
    Y = X.copy()    # autoencoder: teacher = input

    net = MLP([N, H, N], ["logistic", "logistic"], args.lr, args.seed)

    learning_curve = []
    for step in range(args.steps):
        p = net.rng.integers(0, N)
        error = net.train_step(X[p], Y[p])
        learning_curve.append(error)

    with open(outdir / "learning_curve.txt", "w") as f:
        f.write("# training step    quadratic error E\n")
        for step, error in enumerate(learning_curve):
            f.write(f"{step} {error:.6f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(learning_curve, linewidth=0.5)
    plt.xlabel("training step")
    plt.ylabel("quadratic error E")
    plt.title(f"Learning curve: {N}-{H}-{N} encoder-decoder")
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve.png", dpi=130)
    plt.close()

    # final forward pass on every pattern (no further weight changes)
    hidden_states = np.zeros((N, H))
    outputs = np.zeros((N, N))
    lines = [f"{N}-{H}-{N} encoder-decoder after {args.steps} training steps", ""]
    max_abs_err = 0.0
    for p in range(N):
        outs = net.forward(X[p])
        hidden_states[p] = outs[1]
        outputs[p] = outs[-1]
        err = np.max(np.abs(Y[p] - outputs[p]))
        max_abs_err = max(max_abs_err, err)
        if N <= 16:
            lines.append(f"pattern {p}: output = {np.round(outputs[p], 3).tolist()}")
    if N > 16:
        lines.append(f"(N={N} patterns -- see io_heatmap.png / results.txt for all outputs)")
    lines.append("")
    lines.append(f"max abs error over all patterns/outputs: {max_abs_err:.4f}")
    with open(outdir / "results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    np.savetxt(
        outdir / "hidden_states.txt",
        hidden_states,
        header=f"hidden-neuron activations ({H} columns) for each of the {N} patterns (rows)",
    )

    # depict input vs. reconstructed output
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(X, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("input")
    axes[0].set_xlabel("input neuron")
    axes[0].set_ylabel("pattern")
    im = axes[1].imshow(outputs, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("reconstructed output")
    axes[1].set_xlabel("output neuron")
    fig.colorbar(im, ax=axes, shrink=0.8, label="activation")
    fig.suptitle(f"{N}-{H}-{N}: input vs. output")
    fig.savefig(outdir / "io_heatmap.png", dpi=130)
    plt.close(fig)

    # visualize the hidden-layer states
    if H == 2:
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(hidden_states[:, 0], hidden_states[:, 1])
        for p in range(N):
            plt.annotate(str(p), hidden_states[p], fontsize=8, xytext=(3, 3), textcoords="offset points")
        plt.xlabel("hidden neuron 1 output")
        plt.ylabel("hidden neuron 2 output")
        plt.title(f"Hidden-layer codes ({N}-{H}-{N})")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(outdir / "hidden_states.png", dpi=130)
        plt.close()
    elif H == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(hidden_states[:, 0], hidden_states[:, 1], hidden_states[:, 2])
        for p in range(N):
            ax.text(*hidden_states[p], str(p), fontsize=8)
        ax.set_xlabel("hidden 1")
        ax.set_ylabel("hidden 2")
        ax.set_zlabel("hidden 3")
        ax.set_title(f"Hidden-layer codes ({N}-{H}-{N})")
        fig.tight_layout()
        fig.savefig(outdir / "hidden_states.png", dpi=130)
        plt.close(fig)
    else:
        plt.figure(figsize=(6, max(3, N * 0.15)))
        plt.imshow(hidden_states, aspect="auto", cmap="viridis")
        plt.colorbar(label="activation")
        plt.xlabel("hidden neuron")
        plt.ylabel("pattern")
        plt.title(f"Hidden-layer codes ({N}-{H}-{N})")
        plt.tight_layout()
        plt.savefig(outdir / "hidden_states.png", dpi=130)
        plt.close()

    print(f"\nsaved results to {outdir}/")


if __name__ == "__main__":
    main()
