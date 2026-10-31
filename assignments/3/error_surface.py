"""
Assignment 14: computes and plots the global error surface G(w0, w1, w2) of a
2-1 network (inputs x1, x2, BIAS, one tanh output neuron) trained to realize
XOR, for the BIAS weight fixed at w0 = 2.0 and w1, w2 swept over [-10, 10].

G(w1, w2) = 1/2 * sum_p (yhat_p - y_p)^2,  y_p = f(w0 + w1*x1_p + w2*x2_p)

Also produces the "Extra task" surface: desired values {0, 1} instead of
{-1, +1}, and f = ReLU instead of tanh.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# XOR patterns: (x1, x2) -> teacher
PATTERNS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TEACHER_PM1 = np.array([-1.0, 1.0, 1.0, -1.0])   # main task: desired values -1/+1
TEACHER_01 = np.array([0.0, 1.0, 1.0, 0.0])      # extra task: desired values 0/1

W0 = 2.0
W_RANGE = np.linspace(-10.0, 10.0, 200)


def relu(z):
    return np.maximum(0.0, z)


def error_surface(transfer, teacher):
    """G(w1, w2) for every (w1, w2) on the grid, with w0 fixed."""
    W1, W2 = np.meshgrid(W_RANGE, W_RANGE)
    G = np.zeros_like(W1)
    for (x1, x2), yhat in zip(PATTERNS, teacher):
        net = W0 + W1 * x1 + W2 * x2
        y = transfer(net)
        G += 0.5 * (yhat - y) ** 2
    return W1, W2, G


def plot_surface(W1, W2, G, title, path):
    fig = plt.figure(figsize=(11, 4.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(W1, W2, G, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.set_zlabel("G(w0=2, w1, w2)")
    ax1.set_title(title)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(W1, W2, G, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax2, label="G")
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.set_title("contour")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# main task: tanh, desired values -1/+1
W1, W2, G = error_surface(np.tanh, TEACHER_PM1)
plot_surface(W1, W2, G, "G(w0=2, w1, w2) -- tanh, XOR with yhat in {-1,+1}", "error_surface_main.png")
print("main task:  G min =", G.min(), "at w1,w2 =", W1.flat[G.argmin()], W2.flat[G.argmin()])

# extra task (no extra points): ReLU, desired values 0/1
W1e, W2e, Ge = error_surface(relu, TEACHER_01)
plot_surface(W1e, W2e, Ge, "Extra: G(w0=2, w1, w2) -- ReLU, XOR with yhat in {0,1}", "error_surface_extra.png")
print("extra task: G min =", Ge.min(), "at w1,w2 =", W1e.flat[Ge.argmin()], W2e.flat[Ge.argmin()])
