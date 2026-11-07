"""
Assignment 17: the "unrealistic learning curve" f(z) = 2**(2 - z + 4*sin(z)),
plotted with the 4 required axis scalings (A: lin-lin, B: lin-x/log-y,
C: log-x/lin-y, D: log-log).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.linspace(0.05, 12, 3000)
f = 2.0 ** (2 - z + 4 * np.sin(z))

print("f(z) range:", f.min(), "...", f.max())

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(z, f)
axes[0, 0].set_title("A: linear x, linear y")
axes[0, 0].set_xlabel("z"); axes[0, 0].set_ylabel("f(z)")

axes[0, 1].plot(z, f)
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("B: linear x, logarithmic y")
axes[0, 1].set_xlabel("z"); axes[0, 1].set_ylabel("f(z)")

axes[1, 0].plot(z, f)
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("C: logarithmic x, linear y")
axes[1, 0].set_xlabel("z"); axes[1, 0].set_ylabel("f(z)")

axes[1, 1].plot(z, f)
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("D: logarithmic x, logarithmic y")
axes[1, 1].set_xlabel("z"); axes[1, 1].set_ylabel("f(z)")

for ax in axes.flat:
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("log_scale_demo.png", dpi=130)
print("saved log_scale_demo.png")
