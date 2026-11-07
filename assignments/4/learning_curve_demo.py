"""
Assignment 16: draws a typical learning curve for an MLP trained with
Backpropagation of Error, single-step (online) learning.

The curve is synthetic (an exponential decay with multiplicative noise), but
its shape matches what we actually observed empirically for the real,
single-step-trained networks in PA-A (see assignments/2/pa_a/data/*/learning_curve.png):
fast initial drop, then a noisy decreasing trend that settles near a low
floor without ever fully smoothing out.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

steps = np.arange(1, 4001)
trend = 0.9 * np.exp(-steps / 500.0) + 0.01       # smooth decreasing trend + noise floor
noise = rng.uniform(0.6, 1.6, size=steps.shape)   # per-pattern error varies a lot
single_step_error = trend * noise

window = 50
moving_avg = np.convolve(single_step_error, np.ones(window) / window, mode="valid")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(steps, single_step_error, color="tab:blue", linewidth=0.6, alpha=0.6,
        label=r"single-step error ${}^pE$ (one pattern per step)")
ax.plot(steps[window - 1:], moving_avg, color="tab:red", linewidth=2.0,
        label=f"moving average (window={window})")
ax.set_xlabel("training step (one randomly picked pattern per step)")
ax.set_ylabel(r"quadratic error ${}^pE = \frac{1}{2}\sum_m(\hat y_m - y_m)^2$")
ax.set_title("Typical learning curve: Backpropagation of Error, single-step learning")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("learning_curve_typical.png", dpi=130)
print("saved learning_curve_typical.png")
