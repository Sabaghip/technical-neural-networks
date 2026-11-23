# PA-C: Radial-Basis-Function Network

Own from-scratch implementation (`rbf.py`) of a 3-layer `N`-`K`-`M` RBF network: Gaussian RBF
neurons in the hidden layer, a linear output layer. No RBF/neural-network library is used, except
`numpy.linalg.pinv` for the optional EXTRA part, as explicitly permitted by the assignment.

**Centers/widths `C_k`, `s_k`:** set via the **Input Data Driven** philosophy (assignment 28) --
own from-scratch k-means clustering on the training inputs gives the `K` centers; each width `s_k`
is the distance to that center's nearest neighboring center (a standard heuristic that keeps
neighboring RBF bumps overlapping).

**Output weights `w_km`:** initialized uniformly in `[-0.5, 0.5]` with an explicit seed, then
trained with **online (single-step) gradient descent**, one learning rate `eta` shared by all
output neurons, exactly the lecture's own rule (sheet 6, "Case P>K: Learn weights"):
```
Delta w_km = eta * (yhat_m - y_m) * r_k
```

**Data:** reused from PA-A (dataset 04: a continuous 2D regression task, 350 patterns, 80/20
train/test split) -- this lets the RBF-Net's performance be compared directly against the MLP that
was trained on the exact same data in PA-A.

## Main run (`main.py`)

Network: `2-20-1` (`K=20` RBF neurons), `eta=0.05`, 60,000 online gradient-descent steps, seed 0.

- mean test error: **0.000457** (max 0.003692) -- comparable to the MLP's 0.00055 on the same data.
- `learning_curve.txt` / `.png` -- training error over steps.
- `results.txt` -- teacher vs. network output for every test pattern.
- `prediction_vs_teacher.png` -- the test set colored by teacher value vs. by network output, with
  the RBF centers marked -- the two color patterns match closely.

## EXTRA: gradient descent vs. Moore-Penrose-Pseudo-Inverse (`pinv_compare.py`)

Same centers/widths (`K=20`), compared at three regimes (`pinv_compare.txt`):

| Case | P | GD test error | GD time | MPPI test error | MPPI time |
|---|---|---|---|---|---|
| P = K | 20 | 0.010398 | 1.21 s | **0.010398** | 0.0004 s |
| P slightly > K | 25 | 0.009392 | 1.22 s | 0.009689 | 0.0003 s |
| P >> K | 280 | 0.000457 | 1.22 s | **0.000442** | 0.0021 s |

**Discussion.** At `P=K` (exact interpolation), gradient descent and the MPPI converge to the
*exact same* weights (identical test error) -- as expected, since with `P=K` the linear system
has a unique exact solution, and gradient descent given enough steps finds that same unique
solution. At `P>>K`, the MPPI's one-shot analytical least-squares solution is even *slightly more
accurate* than 60,000 steps of gradient descent, and over ~570x faster (2ms vs. 1.2s) -- exactly
the trade-off expected from the theory in assignment 29: MPPI is mathematically exact and has
constant, predictable cost, while gradient descent only approximates the same optimum and needs
many iterations to get close. The practical downside of MPPI not visible here (per assignment 29's
own slide) is that for larger `K` the matrix inversion can become numerically unstable or
expensive -- not an issue at this scale (`K=20`).
