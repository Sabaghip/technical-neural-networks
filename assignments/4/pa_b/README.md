# PA-B: Encoder-Decoder (Autoencoder) Tasks

Implements and trains an `N`-`H`-`N` MLP (own implementation, `mlp.py` -- reused unmodified from
PA-A) to reconstruct `N` one-hot input patterns through a bottleneck hidden layer of `H` neurons,
trained with online Backpropagation of Error (`run_encoder.py`).

**Transfer functions:** logistic in both the hidden and the output layer, since all inputs and
teacher values are exactly `0.0`/`1.0` and logistic's range `(0,1)` matches directly.

**Learning rate:** `eta = 0.5` for both weight layers in all three tasks.

**Weight init / reproducibility:** uniform(-2, 2), explicit seed (as implemented in `mlp.py`).

| Task | Training steps | Seed | Max abs. error (all patterns) | Notes |
|---|---|---|---|---|
| [8-3-8](8-3-8/) | 20,000 | 0 | 0.104 | Converges easily -- 2^3 = 8 matches the pattern count exactly, hidden codes land near the 8 corners of the unit cube (see `hidden_states.png`). |
| [8-2-8](8-2-8/) | 200,000 | 0 | 0.095 | 2^2 = 4 < 8, so this only works because hidden activations are continuous, not binary. Needed ~10x more steps than 8-3-8: the 8 patterns have to be arranged so each one is linearly separable from the other 7 by the output layer, a slower, fine-grained optimization. Final codes sit at the 8 vertices of a convex octagon -- exactly the argument from assignment 18. |
| [64-6-64](64-6-64/) | 700,000 | 4 | 0.085 | 2^6 = 64 matches exactly, as in 8-3-8, just bigger. The first seed tried (0) got several patterns permanently stuck near-zero output (a saturation/vanishing-gradient trap under online SGD with no momentum) even after 800k steps; a short seed search found seed 4 converges cleanly. Interestingly, one of the 6 hidden neurons saturates to ~1.0 for every pattern (see `hidden_states.png`) -- the network ends up using only 5 effectively-variable continuous dimensions to separate all 64 patterns, the same continuous-coding trick as 8-2-8, just in 5D instead of 2D. |

Each task folder contains:
- `learning_curve.txt` / `.png` -- training error over steps
- `results.txt` -- final output for every pattern (no further weight changes)
- `io_heatmap.png` -- input matrix vs. reconstructed output matrix, side by side
- `hidden_states.txt` / `.png` -- the hidden-neuron activations for every pattern (2D scatter for
  H=2, 3D scatter for H=3, heatmap for H=6)

Run any task with, e.g.:
```bash
python3 run_encoder.py --n 8 --h 3 --outdir 8-3-8 --lr 0.5 0.5 --steps 20000 --seed 0
```
