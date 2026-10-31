"""
Multi-Layer-Perceptron with Backpropagation of Error, implemented from scratch.

Only numpy is used, for array/matrix arithmetic

Implements the 7-step Backpropagation of Error algorithm from the lecture:
  1. Initialization      -> MLP.__init__
  2. Pick a pattern p    -> done by the caller (see main.py)
  3. Forward              -> MLP.forward
  4. Compare               -> error in MLP.train_step
  5. Backward              -> delta computation in MLP.train_step
  6. Update                -> weight update in MLP.train_step
  7. Ready?                 -> stopping criterion, handled by the caller
"""

import numpy as np

TRANSFER_FUNCTIONS = {
    "tanh": (np.tanh, lambda x: 1.0 - x ** 2),
    "logistic": (
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda x: x * (1.0 - x),
    ),
    "identity": (lambda x: x, lambda x: np.ones_like(x)),
}


class MLP:
    """Feed-forward multi-layer perceptron, trained with online Backpropagation of Error."""

    def __init__(self, layer_sizes, transfer_names, learning_rates, seed):
        """
        layer_sizes:    e.g. [2, 4, 1] -> N=2 inputs, one hidden layer with 4
                        neurons, M=1 output neuron. len(layer_sizes) <= 4.
        transfer_names: transfer function name per layer after the input
                        layer, e.g. ["tanh", "logistic"]. Each one of
                        "tanh" / "logistic" / "identity".
        learning_rates: learning rate eta per weight layer
                        (len == len(layer_sizes) - 1).
        seed:           explicit random seed.
        """
        assert len(layer_sizes) <= 4, "maximum of 4 layers allowed"
        assert max(layer_sizes) <= 1000, "maximum of 1000 neurons per layer allowed"
        assert len(transfer_names) == len(layer_sizes) - 1
        assert len(learning_rates) == len(layer_sizes) - 1

        self.layer_sizes = layer_sizes
        self.transfer_names = transfer_names
        self.learning_rates = learning_rates
        self.rng = np.random.default_rng(seed)

        self.weights = [
            self.rng.uniform(-2.0, 2.0, size=(layer_sizes[l + 1], layer_sizes[l] + 1))
            for l in range(len(layer_sizes) - 1)
        ]

    def forward(self, x):
        """Propagate one input pattern x forward (step 3)."""
        out = np.asarray(x, dtype=float)
        outs = [out]
        for l, W in enumerate(self.weights):
            out_biased = np.concatenate(([1.0], out)) 
            net = W @ out_biased
            f, _ = TRANSFER_FUNCTIONS[self.transfer_names[l]]
            out = f(net)
            outs.append(out)
        return outs

    def train_step(self, x, y_hat):
        """One online Backpropagation of Error step on a single pattern (x, y_hat)."""
        outs = self.forward(x)                                 # step 3: forward
        y = outs[-1]
        error = 0.5 * np.sum((y_hat - y) ** 2)                 # step 4: compare

        # step 5: backward
        deltas = [None] * len(self.weights)
        _, f_prime_out = TRANSFER_FUNCTIONS[self.transfer_names[-1]]
        deltas[-1] = (y_hat - y) * f_prime_out(outs[-1])

        for l in range(len(self.weights) - 2, -1, -1):
            _, f_prime = TRANSFER_FUNCTIONS[self.transfer_names[l]]
            W_next = self.weights[l + 1][:, 1:] 
            deltas[l] = f_prime(outs[l + 1]) * (deltas[l + 1] @ W_next)

        # step 6: update
        for l, W in enumerate(self.weights):
            out_biased = np.concatenate(([1.0], outs[l]))
            W += self.learning_rates[l] * np.outer(deltas[l], out_biased)

        return error

    def predict(self, x):
        """Forward pass only."""
        return self.forward(x)[-1]
