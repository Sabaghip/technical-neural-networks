# Technical Neural Networks — Coursework

> MA-INF 4204, University of Bonn, WS26/27
> Autonomous Intelligent Systems group, Institute for Computer Science VI (Dr. Nils Goerke)

This repository collects my worked solutions to the weekly assignments of the
**Technical Neural Networks** graduate course at the University of Bonn. The course covers
artificial neural networks from their biological motivation through their historical
development to the major technical paradigms — Perceptrons, threshold logic, multi-layer
networks and backpropagation, self-organizing maps, Hopfield networks, and related models —
combining pencil-and-paper derivations with programming exercises.

Each assignment sheet is solved from scratch (derivations, proofs, and diagrams worked out
by hand) and typeset in LaTeX. Where the course explicitly calls for it, one exercise per
sheet is solved with AI assistance, and that is disclosed directly in the corresponding
solution document.

## Repository structure

```
assignments/
  1/
    solution.tex   # LaTeX source
    solution.pdf   # compiled solution
  2/
    solution.tex
    solution.pdf
    pa_a/            # PA-A: MLP + Backpropagation of Error, implemented in Python/numpy
      mlp.py           # the MLP + Backpropagation of Error implementation
      main.py          # example run
      run_dataset.py   # trains/evaluates mlp.py on one officially provided dataset
      data/01.../06/   # the 6 official course datasets, each with its own results
  3/
    solution.tex
    solution.pdf
    error_surface.py   # generates the error-surface plots referenced in solution.tex
  4/
    solution.tex
    solution.pdf
    pa_b/              # PA-B: encoder-decoder (autoencoder) tasks, Python/numpy
      mlp.py             # same MLP implementation, reused unmodified from PA-A
      run_encoder.py     # trains/evaluates an N-H-N encoder-decoder task
      8-3-8/ 8-2-8/ 64-6-64/   # one folder per task, with its own results + report
README.md
.gitignore
```

Lecture slides and the official assignment-sheet PDFs are course material owned by the
instructor and are intentionally **not** version-controlled here — only my own solutions
are tracked. The small `.txt` datasets provided for the programming assignments (e.g.
PA-A's `data/`) are included, since they're the input data needed to reproduce the results.

## Assignments

| # | Topics covered | Status |
|---|---|---|
| [01](assignments/1) | Biological principles behind technical neurons · history of neural network research · Perceptron weight derivation & linear separability · the BIAS / BIAS-weight formulation · McCulloch–Pitts threshold neurons for Boolean logic (AND, NAND, NOR) | ✅ done |
| [02](assignments/2) | The δ-rule · counting weights (incl. BIAS) across multi-layer MLP topologies · formal analytical proof that XOR is not linearly separable · deriving the Backpropagation learning rule for a custom error function · derivatives of the logistic and tanh activation functions · **PA-A:** a from-scratch Python/numpy MLP trained with Backpropagation of Error, run on all 6 officially provided datasets (XOR variants, a 4-in/2-out logic function, a 2D regression task, and two more Boolean-style functions) | ✅ done |
| [03](assignments/3) | Deriving Backpropagation's δ for both output and hidden layers from a custom (4th-power) error function · a fully worked numeric Backpropagation weight-update calculation · plotting the global error surface of a tanh Perceptron attempting XOR (with a ReLU/0-1 variant) · single-step vs. cumulative (batch) learning compared | ✅ done |
| [04](assignments/4) | Typical single-step learning curve · the effect of linear vs. logarithmic axis scaling on reading a learning curve · why MLPs can solve the 8-3-8 and 8-2-8 encoder-decoder tasks but not 8-1-8 · deriving the weight-decay (L2-regularized) learning rule · deriving the input-space gradient of the error for gradient-based input optimization · **PA-B:** trained the same from-scratch MLP on 8-3-8, 8-2-8 and 64-6-64 encoder-decoder tasks, empirically confirming the binary-code / convex-polygon arguments from assignment 18 | ✅ done |

*(This table — and the skills list below — is updated as each new assignment is completed.)*

## Skills & topics demonstrated

- Deriving Perceptron decision boundaries / separating hyperplanes from labeled data by hand
- Realizing Boolean logic (AND, NAND, NOR, ...) with single threshold neurons
- The BIAS/BIAS-weight trick for folding a neuron's firing threshold into its weight vector
- Historical and conceptual grounding in neural network research (McCulloch–Pitts,
  Rosenblatt, Minsky & Papert, Hopfield, the PDP group / backpropagation, modern deep
  learning)
- Technical writing and diagramming in LaTeX (TikZ, pgfplots)
- Deriving the Backpropagation of Error learning rule from an arbitrary error function via
  the chain rule (standard squared error and beyond)
- Formal, non-geometric proofs of Perceptron capability limits (e.g. XOR / linear
  separability)
- Calculus of activation functions: derivatives of the logistic and tanh functions, and the
  relationship between them
- Multi-layer network architecture design: counting weights (including BIAS) across
  multiple hidden layers
- Implementing a Multi-Layer Perceptron and Backpropagation of Error from scratch in
  Python/numpy (no neural-network libraries), including reproducible weight
  initialization, per-layer learning rates and transfer functions, and a learning-curve
  evaluation on held-out test data
- Fitting the same from-scratch implementation to several real datasets with different
  shapes: multi-output Boolean functions, and a continuous 2D regression task evaluated
  with a proper train/test split
- Visualizing and interpreting a neural network's error/objective surface as a function of
  its weights (bounded vs. unbounded transfer functions, single vs. multiple minima)
- Comparing online/stochastic vs. batch gradient-descent variants of Backpropagation of
  Error
- Reading learning curves under different axis scalings (linear/log), and why log-scaled
  axes are standard practice for visualizing training error
- Understanding encoder-decoder (autoencoder) bottleneck capacity: how binary vs. continuous
  hidden codes and linear separability determine which network sizes can solve a task
- Deriving weight decay (L2 regularization) as a Backpropagation learning-rule extension,
  and the input-space gradient of the error for gradient-based input optimization
- Training autoencoder-style MLPs end to end and diagnosing real optimization behavior
  (slow convergence on tight packing problems, saturation/dead-unit traps under plain
  online SGD, the effect of random seed and training length)

## Tools

Solutions are typeset with `pdflatex` using TikZ and pgfplots for diagrams and plots. Each
assignment folder is self-contained — build the PDF with:

```bash
pdflatex solution.tex
```

## Author

Mohamadhosein Sabaghi
