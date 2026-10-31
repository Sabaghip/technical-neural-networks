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
  01/
    solution.tex   # LaTeX source
    solution.pdf   # compiled solution
README.md
.gitignore
```

Lecture slides and the official assignment-sheet PDFs are course material owned by the
instructor and are intentionally **not** version-controlled here — only my own solutions
are tracked.

## Assignments

| # | Topics covered | Status |
|---|---|---|
| [01](assignments/1) | Biological principles behind technical neurons · history of neural network research · Perceptron weight derivation & linear separability · the BIAS / BIAS-weight formulation · McCulloch–Pitts threshold neurons for Boolean logic (AND, NAND, NOR) | ✅ done |
| 02 | — | 🔜 pending |

*(This table — and the skills list below — is updated as each new assignment is completed.)*

## Skills & topics demonstrated

- Deriving Perceptron decision boundaries / separating hyperplanes from labeled data by hand
- Realizing Boolean logic (AND, NAND, NOR, ...) with single threshold neurons
- The BIAS/BIAS-weight trick for folding a neuron's firing threshold into its weight vector
- Historical and conceptual grounding in neural network research (McCulloch–Pitts,
  Rosenblatt, Minsky & Papert, Hopfield, the PDP group / backpropagation, modern deep
  learning)
- Technical writing and diagramming in LaTeX (TikZ, pgfplots)

## Tools

Solutions are typeset with `pdflatex` using TikZ and pgfplots for diagrams and plots. Each
assignment folder is self-contained — build the PDF with:

```bash
pdflatex solution.tex
```

## Author

Mohamadhosein Sabaghi
