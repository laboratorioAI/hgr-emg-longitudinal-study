# Hyperparameters

Every value below is quoted directly from the scripts in this repository (file and line noted), not
from the paper's compressed prose. The paper targets the OJSP track of ICASSP 2027 (8 pages of
technical content + 1 references-only page), which leaves more room than the standard 4+1-page track,
but this table is still kept as a single authoritative reference -- if the paper text and this table
ever disagree, this table (and the underlying `.m` file) is authoritative.

## Signal preprocessing and windowing (`06_shared_utilities/Shared.m`)

| Parameter | Value |
|---|---|
| Sampling rate | 200 Hz |
| Frame window | 300 samples (1.5 s) |
| Frame stride | 30 samples |
| Frame-label tolerance rule | ≥75% of frame inside gesture interval, OR ≥50% of the interval's total duration covered |
| STFT window / overlap | 24 samples / 50% |
| STFT frequency bins kept | 13 |
| Class-imbalance correction | `w_c = N / (K * n_c)` (scikit-learn `balanced` equivalent), computed once from Month-0 training |

## CNN-Transformer-encoder backbone (`02_backbone_training/trainBackboneTransformer.m`)

| Parameter | Value | Source line |
|---|---|---|
| Random seed | `rng(9)` | L41 |
| Mini-batch size | 64 | L130 |
| Max epochs | 10 | L131 |
| Initial learning rate | 5e-4 | L134 |
| Learning-rate schedule | piecewise, ×0.2 every 5 epochs | L136-137 |
| L2 regularization | 0.005 | L135 |
| Gradient threshold | 1 | L140 |
| Validation patience (early stopping) | 5 | L147 |
| Large capacity | 6 Inception blocks, d_model=128, 8 heads | -- |
| Medium capacity | 4 Inception blocks, d_model=64, 4 heads | -- |
| Small capacity | 2 Inception blocks, d_model=32, 2 heads | -- |
| Feed-forward expansion ratio | ×2 (all capacities) | -- |
| Freeze point | final dropout layer after the Transformer-encoder feed-forward block (`dropout_2`) | -- |

## Softmax decision head (`03_decision_heads/trainSoftmaxHead.m`)

| Parameter | Value | Source line |
|---|---|---|
| Random seed | `rng(9)` | L23 |
| Input normalization | z-score, computed from Month-0 training | -- |
| Initial learning rate | 0.01 | L58 |
| L2 regularization | 0.001 | L59 |
| Max epochs | 200 | L60 |
| Mini-batch size | 128 | L61 |
| Validation patience (early stopping) | 10 | L66 |

## LinUCB contextual bandit (`03_decision_heads/context_bandit.m`)

| Parameter | Value | Source line |
|---|---|---|
| Exploration coefficient α | 0.5 | L53 |
| Max warm-start epochs | 20 | L54 |
| Validation patience (early stopping) | 5 | L55 |
| Ridge regression prior | `A_i` initialized to identity, `b_i` initialized to zero, per arm | -- |
| Update rule | Sherman-Morrison rank-one update, O(d²) | -- |
| Warm-start protocol | Month-0 only, same shuffled-epoch procedure as Softmax; stopping rule is equivalent in kind (validation-based patience) but not identical in epoch budget or patience value (see max warm-start epochs / validation patience above vs. Softmax's 200 / 10) | -- |
| Evaluation after freeze | pure exploitation (`argmax θᵀx`), exploration term dropped | -- |

## Covariate-shift tests (`04_covariate_shift/evaluateCovariateShift.m`)

| Parameter | Value |
|---|---|
| MMD² kernel | Gaussian RBF, median-heuristic bandwidth |
| Permutation-test iterations | 2000 |
| Domain classifier | logistic regression, 5-fold cross-validation |
| Multiple-comparison correction | Holm-Bonferroni across 12 tests (6 months × 2 tests) |
| Per-month random seed | `rng(9 + m)` |
