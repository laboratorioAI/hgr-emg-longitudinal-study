# Stage 2 -- Decision Heads: Softmax vs. LinUCB

Trains and compares the two decision mechanisms the paper reports, on top of the frozen backbone
representation from Stage 1. This is the paper's central comparison.

## Scripts

1. **`trainSoftmaxHead.m`** -- the supervised-learning paradigm. A single fully-connected + softmax
   layer on the frozen representation, trained once on Month 0 (z-score-normalized input, weighted
   cross-entropy, early-stopped on Month-0 validation), then frozen and evaluated on Month 1--6.
   Parametrized by backbone capacity, same as `02_backbone_training/trainBackboneTransformer.m`.

2. **`context_bandit.m`** -- the reinforcement-learning paradigm (LinUCB). Warm-started once on Month 0
   under the same kind of validation-based stopping rule as Softmax (see the paper's Methodology for the
   exact epoch/patience budgets, which differ between the two), then frozen: Month 1--6 evaluation uses
   pure exploitation (the upper-confidence exploration term is dropped once learning has stopped). This
   is the LinUCB condition reported in the paper.

3. **`consolidateSizeAblation.m`** -- gathers the per-capacity Softmax/LinUCB results (large/medium/
   small) into the single consolidated table/figure format used in the paper (Table 1, Fig. 2).

The remaining scripts in this folder are exploratory extensions from earlier stages of this project and
are **not** part of the paper's reported comparison:

4. **`context_bandit_adaptive.m`** -- starts from the same frozen LinUCB state and keeps updating
   online, month over month, using only the reward signal, instead of staying frozen after warm-start.

5. **`multiSeedBanditVerification.m`** -- re-runs the frozen-vs-adaptive LinUCB comparison across
   multiple random seeds, to check that comparison is not an artifact of one seed's warm-start.

6. **`evaluateMonthByMonthMatrix.m`** -- generalizes the "train once on Month 0, evaluate on every later
   month" protocol into a full month x month accuracy matrix (train on any month, evaluate on any
   other), for the Softmax condition only.

7. **`compareParadigms.m`** -- loads the most recent results from scripts 1, 2, and 4, and produces
   comparison plots across all conditions (frozen and adaptive), a superset of what the paper's Fig. 2
   shows.

## Reproducibility note

Every script here auto-detects its required input (backbone capacity, frozen-feature cache) by
filename timestamp in `Models/`, matching the pattern documented in the top-level README -- run stages
in order and you never need to hand off variables between scripts.
