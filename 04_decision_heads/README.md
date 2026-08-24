# Stage 2 -- Decision Heads: Softmax vs. LinUCB

Trains and compares the two decision mechanisms the paper is centrally about, on top of the frozen
backbone representation from Stage 1.

## Scripts

1. **`trainSoftmaxHead.m`** -- the supervised-learning paradigm. A single fully-connected + softmax
   layer on the frozen representation, trained once on Month 0 (z-score-normalized input, weighted
   cross-entropy, early-stopped on Month-0 validation), then frozen and evaluated on Month 1--6.
   Parametrized by backbone capacity, same as `02_backbone_training/trainBackboneTransformer.m`.

2. **`context_bandit.m`** -- the reinforcement-learning paradigm (LinUCB). Warm-started once on Month 0
   under an equivalent validation-based stopping rule, then frozen: Month 1--6 evaluation uses pure
   exploitation (the upper-confidence exploration term is dropped once learning has stopped).

3. **`context_bandit_adaptive.m`** -- a third condition (not part of the ICASSP paper's headline
   comparison, which deliberately keeps both mechanisms frozen for a matched comparison): starts from
   the same frozen LinUCB state and keeps updating online, month over month, using only the reward
   signal. Kept here because `compareParadigms.m` can plot it alongside the frozen conditions.

4. **`multiSeedBanditVerification.m`** -- re-runs the frozen-vs-adaptive LinUCB comparison across
   multiple random seeds, to check the comparison is not an artifact of one seed's warm-start.

5. **`evaluateMonthByMonthMatrix.m`** -- generalizes the "train once on Month 0, evaluate on every later
   month" protocol into a full month × month accuracy matrix (train on any month, evaluate on any
   other), for the Softmax condition only.

6. **`consolidateSizeAblation.m`** -- gathers the per-capacity Softmax/LinUCB results (large/medium/
   small) into the single consolidated table/figure format used in the paper.

7. **`compareParadigms.m`** -- loads the most recent results from scripts 1--3 and produces the
   Softmax-vs-LinUCB comparison plots and summary table.

## Reproducibility note

Every script here auto-detects its required input (backbone capacity, frozen-feature cache) by
filename timestamp in `Models/`, matching the pattern documented in the top-level README -- run stages
in order and you never need to hand off variables between scripts.
