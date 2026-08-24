# Stage 3 -- Personalized Per-Participant Models

Every model in Stages 1--2 is a *population* model: one classifier trained on pooled Month-0 data from
all 19 participants. A stable population-level accuracy curve is consistent both with every participant
drifting a little, and with a few participants collapsing while the rest stay stable -- the population
view cannot tell these apart. This stage repeats the evaluation with one independent model per
participant instead, to check which of those two stories is actually true.

## Scripts

1. **`evaluatePerUserModels.m`** -- the first version of this idea: one Softmax head per participant,
   trained on the frozen large backbone's representation, using only that participant's own Month-0
   data.

2. **`trainPersonalizedModels.m`** -- the fuller version reported in the paper: 19 independent models
   per configuration for the two most stable classical configurations found in Stage 1's robustness
   sweep (linear-kernel SVM with `C=10`; k-NN with `k=9`, cosine distance), trained and evaluated using
   only each participant's own data.

3. **`trainPersonalizedSoftmax.m`** -- the same per-participant design applied to a Softmax head on the
   frozen small backbone's *learned* representation, so the paper can compare between-participant
   dispersion for hand-crafted features (SVM/k-NN) against a learned representation under the identical
   personalized-model protocol.

## Reproducibility note

All 19 participants have complete Month 0--6 data, so none are excluded from this stage. Each
personalized model is trained from that one participant's own Month-0 80/20 split, class weights, and
z-score normalization statistics -- computed independently per participant, not reused from the
population-level cache in Stages 1--2.
