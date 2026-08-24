# Stage 1 -- Deep Backbone Training

Trains the CNN-Transformer-encoder backbone (the paper's main architecture) and the TCN alternative
architecture, then freezes each and caches its representation across all seven monthly sessions.

## Scripts, in order

1. **`trainBackboneTransformer.m`** -- trains the CNN-Inception + Transformer-encoder backbone on
   Month-0 data only, with early stopping on the Month-0 validation split. Parametrized by capacity:
   call it with `'grande'` (large, 6 Inception blocks, d_model=128), `'mediano'` (medium, 4 blocks,
   d_model=64), or `'pequeno'` (small, 2 blocks, d_model=32) -- the paper reports all three. Run it once
   per capacity you want.

2. **`trainBackboneTCN.m`** -- trains the alternative temporal-convolutional-network backbone, used as
   a second architecturally-distinct sanity check in earlier stages of this project (not part of the
   ICASSP paper's headline comparison, which uses SVM/k-NN for that role instead).

3. **`buildFeatureCache.m`** -- takes a backbone trained by step 1 (auto-detects the most recent
   matching `Models/Backbone_*.mat` for a given capacity), freezes it at its final dropout layer, and
   passes every frame of every session (Month 0 through Month 6) through it once to cache the frozen
   representation. This cache is what every decision head in `04_decision_heads/` and every
   personalized-model script in `06_personalized_models/` consumes -- the backbone itself is never
   retrained again after this step.

4. **`test_smoke_backbone_sizes.m`** -- a fast structural smoke test (no training, no data) that builds
   the `layerGraph` for all three capacities and fails loudly if any topology is invalid. Useful to run
   before committing to a multi-hour training run.

## Reproducibility note

`trainBackboneTransformer.m` uses a fixed random seed and early stopping keyed to Month-0 validation
accuracy (`ValidationPatience`); this is standard practice, but see the paper's Discussion for why it
mechanically makes the Month-0(validation) accuracy a slightly favorable point of comparison against
later months.
