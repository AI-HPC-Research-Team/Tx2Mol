# Implemented protocol

## Data construction and ordering

The supplied subLINCS matrices are headerless CSVs with `cell_line, compound_id, SMILES` followed by 978 expression values. They are stored losslessly as gzip files. No per-gene normalization, transformation, sign reversal or gene reordering is performed inside this release. GeneVAE uses only the expression columns; post-training additionally consumes SMILES and cell labels.

Each target CSV contains a 978-gene KEGG-ID header and numeric signature rows. The original generation protocol uses the first numeric row; this is preserved. All ten headers must match `data/gene_order.json`. The training matrices lack gene-ID headers, so their correspondence to this expected order is supplied provenance, not independently demonstrated by a header comparison. The code fails on incorrect dimensions, nonnumeric/nonfinite expression or target-order mismatch.

## GeneVAE

Encoder: 978 → 512 → 256 → 128, with ReLU and dropout 0.2 after every hidden layer. Separate 128 → 64 linear heads produce μ and log variance. The decoder mirrors the encoder: 64 → 128 → 256 → 512 → 978, with a linear output.

The latent sample is `z = μ + exp(logvar/2) × ε`, with ε drawn from a standard normal. The objective is `α × Σ(x_hat − x)^2 + (1 − α) × β × KL`, with a summed standard-normal KL term, β = 1, and α decreasing linearly from 0.99 to 0.5 over the first 1,000 of 2,000 epochs. α remains 0.5 afterwards. Adam uses learning rate 1e−4 and batch size 64. Fresh release runs explicitly use seed 0 and save the final epoch's model. No validation early stopping is used for GeneVAE.

During post-training and generation the VAE is frozen and in evaluation mode. Its stochastic sample, not its posterior mean, provides the condition. The post-training loader's dropout value 0.1 does not activate dropout in evaluation mode.

## Conditional molecular model

The archived NovoMolGen 300M backbone has hidden width 768, 32 layers, 12 attention heads, intermediate width 3,072 and an 84-token atom-wise SMILES vocabulary. BOS/EOS are added by the tokenizer. Maximum sequence length in this task is 100.

Concatenate 64 latent coordinates with a 14-dimensional cell-line one-hot vector. A 78 → 1,536 → 768 MLP, GELU/dropout/LayerNorm and L2 normalization scaled by √768 produce one prefix token. The generator and projection are trainable.

Post-training uses AdamW, generator learning rate 2e−5, projection rate 5e−5, weight decay 0.01, epsilon 1e−8, batch 32, accumulation 1, up to 20 epochs, a 10% warm-up and cosine decay. Gradient norm clipping is 1.0, matching the value actually executed by the historical script, rather than its unused saved argument 0.6. GeneVAE remains frozen. The default uses BF16 and seed 42.

The full training checkpoint criterion preserves the historical weighted combination of molecular validity, uniqueness, diversity and QED (see `_generation_metrics` for the executable definition). Checkpoints and resolved settings are saved on validation improvement; patience is 3 with minimum improvement 0.001. The smoke-only `--skip_generation_eval` mode selects by validation LM loss and explicitly marks outputs as smoke tests.

### Historical InfoNCE limitation

The added loss has weight 0.2 and temperature 0.07, is bidirectional, and groups negatives by cell line. The historical script attempts to use the first SMILES-position hidden state and the GE prefix representation.

However, the pinned custom backbone returns its hidden states as a single `[batch, length, hidden]` tensor. The original script indexes this value as if it were a tuple (`hidden_states[-1]`), producing a two-dimensional tensor. Its fallback then uses the first input-token embedding. Because the tokenizer adds BOS, the molecular side in this archived implementation is the same context-free BOS embedding for every sample. It therefore does **not** provide a molecule-specific contrastive representation within a cell group. Prefix-conditioned language-model training still runs.

This release preserves that computation for historical reproducibility, emits a warning and records the observed representation. Correcting the objective would require a separately labelled implementation and retraining; it would not reproduce the archived checkpoint. Do not describe the archived loss as pooled whole-molecule alignment.

## Generation and molecular metrics

Default generation uses MCF7, 10 runs per target, 100 attempted strings per run, temperature 1.0, top-p 0.95, top-k 100 and maximum length 100. Run `r` uses seed `42 + r` for Python, NumPy, PyTorch and CUDA; all attempted strings are retained, without retries to replace invalid strings. These are repeated sampling runs with a fixed checkpoint, not training epochs.

### Paper maximum Tanimoto evaluation

`tx2mol.generate` and `tx2mol.evaluate` share the same scoring and selection implementation. Generation automatically evaluates **all valid generated molecules**, exports every run maximum, and retains the complete winning groups. Known ligands are canonicalized, deduplicated, and excluded if their canonical SMILES occur in the training set (metadata column 2, zero-based). Validation SMILES are not used to exclude reference ligands. Generated molecules are not filtered by novelty. The unchanged historical evaluator is retained as an independent reference, not a runtime dependency.

For target `t` and run `r`, compute `S(t,r) = max Tanimoto(Morgan(g), Morgan(l))` over valid generated molecules `g` and eligible known ligands `l`. Use radius 2, 2,048 bits, and the historical RDKit default `useChirality=False`. A value of 1 does not establish stereochemical identity. Empty molecule/reference sets retain the historical score of zero.

Select `max_r S(t,r)` across ten runs per target and retain the entire winning group. Ties use the lowest run index. The final mean is across the ten selected target maxima. Reevaluating all 100 archived runs gives the historical mean **0.9136607142857143**; see `examples/paper_protocol/` for the full inputs and expected scores.

### Generator diagnostics

The generator also retains the following diagnostics in `run_metrics.csv` and `aggregate_metrics.csv`. Every primary `max_tanimoto` field uses the all-valid definition above.

- Validity: valid RDKit molecules with at least two atoms / all attempts.
- Uniqueness: unique canonical valid SMILES / valid attempts.
- Novelty: unique valid molecules absent from the union of training and validation SMILES / unique valid molecules.
- Diversity: one minus mean pairwise Tanimoto similarity over valid attempts, including duplicates, using radius-2, 2,048-bit Morgan fingerprints. The historical fewer-than-two-valid convention returns 1.0; inspect the molecule count before interpreting it.
- Source-ligand similarity: each valid attempt receives its maximum eligible-ligand similarity; each run receives the maximum over its attempts. Mean pairwise and mean-maximum Tanimoto fields are not produced by the current generator.
- `intdivp` preserves the legacy RDKFingerprint suffix-wise mean-distance statistic. It is order-sensitive and is not the same as Morgan diversity. The release uses deterministic first-occurrence order instead of set iteration order.
- QED, SA, logP, Lipinski compliance and molecular weight are averaged over unique novel molecules. Legacy empty-set zero conventions are retained; counts are supplied so zeros are not mistaken for observed molecular properties.

Percentages are reported on a 0–100 scale. `aggregate_metrics.csv` reports the highest `max_tanimoto` and its run index for each target; other diagnostics use the arithmetic mean and sample standard deviation (ddof=1), with zero SD for a single run. `best_max_tanimoto.csv` records the selected scores and witness pairs; `best_run_attempts.csv` retains the complete winning groups. All files are created automatically by generation.

## Seed and checkpoint provenance

Seed 0 is an explicit choice for fresh GeneVAE reproduction. The historical entry point only applied it when `--use_seed` was passed; the old log does not establish that flag. Historical generation also did not explicitly set a seed. The release consequently supports repeatable new runs and archived-weight inference, not bitwise claims about old unseeded output files.

GPU kernels, device models and software versions can still affect results across environments. Each run records its environment, resolved inputs and hashes. Newly trained and archived GeneVAE weights must not be interchanged with a fixed conditional projection.
