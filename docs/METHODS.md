# Methods

## Data representation

The supplied subLINCS matrices contain `cell_line, compound_id, SMILES` followed by 978 expression values. GeneVAE uses the expression vector; post-training also uses SMILES and cell labels. Target inputs use the first numeric signature row, with genes ordered according to `data/gene_order.json`. Inputs are used without additional normalization or gene reordering. Dataset formats and provenance are described in the [data guide](../data/README.md).

## GeneVAE pretraining

The encoder has dimensions **978 → 512 → 256 → 128**, with ReLU and dropout 0.2. Separate linear heads produce the mean and log variance of a **64-dimensional** Gaussian latent variable. The decoder mirrors the encoder and ends with a linear 978-dimensional output.

Sampling uses `z = μ + exp(logvar/2) × ε`, where `ε ~ N(0, I)`. The loss is `α × Σ(x_hat − x)² + (1 − α) × β × KL`, with summed KL divergence to the standard normal and β = 1. Over 2,000 epochs, α decreases linearly from 0.99 to 0.5 during the first 1,000 epochs, then remains fixed. Adam uses learning rate `1e-4`, batch size 64, and seed 0. The final epoch's model is saved.

## Tx2Mol post-training

The NovoMolGen 300M backbone uses 32 layers, 12 attention heads, hidden width 768, and intermediate width 3,072. Its atom-wise SMILES tokenizer has an 84-token vocabulary and adds BOS/EOS; maximum sequence length is 100.

GeneVAE is frozen in evaluation mode and supplies a sampled latent vector. Concatenating this vector with a 14-dimensional cell-line one-hot encoding gives 78 inputs to a **78 → 1,536 → 768** projection with GELU, dropout, LayerNorm, and L2 normalization scaled by √768. The resulting prefix conditions the trainable molecular model.

The objective combines causal language-model loss with cell-grouped bidirectional InfoNCE (weight 0.2; temperature 0.07). For this term, the released backend uses the BOS input embedding as its molecular representation.

| Training setting | Value |
| --- | --- |
| Optimizer | AdamW; weight decay 0.01; epsilon `1e-8` |
| Model / projection learning rate | `2e-5` / `5e-5` |
| Batch size / gradient accumulation | 32 / 1 |
| Maximum epochs | 20 |
| Scheduler | 10% warm-up, cosine decay |
| Gradient clipping / precision / seed | 1.0 / BF16 / 42 |
| Early-stopping patience / minimum improvement | 3 / 0.001 |

Checkpoint selection uses `0.35 × validity + 0.25 × uniqueness + 0.25 × diversity + 0.15 × QED`, with validity and uniqueness expressed as fractions. The executable training definitions are in [`_generation_metrics`](../tx2mol/finetune.py).

## Molecular generation and evaluation

Defaults are MCF7 conditioning, 10 runs per target, and 100 samples per run, using a fixed checkpoint. Sampling uses temperature 1.0, top-p 0.95, top-k 100, maximum length 100, and batch size 32. Run `r` uses seed `42 + r` for Python, NumPy, PyTorch, and CUDA. Invalid strings and duplicates remain in the recorded sampling budget.

### Paper maximum Tanimoto evaluation

Use **Morgan fingerprints with radius 2, 2,048 bits, and `useChirality=False`**. Known ligands are canonicalized, deduplicated, and excluded if present among canonical training SMILES. Generated molecules are scored using **all valid molecules**, without a novelty filter.

For target `t` and run `r`, calculate `S(t,r) = max Tanimoto(Morgan(g), Morgan(l))` across valid generated molecules `g` and eligible ligands `l`. Select `max_r S(t,r)` for each target; ties use the lowest run index. The overall mean averages these selected target maxima. Empty molecule or ligand sets receive zero. Generation and CPU reevaluation use the same implementation in [`tx2mol.evaluate`](../tx2mol/evaluate.py).

### Additional molecular metrics

| Metric | Definition |
| --- | --- |
| Validity | Valid RDKit molecules with at least two atoms / all samples |
| Uniqueness | Unique canonical valid SMILES / valid samples |
| Novelty | Unique valid molecules absent from training and validation sets / unique valid molecules |
| Diversity | One minus mean pairwise Morgan Tanimoto over valid samples, including duplicates; fewer than two molecules gives 1.0 |
| QED, SA, logP, Lipinski compliance, molecular weight | Means over unique novel molecules; empty sets receive zero |
| `intdivp` | Legacy, order-sensitive RDKFingerprint suffix-wise mean-distance statistic |

Rates are reported as percentages. Across runs, maximum Tanimoto is aggregated by maximum; other diagnostics use the mean and sample standard deviation, with zero SD for a single run. Complete winning groups and molecule–ligand matches are exported automatically.

## Reproducibility and resources

The packaged environment pins Python 3.10, PyTorch 2.1.2, CUDA 11.8, and FlashAttention 2.6.1. GPU checks used one RTX 4090 (24 GB); exact validation environments and scope are in [Validation](VALIDATION.md). Runs record seeds, settings, software versions, and checkpoint/data hashes. Keep GeneVAE paired with its trained Tx2Mol projection; sampling may vary across hardware and software environments.
