# Reproduction tutorial

[Back to the project overview](../README.md).

Phenotype-driven de novo molecular design from gene expression signatures.

This repository contains the executable pipeline, configurations, supplied datasets, regression tests, and a ten-target generation example:

```text
GeneVAE pretraining -> Tx2Mol post-training -> phenotype-guided SMILES generation
```

Choose the workflow that matches your goal:

| Goal | Tutorial |
| --- | --- |
| Generate with the released model | [Quick start](#2-quick-start-use-the-released-checkpoint) |
| Train and generate a new model | [Training](#4-train-the-three-stage-pipeline) |
| Check the full pipeline before a long run | [Smoke test](#5-validate-the-pipeline) |

Source code and data are ordinary repository files. Large weights are [Release assets](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0), downloaded and verified by the commands below. No private server paths or credentials are required.

## 1. Install the environment

### Requirements

- Linux with an NVIDIA Ampere-or-newer GPU and a compatible NVIDIA driver. The integrated pipeline was tested on one RTX 4090 with 24 GB VRAM.
- Conda or Miniconda and Git. Reserve at least 10 GB for downloads, extracted models, and initial outputs; training checkpoints can require additional space.
- Python 3.10, CUDA toolkit 11.8, PyTorch 2.1.2, Transformers 4.46.2, and FlashAttention 2.6.1. The environment files pin the tested versions.

GeneVAE alone supports CPU/V100. The archived Tx2Mol backbone requires CUDA and FlashAttention; macOS and CPU-only machines are not the full training/generation environment. This release supports one GPU/process.

```bash
git clone https://github.com/Yaxin-Xu/Tx2Mol.git
cd Tx2Mol

conda env create -f environment.yml
conda activate tx2mol
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
bash scripts/install.sh
```

The installer installs PyTorch, the pinned dependencies, FlashAttention, and the local package. FlashAttention may compile during installation; `environment.yml` supplies the CUDA toolkit and compiler. Installation ends with an environment check.

Before running a model, check the environment and all supplied data:

```bash
python scripts/check_environment.py
python scripts/validate_data.py
```

The first command reports versions, GPU, and `cuda_available: true`. The second verifies file hashes, dimensions, target gene ordering, and SciPlex3 reference pairing, ending with `Data validated.`

**Run all commands below from the cloned repository root with the `tx2mol` environment active.**

## 2. Quick start: use the released checkpoint

This workflow uses the archived epoch-9 Tx2Mol model and the exact GeneVAE paired with it. Training is not required.

### Download the reference weights

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --reference
```

The approximately 1.13 GB archive and every extracted file are verified against `assets/release_manifest.json`. The installed layout is:

```text
checkpoints/reference/
├── gene_vae.pt
└── tx2mol/
    ├── pytorch_model.bin
    ├── ge_projection.pt
    ├── config.json
    ├── modeling_novomolgen.py
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── args.json
    └── best_metrics.json
```

The small `checkpoints/gene_vae_reference.pt` tracked in Git is the same reference GeneVAE. The installer places it at the path used by `configs/generate_reference.json`, alongside the complete molecular model.

### Generate a small example for all ten targets

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --num_runs 1 --num_samples 10 --batch_size 10 \
  --output_dir outputs/reference_demo
```

This attempts **100 molecules in total**: ten each for AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53. Invalid strings and duplicates are retained in the raw output; the budget does not guarantee 100 distinct valid molecules.

Generation automatically evaluates all attempts and exports the selected results. Inspect the summary:

```bash
python - <<'PY'
import pandas as pd
result = pd.read_csv("outputs/reference_demo/best_max_tanimoto.csv")
columns = ["target", "best_group_1based", "max_tanimoto"]
print(result[columns].to_string(index=False))
PY
```

The archived 100-attempt execution example is in [`examples/reference_demo/`](../examples/reference_demo/). It demonstrates execution and is not a full performance benchmark. Exact strings depend on checkpoint, seed, batch size, device, and software versions.

### Run the full ten-target example

```bash
python -m tx2mol.generate --config configs/generate_reference.json
```

Defaults are **10 targets × 10 runs × 100 attempts = 10,000 attempts**, with MCF7 conditioning, seed `42 + run_idx`, batch size 32, temperature 1.0, top-p 0.95, top-k 100, and maximum length 100. Results go to `outputs/reference_targets/`.

This one command performs generation, all-valid maximum Tanimoto scoring, best-run selection, and complete winning-group export. Everything is saved in `outputs/reference_targets/`. The ten runs sample a fixed checkpoint; they are not ten additional training epochs.

To select one target or change the sampling budget:

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --targets EGFR --num_runs 3 --num_samples 100 \
  --output_dir outputs/reference_egfr
```

Explicit command-line options override JSON settings. Use a new output directory for each experiment; existing run outputs are protected from overwriting.

Evaluation automatically follows `--num_runs`, `--num_samples`, and `--targets`, including custom budgets. The program refuses to finalize missing runs, incomplete attempt records, duplicate identifiers, or missing configured targets.

## 3. Read the results

The output directory contains the following results:

| File | Contents |
| --- | --- |
| `best_max_tanimoto.csv` | Highest run maximum per target |
| `run_max_tanimoto.csv` | Maximum Tanimoto and matching molecule–ligand pair for each run |
| `evaluation_summary.json` | Mean of selected target maxima and evaluation settings |
| `run_metrics.csv` | Molecular metrics for each run |
| `aggregate_metrics.csv` | Per-target metric summaries |
| `metadata.json` | Run settings, seeds, software versions, and checkpoint/data hashes |

For each target, select the highest maximum Tanimoto across ten runs, using **all valid generated molecules**. The overall mean averages the ten selected target maxima. Similarity uses radius-2, 2,048-bit Morgan fingerprints against known ligands after excluding training-set molecules.

See [Methods](METHODS.md) for metric definitions and [evaluation examples](../examples/paper_protocol/) for expected results and CPU reevaluation. The independent compound–phenotype compatibility predictor is outside this three-stage release.

## 4. Train the three-stage pipeline

This workflow trains a new GeneVAE, post-trains the molecular backbone with that GeneVAE frozen, and generates with the selected new checkpoint. It follows the documented recipe; it does not promise byte-identical reconstruction of an earlier unseeded historical run.

### Download the starting molecular backbone

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --base
```

This downloads the approximately 630 MB `novomolgen-300m-base.tar.gz`, verifies it, and installs the pinned NovoMolGen 300M AtomWise backbone under `pretrained/novomolgen/`. This unconditioned starting model differs from the released epoch-9 conditional checkpoint. Upstream attribution is in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### Stage 1: pretrain GeneVAE

```bash
python -m tx2mol.pretrain --config configs/pretrain.json
```

The input is `data/train.csv.gz`. GeneVAE uses 978 inputs, hidden widths 512–256–128, a 64-dimensional Gaussian latent space, and a mirrored decoder. Defaults are Adam, learning rate `1e-4`, batch size 64, dropout 0.2, seed 0, and 2,000 epochs. The reconstruction coefficient decreases from 0.99 to 0.5 over the first 1,000 epochs, then stays fixed.

Outputs are `outputs/gene_vae/gene_vae.pt`, `args.json`, `gene_order.json`, and `loss.csv`. This stage saves the final epoch's model without validation early stopping.

### Stage 2: post-train Tx2Mol

```bash
python -m tx2mol.finetune --config configs/finetune.json
```

The configuration loads the Stage 1 GeneVAE, the downloaded backbone, and the supplied train/validation data. GeneVAE is frozen; its sampled latent vector is combined with cell-line encoding to construct the conditioning prefix.

| Setting | Default |
| --- | --- |
| Optimizer | AdamW |
| Generator / projection learning rate | `2e-5` / `5e-5` |
| Batch size / gradient accumulation | 32 / 1 |
| Maximum epochs / early-stopping patience | 20 / 3 |
| Warm-up / scheduler | 10% / cosine decay |
| Maximum SMILES length | 100 |
| Precision / seed | BF16 / 42 |
| Historical contrastive weight / temperature | 0.2 / 0.07 |

Validation improvement selects checkpoints using the implemented generation-based criterion. `outputs/tx2mol/training_log.csv` records progress; `outputs/tx2mol/best_checkpoint.json` points to the selected checkpoint. That directory includes the model, projection, tokenizer, cell-line mapping, gene order, resolved settings, and expected GeneVAE hash.

The historical InfoNCE BOS-embedding fallback is retained and reported by the code. It does not represent whole-molecule contrastive alignment; fixing it would require a separately trained model. See [the implemented protocol](METHODS.md#historical-infonce-limitation).

### Stage 3: generate with your new checkpoint

```bash
python -m tx2mol.generate --config configs/generate.json
```

The loader resolves `outputs/tx2mol/best_checkpoint.json` and uses `outputs/gene_vae/gene_vae.pt`. It applies the same ten-target sampling and evaluation protocol, then writes molecules, run scores, selected maxima, and winning groups together under `outputs/targets/`.

After installing the base asset, `bash scripts/run_all.sh` runs the three stages in sequence. Use it when the default output paths are unused.

### Repeat an experiment in a separate directory

Override all connected paths consistently:

```bash
python -m tx2mol.pretrain --config configs/pretrain.json \
  --output_dir outputs/run02/gene_vae

python -m tx2mol.finetune --config configs/finetune.json \
  --saved_gene_vae outputs/run02/gene_vae/gene_vae.pt \
  --output_dir outputs/run02/tx2mol

python -m tx2mol.generate --config configs/generate.json \
  --saved_gene_vae outputs/run02/gene_vae/gene_vae.pt \
  --model_dir outputs/run02/tx2mol \
  --output_dir outputs/run02/generated
```

Keep each GeneVAE paired with the Tx2Mol model trained on its latent space. The loader checks recorded hashes. Do not substitute a newly pretrained GeneVAE into the archived epoch-9 model.

## 5. Validate the pipeline

In the installed environment, run the CPU regression suite and data checks:

```bash
python -m unittest discover -s tests -v
python scripts/validate_data.py
```

After installing the base asset, test the complete connection on a CUDA GPU:

```bash
bash scripts/smoke_test.sh outputs/smoke01
```

This performs two GeneVAE updates, two Tx2Mol updates on small example subsets, saves/reloads the new checkpoint, and attempts two molecules per target. The smoke test selects checkpoints by validation language-model loss, explicitly differing from the full training criterion. It checks installation and interoperability, not scientific performance.

The earlier Linux/CUDA checks and limitations are recorded in [`docs/VALIDATION.md`](VALIDATION.md). Full 2,000/20-epoch training was not repeated during packaging, and installation on a clean host was not independently tested.

## 6. Datasets and custom inputs

```text
data/
├── train.csv.gz                  # 25,364 supplied training rows
├── val.csv.gz                    # 2,000 supplied validation rows
├── test.csv.gz                   # 14 supplied held-out rows
├── gene_order.json               # expected order of 978 gene IDs
├── examples/                     # small subsets for pipeline checks
├── targets/
│   ├── test/                     # ten target signatures
│   └── known_ligands/            # source_TARGET.csv
├── sciplex3/
│   ├── test/                     # A549, K562, MCF7 matrices
│   └── known_ligands/            # paired perturbing molecules
└── patient/
    ├── test/                     # 12 disease input files
    └── known_ligands/            # 12 default ligand collections
        └── alternate_collection/ # 12 separately retained collections
```

The shared training matrices are headerless: `cell_line, compound_id, SMILES`, followed by 978 expression values. These are already processed signatures. The code preserves their values and order; upstream normalization and biological gene-ID alignment cannot be reconstructed from the headerless files alone. The 14-row held-out split is not a large additional benchmark.

The executable tutorial covers **targets**. SciPlex3 has 144 rows per cell line, 11 metadata columns, and 978 expression columns; its 143 non-control references retain `test_row` pairing. Patient data are restricted to the 12 diseases in the updated disease heatmap, with matching default and alternate ligand collections. These datasets are supplied with their original formats; the target-only CLI does not automatically implement the SciPlex3 or patient protocols. See [`data/README.md`](../data/README.md) for the disease list and file names.

For an additional target, prepare `YOUR_TARGET.csv` with the 978 gene IDs from `data/gene_order.json` as its header and the numeric signature in the first data row. Prepare `source_YOUR_TARGET.csv` with a `SMILES` column, and choose a cell line supported by the checkpoint's mapping:

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --target_dir /path/to/custom/test \
  --source_ligands_dir /path/to/custom/known_ligands \
  --targets YOUR_TARGET --cell_line MCF7 \
  --output_dir outputs/custom_target
```

Match the intended biological protocol and expression scale before running. The loader uses the first numeric row and does not normalize, reverse, pad, or reorder signatures. It rejects nonfinite values, wrong dimensions, mismatched gene headers, and unsupported cell labels.

## 7. Troubleshooting

| Problem | What to check |
| --- | --- |
| Environment or FlashAttention errors | Use the pinned environment in [section 1](#1-install-the-environment) and rerun `bash scripts/install.sh`. |
| CUDA out of memory | Reduce `--batch_size`. During post-training, adjust `--grad_accum` to preserve the effective batch size. |
| Checkpoint loading errors | Check model paths, install the required weights from sections 2 or 4, and use the GeneVAE paired with the Tx2Mol checkpoint. |

## Reproducibility records

- [`configs/`](../configs/): executable settings for all three stages and archived generation.
- [`assets/data_manifest.json`](../assets/data_manifest.json): exact supplied data hashes and provenance.
- [`assets/release_manifest.json`](../assets/release_manifest.json): archive and checkpoint hashes.
- [`docs/METHODS.md`](METHODS.md): architecture, losses, tokenization, metrics, and seeds.
- [`docs/PROVENANCE.md`](PROVENANCE.md): historical entry points, integration changes, and provenance limitations.
- [`docs/VALIDATION.md`](VALIDATION.md): checks performed and their scope.

Fresh runs explicitly use seed 0 for GeneVAE, 42 for post-training, and `42 + run_idx` for generation. GeneVAE samples its latent variable even in evaluation mode. Keep hardware, software versions, and batch size fixed for strict comparisons, and retain each run's `metadata.json`.

The earlier [combined source-and-checkpoint ZIP](https://github.com/Yaxin-Xu/Tx2Mol/releases/download/v1.0/Tx2Mol-with-checkpoint.zip) remains an archived snapshot. The current repository files and this tutorial are the maintained entry point; large weights remain downloadable Release assets.
