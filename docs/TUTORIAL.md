# Reproduction tutorial

[Project overview](../README.md) · [Methods and parameters](METHODS.md)

```text
GeneVAE pretraining -> Tx2Mol post-training -> phenotype-guided SMILES generation
```

Use the released checkpoint in section 2, or train a new model in section 4.

<a id="1-install-the-environment"></a>

## 🛠️ 1. Install the environment

Requires Linux, Conda, Git, and one NVIDIA Ampere-or-newer GPU. The pipeline was tested on an RTX 4090 (24 GB). Reserve at least 10 GB; training checkpoints need additional space. The environment pins Python 3.10, PyTorch 2.1.2, CUDA 11.8, Transformers 4.46.2, and FlashAttention 2.6.1. GeneVAE alone can run on CPU; the full pipeline requires CUDA.

```bash
git clone https://github.com/Yaxin-Xu/Tx2Mol.git
cd Tx2Mol
conda env create -f environment.yml
conda activate tx2mol
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
bash scripts/install.sh
python scripts/check_environment.py
python scripts/validate_data.py
```

Check for `cuda_available: true` and `Data validated.` Run subsequent commands from the repository root with the `tx2mol` environment active. Choose a new output directory for each experiment. Explicit command-line options override JSON settings.

<a id="2-quick-start-use-the-released-checkpoint"></a>

## 🚀 2. Quick start: use the released checkpoint

Download the epoch-9 Tx2Mol checkpoint and its paired GeneVAE:

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --reference
```

The approximately 1.13 GB archive is downloaded from [Releases](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0), verified against the weight manifest, and installed under `checkpoints/reference/`.

### Generate a small example for all ten targets

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --num_runs 1 --num_samples 10 --batch_size 10 \
  --output_dir outputs/reference_demo
```

This samples ten molecules for each of AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53. The sampling budget includes invalid strings and duplicates. Generation automatically evaluates the molecules and saves the results.

### Run the full ten-target example

```bash
python -m tx2mol.generate --config configs/generate_reference.json
```

Defaults are **10 targets × 10 runs × 100 samples**, saved under `outputs/reference_targets/`. These are sampling runs with a fixed checkpoint. Settings are in [generate_reference.json](../configs/generate_reference.json).

To change the target or sampling budget:

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --targets EGFR --num_runs 3 --num_samples 100 \
  --output_dir outputs/reference_egfr
```

## 3. Read the results

| File | Contents |
| --- | --- |
| `best_max_tanimoto.csv` | Highest run maximum per target |
| `run_max_tanimoto.csv` | Maximum Tanimoto and matching molecule–ligand pair for each run |
| `evaluation_summary.json` | Mean of selected target maxima and evaluation settings |
| `run_metrics.csv` | Molecular metrics for each run |
| `aggregate_metrics.csv` | Per-target metric summaries |
| `metadata.json` | Run settings, seeds, software versions, and checkpoint/data hashes |

The selected score is **each target's highest maximum Tanimoto**, scoring all valid molecules. The overall mean averages the selected target maxima. See [Methods](METHODS.md#paper-maximum-tanimoto-evaluation) for the exact calculation.

<a id="4-train-the-three-stage-pipeline"></a>

## 🧠 4. Train the three-stage pipeline

Download the starting molecular backbone:

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --base
```

This installs the approximately 630 MB base asset under `pretrained/novomolgen/`.

<a id="stage-1-pretrain-genevae"></a>

### 🧬 Stage 1: pretrain GeneVAE

```bash
python -m tx2mol.pretrain --config configs/pretrain.json
```

Uses `data/train.csv.gz` and saves the final model to `outputs/gene_vae/gene_vae.pt`, alongside the settings, gene order, and loss log.

<a id="stage-2-post-train-tx2mol"></a>

### 🧠 Stage 2: post-train Tx2Mol

```bash
python -m tx2mol.finetune --config configs/finetune.json
```

Loads the Stage 1 GeneVAE and keeps it frozen while training the molecular model and conditioning projection. Progress is saved to `outputs/tx2mol/training_log.csv`; `best_checkpoint.json` identifies the checkpoint selected by the generation-based composite score.

<a id="stage-3-generate-with-your-new-checkpoint"></a>

### 🧪 Stage 3: generate with your new checkpoint

```bash
python -m tx2mol.generate --config configs/generate.json
```

Loads the selected checkpoint and its paired GeneVAE, then generates and evaluates the ten-target example under `outputs/targets/`.

The defaults connect all three stages. After installing the base asset, `bash scripts/run_all.sh` runs them in sequence. Architecture, training settings, and evaluation definitions are documented in [Methods](METHODS.md); editable settings are in [configs](../configs/).

### Repeat an experiment in a separate directory

Keep each GeneVAE paired with the Tx2Mol model trained on its latent space. Update all connected paths together:

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

## 5. Validate the pipeline

```bash
python -m unittest discover -s tests -v
python scripts/validate_data.py
```

Completed checks and their scope are summarized in [Validation](VALIDATION.md).

## 6. Datasets and custom inputs

Shared training data are under `data/`. The `targets/`, `sciplex3/`, and `patient/` directories each contain `test/` and `known_ligands/`; patient data cover the 12 diseases in the updated heatmap. See the [data guide](../data/README.md) for formats and file lists. This tutorial implements the target workflow; SciPlex3 and patient analyses require their respective protocols.

For a new target, prepare:

- `YOUR_TARGET.csv`: the 978 ordered gene IDs from `data/gene_order.json` as the header, followed by the numeric signature. The loader uses the first data row without normalization or gene reordering.
- `source_YOUR_TARGET.csv`: known ligands in a `SMILES` column.

Use a cell line supported by the checkpoint:

```bash
python -m tx2mol.generate --config configs/generate_reference.json \
  --target_dir /path/to/custom/test \
  --source_ligands_dir /path/to/custom/known_ligands \
  --targets YOUR_TARGET --cell_line MCF7 \
  --output_dir outputs/custom_target
```

## 7. Troubleshooting

| Problem | What to check |
| --- | --- |
| Environment or FlashAttention errors | Use the pinned environment and rerun `bash scripts/install.sh`. |
| CUDA out of memory | Reduce `--batch_size`; adjust `--grad_accum` during post-training. |
| Checkpoint loading errors | Check model paths and use the GeneVAE paired with the Tx2Mol checkpoint. |

## Reproducibility records

Retain each run's `metadata.json`. [Methods](METHODS.md) defines metrics and seeds; [Provenance](PROVENANCE.md) records code, data, and checkpoint origins. [Data](../assets/data_manifest.json) and [weight](../assets/release_manifest.json) manifests provide checksums.
