# Tx2Mol

Phenotype-driven de novo molecular design from gene expression signatures.

```text
GeneVAE pretraining -> Tx2Mol post-training -> phenotype-guided SMILES generation
```

Start with the commands below. The [full reproduction tutorial](docs/TUTORIAL.md) explains parameters, outputs, custom inputs, and repeated runs.

## 1. Install the environment

Requires Linux, Conda, and an NVIDIA Ampere-or-newer GPU. The pipeline was tested on one RTX 4090 (24 GB); the environment pins Python 3.10 and CUDA 11.8.

```bash
git clone https://github.com/Yaxin-Xu/Tx2Mol.git
cd Tx2Mol
conda env create -f environment.yml
conda activate tx2mol
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
bash scripts/install.sh
python scripts/validate_data.py
```

Run subsequent commands from the repository root with the `tx2mol` environment active. Use a new output directory for each experiment.

## 2. Quick start: use the released checkpoint

Download the Tx2Mol checkpoint and its matching GeneVAE, then generate a small ten-target example:

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --reference

python -m tx2mol.generate --config configs/generate_reference.json \
  --num_runs 1 --num_samples 10 --batch_size 10 \
  --output_dir outputs/reference_demo
```

This makes **100 generation attempts** across ten targets. Weights are downloaded from [Releases](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0) and verified automatically. An [archived execution example](examples/reference_demo/) is included.

For the full example (**10 targets × 10 runs × 100 attempts**):

```bash
python -m tx2mol.generate --config configs/generate_reference.json
python scripts/evaluate_release_attempts.py \
  --attempts outputs/reference_targets/raw_attempts.csv \
  --output-dir outputs/reference_targets_evaluation
```

## 3. Read the results

For the full example, read `outputs/reference_targets_evaluation/best_max_tanimoto.csv`: **each target's highest maximum Tanimoto across ten runs**, scoring all valid molecules. `best_run_attempts.csv` contains every attempt in the winning groups. The generator's `raw_attempts.csv` retains all samples; `aggregate_metrics.csv` contains separate across-run diagnostics, including novel-only similarity.

[Recompute the archived results](examples/paper_protocol/) without a GPU: the mean of the ten selected target maxima is **0.9136607 (0.914)**. Fresh samples may differ. See [metric definitions and outputs](docs/TUTORIAL.md#3-read-the-results).

## 4. Train the three-stage pipeline

Install the starting backbone, then run GeneVAE pretraining, Tx2Mol post-training, and generation in order:

```bash
python scripts/prepare_assets.py \
  --github_repo Yaxin-Xu/Tx2Mol --tag v1.0 --base

python -m tx2mol.pretrain --config configs/pretrain.json
python -m tx2mol.finetune --config configs/finetune.json
python -m tx2mol.generate --config configs/generate.json
python scripts/evaluate_release_attempts.py \
  --attempts outputs/targets/raw_attempts.csv \
  --output-dir outputs/targets_evaluation
```

The defaults connect all three stages and save generated results under `outputs/targets/`. Keep each Tx2Mol checkpoint paired with the GeneVAE used during its training. See the [training tutorial](docs/TUTORIAL.md#4-train-the-three-stage-pipeline) for hyperparameters, backbone attribution, and implementation details.

## 5. Validate the pipeline

```bash
python -m unittest discover -s tests -v
# After installing the base weights, run a short GPU integration check:
bash scripts/smoke_test.sh outputs/smoke01
```

The smoke test checks execution, not scientific performance. Completed checks and their scope are recorded in [VALIDATION.md](docs/VALIDATION.md).

## 6. Datasets and documentation

Shared training data are under `data/`. The `targets/`, `sciplex3/`, and `patient/` subdirectories each contain `test/` and `known_ligands/`; patient data cover the 12 diseases in the updated heatmap. The executable example covers ten targets; SciPlex3 and patient data require their respective protocols.

- [Full tutorial](docs/TUTORIAL.md): installation, generation, training, and custom inputs.
- [Data guide](data/README.md): file formats, disease list, and ligand collections.
- [Methods](docs/METHODS.md) and [provenance](docs/PROVENANCE.md): architecture, metrics, seeds, and implementation limitations.
- [Configurations](configs/), [data manifest](assets/data_manifest.json), and [weight manifest](assets/release_manifest.json): reproducibility settings and checksums.

## 7. Troubleshooting

| Problem | Solution |
| --- | --- |
| Environment or FlashAttention errors | Use the pinned environment and rerun `bash scripts/install.sh`. |
| CUDA out of memory | Reduce `--batch_size`; adjust `--grad_accum` during post-training. |
| Checkpoint loading errors | Check model paths and use the GeneVAE paired with the Tx2Mol checkpoint. |
