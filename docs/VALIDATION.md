# Validation performed on this release

Validated on 2026-09-23 using one NVIDIA RTX 4090 (24 GB), Python 3.10.18, PyTorch 2.1.2+cu118, Transformers 4.46.2, Accelerate 0.34.0, FlashAttention 2.6.1, RDKit 2024.9.6 and NumPy 1.26.4.

Completed checks:

- **19 regression tests passed**, including legacy checkpoint shapes, VAE loss/interface, preserving headerless first rows, gzip reading, strict gene ordering, invalid input rejection, correct checkpoint/VAE matching, sampling every attempted string, duplicate-aware metrics, across-run aggregation, historical InfoNCE fallback, configuration overrides, and refusing to overwrite pretraining weights.
- Full train/validation/test matrices and all ten target/ligand pairs passed SHA256, shape and finite-value checks.
- **Three-stage integration:** two GeneVAE optimizer updates on CPU; two full-backbone Tx2Mol updates; save/reload the resulting checkpoint; two generation attempts for every target, **20 attempts total**.
- A separate GeneVAE update on CUDA passed using the default 2,000-epoch configuration with an explicit one-step cap.
- **Default batch size 32** passed a GPU optimizer update on the complete training matrix. The full generation-based validation criterion was exercised with eight generated samples and saved a loadable checkpoint. This was a capped validation run, not a complete 20-epoch experiment.
- The archived epoch-9 model and its matching GeneVAE generated **10 attempts × 10 targets = 100 attempts**. All 100 were valid under the specified RDKit rule. The raw strings and per-target metrics are in `examples/reference_demo/`; the small sample is an execution example, not a new performance estimate.
- Repeating the archived-model AKT1 example with the same seed, batch size and environment produced the same ten raw SMILES strings.

The complete 2,000-epoch GeneVAE and up-to-20-epoch Tx2Mol runs were **not** repeated during packaging. The environment specification matches the tested existing environment; installing every package from scratch on a clean host was not separately tested.

The last packaging change added only an early existing-output guard to GeneVAE pretraining. Its refusal and unchanged-checkpoint-hash regression passed after the integration run. Runtime manifests retain the exact source hashes used by each check.

`examples/reference_demo/metadata.json` records checkpoint/data hashes and configuration. Source and weight integrity manifests are supplied in `assets/`. CPU tests require the documented PyTorch/RDKit dependencies but do not require the large molecular weights.

## Repository publication checks (2026-09-25)

The 19 CPU regression tests and complete supplied-data validation were rerun locally on macOS with Python 3.12 and PyTorch 2.6.0. One test assertion now resolves the expected temporary directory path to accommodate macOS `/var` versus `/private/var` aliases; model computations are unchanged. These checks do not replace the Linux/CUDA validation above. CSV files are excluded from Git text normalization so their recorded byte hashes survive cloning.

## Patient cohort update (2026-09-25)

The test, default-ligand, and alternate-ligand directories each contain exactly the 12 disease stems in the updated heatmap. The extra historical `gastric` files were removed, while `stomach cancer` was retained. All retained data hashes are unchanged. Complete data validation passed for the shared datasets and all 62 remaining scenario CSV files, including SciPlex3 row pairing. This data-only update did not rerun model training or generation.
