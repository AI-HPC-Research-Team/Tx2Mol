# Validation performed on this release

Validated on 2026-09-23 using one NVIDIA RTX 4090 (24 GB), Python 3.10.18, PyTorch 2.1.2+cu118, Transformers 4.46.2, Accelerate 0.34.0, FlashAttention 2.6.1 and NumPy 1.26.4. The initial inventory reported RDKit distribution metadata as 2024.9.6; the subsequent generation audit found the imported runtime to be **2022.09.5**. Runtime and installed-distribution version records should not be conflated.

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

## Paper-protocol recovery and full generation checks (2026-09-25)

- Reevaluated every archived run with the recovered independent evaluator: **all 100 per-run maximum Tanimoto scores match the historical evaluation table exactly**. The mean of the ten selected target maxima is **0.9136607142857143**.
- The earlier public-generator snapshot completed **10,000 attempts**, with 9,728 valid molecules. Reapplying the recovered all-valid evaluator gives **0.9300271739130433**. That snapshot's built-in metrics used the novel subset; the maintained generator now uses all-valid similarity directly.
- The original sampler completed a separate **10,000-attempt** run with 9,737 valid molecules and mean selected maximum **0.9205116245694605**. Seven of ten selected target maxima equal the historical values. The only sampling-run additions were reference-fingerprint caching and raw/RNG tracing; 3,000 scalar pair comparisons verified the cache preserved scores. An incomplete timing trial was excluded.
- Independently checked all **300 per-run maxima** and **30 witness molecule pairs** across these three datasets. Public gzip training data produce the same scores as the original server inputs.
- A larger fixed-seed repeat check matched 81/100 strings in one repetition and 100/100 in another. The earlier ten-string agreement does not establish general bitwise determinism.

All historical valid-SMILES lists and both fresh raw-attempt datasets are published in `examples/paper_protocol/`. These generation/evaluation checks use existing weights; full model training and a clean-host installation were not repeated.

Before publication, the bundled commands were run on all three datasets using the repository's gzip training data and known ligands, on macOS with Python 3.12.4, pandas 2.2.3, and RDKit 2025.09.4. All 300 per-run maxima matched the server results exactly, all 30 witness pairs were verified, and both sets of 1,000 selected attempts matched their complete source groups. **All 23 regression tests passed**, including the unchanged-evaluator checksum, all-valid scoring, canonical training exclusion, tie handling, complete winning-group export, malformed/incomplete input rejection, and overwrite protection. See `examples/paper_protocol/publication_verification.json`.

## Integrated generation and evaluation (2026-09-25)

The current generator calls `tx2mol.evaluate` directly, scores all valid molecules during generation, and exports the complete winning groups automatically. Its aggregate maximum is selected across runs, not averaged. The CPU reevaluation command uses the same scorer.

All **300 archived per-run maxima** match the unchanged historical evaluator exactly under the native implementation. Both sets of 1,000 winning attempts match their full source groups. **25 tests passed on both macOS and the Linux server**, including the complete generation-to-evaluation control flow with a mocked sampler, non-novel molecules attaining the maximum, ligand deduplication, missing-target rejection, and automatic result export. The numerical comparison and source hashes are in `assets/integrated_evaluation_validation.json`. These checks do not rerun training.

The exact full-example command, `python -m tx2mol.generate --config configs/generate_reference.json`, then completed **10,000 fresh attempts, 9,730 valid molecules, and 100 runs** on the RTX 4090. It automatically wrote all scores and **1,000 attempts in the ten winning groups**. The mean of selected target maxima was **0.9300271739130433**. All 100 scores matched the unchanged historical evaluator exactly; all 100 witness pairs passed scalar Tanimoto checks. A separate CPU reevaluation produced byte-identical run-score, selected-score, and winning-group CSVs. The complete execution example is in `examples/integrated_generation/`.

The pinned `rdkit==2024.9.6` was also installed into an isolated directory and verified by its imported runtime version, without changing the existing GPU environment. Under that pinned runtime, all 25 tests passed; the earlier original-sampler dataset and the new integrated dataset each retained all 100 per-run scores exactly. The tutorial's **100-attempt GPU quick start** also completed with 100 valid molecules and automatic evaluation/export using RDKit 2024.09.6. The full 10,000-attempt generation above used the server's existing RDKit 2022.09.5 runtime. A complete clean-host environment installation and full retraining remain outside these checks.
