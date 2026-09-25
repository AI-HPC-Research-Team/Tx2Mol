# Validation summary

Checks completed on 2026-09-23–25. Results apply to the source versions recorded with each check.

## Environment

GPU checks used one NVIDIA RTX 4090 (24 GB), Python 3.10.18, PyTorch 2.1.2+cu118, CUDA 11.8, Transformers 4.46.2, Accelerate 0.34.0, FlashAttention 2.6.1, and NumPy 1.26.4. The full generation run imported RDKit **2022.09.5**; the initial package inventory had reported 2024.9.6. The pinned **2024.09.6** runtime was subsequently tested separately, as specified below.

## Completed checks

| Check | Result |
| --- | --- |
| Regression suite | 25 tests passed on Linux and macOS, covering VAE/checkpoint compatibility, data loading, sampling, and integrated evaluation. |
| Supplied data | Hash, shape, gene-order, finite-value, and SciPlex3 pairing checks passed; all 62 scenario CSV files were checked. Patient inputs and each ligand collection contain the same 12 disease stems. |
| Three-stage integration | Two GeneVAE updates, two Tx2Mol updates, checkpoint save/reload, and 20 generation samples completed. |
| Training defaults | A GeneVAE CUDA update and a Tx2Mol batch-32 update passed; generation-based validation saved a loadable checkpoint. |
| Historical evaluation | All 300 run maxima across three archived datasets matched the independent evaluator; all 30 selected molecule–ligand pairs were verified. |
| Integrated full example | 10,000 samples, 9,730 valid molecules, and 100 runs completed; scores and all 1,000 samples in the selected groups were exported automatically. All 100 run scores and matching molecule pairs were verified independently; CPU reevaluation produced identical result CSVs. |
| Pinned RDKit 2024.09.6 | All 25 tests passed; reevaluation preserved all 200 scores from the original-sampler and integrated datasets. The 100-sample GPU quick start completed with 100 valid molecules. |
| Latest code cleanup | All 25 CPU tests and data checks passed; the retained generation-based checkpoint evaluation was verified unchanged. |

## Evaluation results

Each value is the mean of the ten selected target maxima, calculated with the same all-valid maximum Tanimoto protocol.

| Dataset | Mean selected maximum |
| --- | --- |
| Historical paper results | 0.9136607143 |
| Earlier public-generator run | 0.9300271739 |
| Fresh original-sampler run | 0.9205116246 |
| Integrated generation run | 0.9300271739 |

These are separate sampling outcomes. A fixed-seed repeat check matched 81/100 strings in one repetition and 100/100 in another; bitwise-identical generation is not guaranteed.

## Scope and evidence

The complete 2,000-epoch GeneVAE and up-to-20-epoch Tx2Mol training runs were not repeated during packaging, and a clean-host installation was not independently tested. Short training checks establish execution only. The full 10,000-sample run used RDKit 2022.09.5; the separate pinned-runtime checks do not constitute full retraining.

- [Integrated execution example](../examples/integrated_generation/): complete outputs and verification report.
- [Paper-protocol examples](../examples/paper_protocol/): archived inputs, expected scores, and CPU reevaluation commands.
- [Validation record](../assets/integrated_evaluation_validation.json): comparisons, source hashes, and environment details.
- [Detailed dated log](https://github.com/AI-HPC-Research-Team/Tx2Mol/blob/673f083119bedaa8e0de807eae39425f4aa3c297/docs/VALIDATION.md): original validation history.
