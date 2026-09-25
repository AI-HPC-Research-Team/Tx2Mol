# Complete generation and automatic evaluation example

These files were produced by the current pipeline on 2026-09-25 with the released Tx2Mol checkpoint and its matching GeneVAE:

```bash
python -m tx2mol.generate --config configs/generate_reference.json
```

The command completed **10 targets × 10 runs × 100 attempts**, including **9,730 valid molecules**. It automatically selected the highest maximum Tanimoto for each target and exported all **1,000 attempts in the winning groups**. The mean of the ten selected maxima was **0.9300271739130433**. This is a fresh seeded execution; the historical experiment's saved molecules give 0.9136607142857143. Fresh samples can differ.

| File | Contents |
| --- | --- |
| `raw_attempts.csv` | All 10,000 attempts, including invalid strings and duplicates |
| `run_max_tanimoto.csv` | All 100 run maxima and witness molecule/ligand pairs |
| `best_max_tanimoto.csv` | Selected run, maximum score, and witness pair per target |
| `best_run_attempts.csv` | The ten complete winning groups |
| `run_metrics.csv`, `aggregate_metrics.csv` | Molecular diagnostics and the same all-valid maximum scores |
| `evaluation_summary.json` | Protocol, software versions, hashes, and mean selected maximum |
| `resolved_config.json` | The executed sampling configuration |
| `verification.json` | Independent comparisons, checkpoint/source identities, and computational environment |

All 100 run maxima match the unchanged historical evaluator exactly. A separate CPU reevaluation produced byte-identical run-score, best-score, and winning-group CSVs. All 100 witness pairs were also checked with scalar Tanimoto calculations. Reevaluate this saved dataset without model weights:

```bash
python -m tx2mol.evaluate \
  --attempts examples/integrated_generation/raw_attempts.csv \
  --output-dir outputs/integrated_example_rescored
```

Run commands from the repository root, use the documented environment, and choose a new output directory. See the [full tutorial](../../docs/TUTORIAL.md) for installation and training. `SHA256SUMS` covers the example files.
