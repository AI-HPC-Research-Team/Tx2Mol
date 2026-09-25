# Reproduce the maximum Tanimoto results

This example supplies every run used in the comparison, not only the selected winners. Each dataset covers ten targets, ten runs per target, and 100 generation attempts per run. Evaluation uses all valid generated molecules, Morgan radius 2 / 2,048 bits, and known ligands after canonical training-set exclusion. The highest run maximum is selected for each target.

Run these commands from the repository root in the documented environment. Evaluation runs on CPU and needs no model weights. Always choose a new output directory.

## Recompute the historical 0.914

```bash
python scripts/evaluate_gxvaes_protocol.py \
  --train data/train.csv.gz \
  --sources data/targets/known_ligands \
  --output-dir outputs/historical_paper_evaluation \
  examples/paper_protocol/historical/generated_molecules
```

`historical/generated_molecules/all_runs_statistics.csv` contains all 100 original valid-SMILES lists. The expected per-run and selected-best scores are in `historical/`. The mean of the ten selected maxima is **0.9136607142857143**, which rounds to **0.914**. This reevaluates archived samples; it does not generate new ones. Invalid raw strings were not retained in the historical source and have not been reconstructed.

## Reevaluate the public generator's 10,000 attempts

```bash
python scripts/evaluate_release_attempts.py \
  --attempts examples/paper_protocol/released/raw_attempts.csv \
  --output-dir outputs/released_paper_evaluation
```

Read `best_max_tanimoto.csv` for the ten selected maxima and `best_run_attempts.csv` for the complete winning groups (1,000 attempts total). The expected mean is **0.9300271739130433**. To rescore the fresh original-sampler run, replace the input with `examples/paper_protocol/fresh_original/raw_attempts.csv` and choose another output directory; its mean is **0.9205116245694605**.

Both fresh-data folders also include these selected-result files for direct inspection. Their full raw inputs and all-run scores remain available beside them.

The two fresh datasets are separate stochastic experiments with the same archived model and GeneVAE. The public generator uses seeds 42–51; the original sampler did not explicitly reset seeds. Their differences are not evidence of exact regeneration of historical molecules.

## Expected selected maxima

| Target | Historical | Public generator | Fresh original sampler |
| --- | ---: | ---: | ---: |
| AKT1 | 1.000000 | 1.000000 | 1.000000 |
| AKT2 | 1.000000 | 1.000000 | 1.000000 |
| AURKB | 1.000000 | 1.000000 | 1.000000 |
| CTSK | 0.450000 | 0.456522 | 0.480769 |
| EGFR | 1.000000 | 1.000000 | 1.000000 |
| HDAC1 | 0.843750 | 0.843750 | 0.843750 |
| MTOR | 1.000000 | 1.000000 | 0.880597 |
| PIK3CA | 0.842857 | 1.000000 | 1.000000 |
| SMAD3 | 1.000000 | 1.000000 | 1.000000 |
| TP53 | 1.000000 | 1.000000 | 1.000000 |
| Mean of selected target maxima | **0.913661** | **0.930027** | **0.920512** |

`maximum_tanimoto_comparison.csv` retains full precision and the fresh winning group numbers. `maximum_tanimoto_witness_pairs.csv` records a generated molecule and eligible known ligand attaining each selected maximum; `valid_ordinal` is zero-based within that run's valid-molecule list. Fingerprints use the original default `useChirality=False`, so a score of 1 does not necessarily mean identical stereochemistry.

`provenance.json` records source/checkpoint identities and sampling settings; `SHA256SUMS` covers the example inputs and expected results. The recovered evaluator is unchanged. The public raw-attempt file retains the generator's novel-subset similarity fields for provenance; use the separate paper evaluator for the all-valid maximum. See [methods](../../docs/METHODS.md#paper-maximum-tanimoto-evaluation) and [validation](../../docs/VALIDATION.md).
