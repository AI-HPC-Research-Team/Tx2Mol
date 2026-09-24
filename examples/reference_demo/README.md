# Archived-model execution example

100 attempts: ten per target, one run, seed 42, batch size 10, maximum length 100.
These are actual release validation outputs, not full ten-run performance results.
All raw attempts are retained. Some targets have no novel molecules in this small
sample; the legacy empty-set metric zeros must be read with the reported counts.

Reproduce with:

```bash
python -m tx2mol.generate --config configs/generate_reference.json --num_runs 1 --num_samples 10 --batch_size 10 --output_dir outputs/reference_demo
```

The metadata records the original validation output path. Paths are relative to
the repository except resolved checkpoint paths used on the validation machine;
hashes, configuration and seed policy establish the portable identities.
The validation metadata predates the input-folder reorganization: target files
now live under `data/targets/test/`, with unchanged content and hashes.
The corresponding ligands now live under `data/targets/known_ligands/`.
