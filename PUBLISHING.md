# Maintaining this release

The public repository is [Yaxin-Xu/Tx2Mol](https://github.com/Yaxin-Xu/Tx2Mol). Users should start with the [README quick start](README.md). The [full reproduction tutorial](docs/TUTORIAL.md) covers environment setup, GeneVAE pretraining, Tx2Mol post-training, and ten-target generation in detail.

## Repository files and model assets

Keep source code, configurations, tests, documentation, and supplied datasets as ordinary files on `main`. Preserve `.gitattributes`: CSV files retain their original bytes so that the data manifest remains valid after cloning. The small reference GeneVAE is tracked in Git.

The large molecular weights are separate assets of [release v1.0](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0):

| Asset | Purpose |
| --- | --- |
| `novomolgen-300m-base.tar.gz` | Starting backbone for fresh Tx2Mol post-training |
| `tx2mol-reference-checkpoints.tar.gz` | Archived epoch-9 Tx2Mol checkpoint and its paired GeneVAE |

`scripts/prepare_assets.py` downloads and verifies these archives using `assets/release_manifest.json`. Extracted model directories, downloads, and new experiment outputs are ignored by Git. Do not rename or replace a published asset without updating its manifest and documenting the provenance of the new model.

The earlier `Tx2Mol-with-checkpoint.zip` and its manifest remain historical snapshots. The `v1.0` tag predates the expanded source publication, so GitHub's automatically generated source archives for that tag do not contain the current pipeline. Clone the repository as shown in the README; the two named model assets work with that checkout.

## Before publishing a source update

Run from the repository root in the installed environment:

```bash
python -m unittest discover -s tests -v
python scripts/validate_data.py
git diff --check -- . ':(exclude)**/*.csv'
git status --short
```

When model loading or the training/generation interface changes, also run the documented GPU smoke test in a new output directory and record its scope in `docs/VALIDATION.md`. A packaging check or a short smoke test does not establish reproduction of full scientific results.

Review the exact staged changes, commit them, and push normally without rewriting published history. Retain the original scientific records and state any newly tested environment or implementation change explicitly.
