# Tx2Mol
Phenotype-driven de novo molecular design from gene expression signatures

## Download the reproducibility bundle

The complete source code, supplied datasets, and archived epoch-9 Tx2Mol checkpoint are available in the [v1.0 release](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0).

- [Download Tx2Mol-with-checkpoint.zip](https://github.com/Yaxin-Xu/Tx2Mol/releases/download/v1.0/Tx2Mol-with-checkpoint.zip) — 1.16 GB (1,157,501,538 bytes).
- [Download the file checksum manifest](https://github.com/Yaxin-Xu/Tx2Mol/releases/download/v1.0/Tx2Mol-with-checkpoint.manifest.json).

The ZIP contains GeneVAE pretraining, Tx2Mol post-training, and ten-target generation code; training/validation data and target, SciPlex3, and patient scenario inputs; and the archived Tx2Mol model, conditioning projection, tokenizer, and matching GeneVAE checkpoint.

## Generate with the included checkpoint

Extract the ZIP and follow `Tx2Mol/README.md` to install the Linux/Python 3.10/CUDA environment. Then run from the extracted `Tx2Mol/` directory:

```bash
python -m tx2mol.generate --config configs/generate_reference.json
```

No separate reference-checkpoint download is required. Fresh post-training requires the unconditioned NovoMolGen base model described in the bundled README. The executable generation example covers ten targets; the other scenario inputs retain their documented formats.

Use the explicitly attached **Tx2Mol-with-checkpoint.zip** for the complete bundle. GitHub's automatic **Source code** archives and a clone of this repository do not include that bundle.

## Verify the download

SHA256 of `Tx2Mol-with-checkpoint.zip`:

```text
e86092df70521af9040793864aff3351e3fa47a8fd8f50e447ee870684d2240b
```
