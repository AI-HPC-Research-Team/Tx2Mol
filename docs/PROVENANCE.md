# Release provenance

## Code and checkpoints

The release combines the author's GeneVAE pretraining and Tx2Mol projects. The original pretraining entry point was `main.py → train_gene_vae.py → GeneVAE.py`; the maintained commands are `tx2mol.pretrain`, `tx2mol.finetune`, and `tx2mol.generate`. Backbone attribution is in [Third-party notices](../THIRD_PARTY_NOTICES.md).

The reference asset contains the original epoch-9 Tx2Mol model, projection, tokenizer, and the GeneVAE used for generation. Checkpoint hashes are in the [weight manifest](../assets/release_manifest.json). The same-named GeneVAE accompanying the pretraining logs has different weights; it must not replace the paired generation checkpoint. Those logs do not uniquely establish the provenance of every historical weight.

## Integration changes

- Preserve VAE parameter names and shapes, infer archived dimensions from weights, and check checkpoint/VAE pairing.
- Read headerless data without losing the first row; use SMILES column 2 (zero-based), replacing the old generator's compound-ID column.
- Validate dimensions, gene order, cell labels, and chemistry dependencies; support gzip inputs and record seeds, settings, hashes, and raw outputs.
- Use stable metric ordering, the executed clipping norm of 1.0, and explicit autocast. Fresh optimization is not guaranteed to be bitwise equivalent to historical runs.

The historical BOS-embedding InfoNCE fallback remains as described in [Methods](METHODS.md#historical-infonce-limitation). The independent compound–phenotype compatibility predictor is outside this three-stage release.

## Evaluation provenance

The paper used the separate [historical evaluator](../scripts/evaluate_gxvaes_protocol.py), which scores all valid molecules. It is retained unchanged for verification. The maintained generator integrates this protocol through `tx2mol.evaluate` and exports results automatically. Archived inline metrics using only novel molecules must be reevaluated for comparison.

[Evaluation examples](../examples/paper_protocol/) contain all 100 historical valid-SMILES lists, complete fresh sampling datasets, expected scores, and checksums. Historical invalid strings were not retained and have not been reconstructed. See [Validation](VALIDATION.md) for the numerical checks.

## Data and archived records

The processed expression matrices, signatures, and retained ligands preserve their supplied numerical values. Patient data cover the 12 diseases in the updated heatmap. The [data guide](../data/README.md) and [manifest](../assets/data_manifest.json) describe current inputs. Raw LINCS retrieval, upstream normalization, and biological gene-ID alignment code were not available in the supplied entry points.

The original combined ZIP and [packaging records](https://github.com/Yaxin-Xu/Tx2Mol/tree/11f69aa0657033a8b8af421581131c96de519d7f/assets) remain historical snapshots. Clone the current repository for the maintained pipeline; [Release assets](https://github.com/Yaxin-Xu/Tx2Mol/releases/tag/v1.0) supply the compatible weights.
