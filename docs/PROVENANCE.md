# Release provenance and changes

The release was assembled from the author's running NovoMolGen/Tx2Mol project and the GeneVAE pretraining project. The original pretraining entry point is `main.py → train_gene_vae.py → GeneVAE.py`. The original post-training and ten-target generation scripts were adapted into `tx2mol.finetune` and `tx2mol.generate`; the new pretraining CLI is `tx2mol.pretrain`.

## Archived checkpoint

The reference asset contains the original epoch-9 Tx2Mol generator, projection, tokenizer and saved arguments, plus the GeneVAE actually used on the generation server. SHA256 values are listed in `assets/release_manifest.json`. Original archived argument files are retained as provenance; their old machine paths are not used as the release's runtime paths.

The same-named GeneVAE file found with the pretraining logs was numerically different from the GeneVAE deployed for generation. Their architecture agrees, but they are not the same state dictionary. Fresh training follows the supplied documented recipe; archived generation uses its matching deployed weights. The release does not claim the current pretraining log uniquely establishes the provenance of every historical weight.

## Deliberate integration fixes

- Uniform `(z, mu, reconstruction)` VAE interface connects pretraining and generation, while preserving all legacy parameter names and shapes.
- Copy hidden-layer lists before reversing the decoder order. Historical saved args may have reversed encoder widths; archived loading infers dimensions from actual weight tensors and validates the whole state dictionary.
- Explicitly read the supplied **headerless** training files, preserving the first sample. Earlier post-training code inferred a header and omitted that row.
- Read SMILES from metadata column **2** (zero-based) in the supplied three-metadata-column matrices. The historical generation script used column 1, which contains compound IDs. `--legacy_reference_column 1` can be used for a clearly labelled comparison, but is not the recommended default.
- Support gzip CSVs, strict finite numeric dimensions, matching target gene order and explicit cell-line mappings. Do not pad/truncate an incorrectly shaped signature or silently map unknown cells to index zero.
- Add stage-specific seeds, raw-attempt exports, hashes and resolved environment/configuration records. Use stable ordering for metric calculations that previously depended on Python set order.
- Validate GeneVAE/checkpoint pairing through the saved VAE hash or the archived release manifest.
- Turn missing required chemistry dependencies into explicit errors instead of fabricated zero-valued scores.
- Use the executed historical clipping value 1.0 as the actual configurable default.
- Separate smoke-test checkpoint selection by LM loss from full training's generation-based composite criterion.
- Use an explicit Accelerator autocast context around the training objective. Some BF16 reductions can differ in precision from the historical prepared-module-only execution; the loss definition is unchanged, but fresh optimization is not bitwise equivalent.

These changes improve execution and make known historical differences explicit. They mean fresh release training/metrics are not a promise of byte-identical reproduction of the historical scripts' accidental header loss, ID-column novelty calculation or unseeded samples.

## Retained limitation

The historical BOS-embedding InfoNCE fallback is preserved, not silently repaired. See `METHODS.md`. The release records what the model actually returned during training. A scientifically revised molecule-specific contrastive objective should be treated as a new experiment and separately trained checkpoint.

## Data provenance boundary

The supplied processed expression matrices, target signatures and reference ligands are packaged without numerical changes. Raw LINCS retrieval, upstream normalization and biological gene-ID alignment code were not present in the supplied training/generation entry points. The release records ordered columns and exact checksums, but does not invent absent upstream preprocessing steps.
