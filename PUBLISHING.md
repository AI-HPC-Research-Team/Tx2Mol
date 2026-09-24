# Publishing the prepared repository

## Combined source and checkpoint ZIP

`Tx2Mol-with-checkpoint.zip` includes the source, data, archived epoch-9 Tx2Mol checkpoint, conditioning projection, tokenizer, and matching GeneVAE. Publish this combined ZIP as a GitHub Release asset for users who want to generate without a separate reference-weight download. After extraction and environment installation, they can run `python -m tx2mol.generate --config configs/generate_reference.json` from `Tx2Mol/`.

The bundled `checkpoints/reference/` remains excluded from ordinary Git commits by `.gitignore`. Fresh post-training still requires `novomolgen-300m-base.tar.gz`. The separate reference archive remains supported for source-only checkouts. Preserve `assets/release_manifest.json` so the individual checkpoint files can be verified against their recorded hashes.

## Source repository and separate assets

The repository folder and the two companion weight archives must be published together for both workflows to work. All ordinary repository files are below GitHub's 100 MiB single-file limit; the largest dataset is compressed to approximately 19 MB. Large weight archives belong in Release assets.

From this repository directory:

```bash
git init
git add .
git commit -m "Release reproducible GeneVAE and Tx2Mol pipeline"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

After authenticating GitHub CLI, upload both companion archives:

```bash
gh release create v1.0 \
  ../release-assets/novomolgen-300m-base.tar.gz \
  ../release-assets/tx2mol-reference-checkpoints.tar.gz \
  --repo OWNER/REPOSITORY \
  --title "Tx2Mol reproducibility release v1.0" \
  --notes "Three-stage implementation, complete supplied data, and checksum-pinned model assets. See README for reproduction commands."
```

The same operation can be performed through the GitHub website. Replace `OWNER/REPOSITORY` in the README download examples with the published address. Do not change either asset filename without updating the manifest/installer.

Neither repository publication nor GitHub authentication is performed by the local preparation scripts. Source code/data archives can be unpacked and committed on another machine; the accompanying weight assets are uploaded separately.
