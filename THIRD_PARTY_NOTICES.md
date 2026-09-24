# Upstream components and attribution

- The molecular backbone is [NovoMolGen](https://github.com/chandar-lab/NovoMolGen), specifically [NovoMolGen_300M_SMILES_AtomWise](https://huggingface.co/chandar-lab/NovoMolGen_300M_SMILES_AtomWise). The supplied model card declares the MIT license. The pinned base and reference model assets retain the upstream custom modeling file and model metadata.
- NovoMolGen citation: Chitsaz et al., *NovoMolGen: Rethinking Molecular Language Model Pretraining*, 2025, [arXiv:2508.13408](https://arxiv.org/abs/2508.13408).
- The GeneVAE implementation and recipe were adapted from the author's supplied `GxVAEs-main` project. Its training equations and state-dict keys are preserved; the release removes an in-place architecture-list mutation and unifies the forward interface.
- Molecular validity, Morgan/RDK fingerprints, QED, logP and synthetic-accessibility scoring use RDKit and its bundled SA_Score implementation. FlashAttention, Transformers, Accelerate and PyTorch remain external dependencies with their respective licenses.
- Dataset and target/reference-ligand files are the supplied research inputs. This release preserves their values and records file hashes; it does not create new data-source provenance or supersede any original data-use terms.

These notices describe included/dependent third-party materials and do not assign a new blanket license to the author's unpublished contributions.
