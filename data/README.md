# Test inputs and known ligands by scenario

Each scenario has its own `test/` and `known_ligands/` directories directly under `data/`:

```text
data/
  targets/
    test/
    known_ligands/
  sciplex3/
    test/
    known_ligands/
  patient/
    test/
    known_ligands/
      alternate_collection/
```

The shared subLINCS train/validation/test splits, expected gene-order file and small training examples remain directly under `data/`. These shared files are separate from the three application test scenarios.

## Targets

`targets/test/` contains ten unchanged target signatures: AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3 and TP53. Their CSV headers identify 978 ordered genes. `targets/known_ligands/` contains the corresponding ten `source_TARGET.csv` files with a `SMILES` header.

These are the inputs consumed by `configs/generate.json` and `configs/generate_reference.json`.

## SciPlex3

`sciplex3/test/` contains the unchanged A549, K562 and MCF7 `*_normalized_final.csv` matrices used by the original generation scripts. Each has 144 data rows: 143 perturbation rows and one control row. The first 11 columns are metadata, including the paired perturbing drug's `SMILES`; the following 978 columns are expression values. Original dose and time information are retained.

The original SciPlex3 generator obtains the reference molecule from each test row's `SMILES`, not from the disease ligand directory. `sciplex3/known_ligands/source_CELL.csv` explicitly exports those 143 non-control references per cell line. Every record retains its original zero-based `test_row`, all 11 metadata columns and `smiles_parse_status`. There is no deduplication, canonicalization, sign reversal or additional filtering. A missing or invalid SMILES remains a paired record and is marked accordingly.

To join references back to expression profiles, use `test_row` as the positional index of the source CSV's data rows, excluding its header. Controls remain in the test matrix but are excluded from the known-ligand export, matching the original loader's control exclusion.

## Patient/disease signatures

`patient/test/` contains the 13 unchanged `test_DISEASE.csv` expression files used by the supplied patient generation script. These are disease-level processed signatures, with a 978-gene header, not newly constructed clinical records.

The default `patient/known_ligands/source_DISEASE.csv` files are exact copies of the original generator's `CREED/new_ligands` collection. They are headerless, with three columns: SMILES, drug identifier and drug name. Twelve files exist in that collection; **gastric has no default ligand file**. This absence is explicitly recorded in `assets/data_manifest.json`.

The separate `patient/known_ligands/alternate_collection/` directory preserves the 13 files from `CREED/ligands`, including gastric. This is a different collection and is not automatically merged with or substituted for the default. Neither collection is newly labelled as approved or exhaustive by this release.

## Verification and execution scope

`assets/data_manifest.json` records all 64 scenario CSV files, their hashes, sizes, schemas, original source locations and any export operation. Run `python scripts/validate_data.py` to check these files and the SciPlex3 row-level pairing.

The packaged command-line generation example is still the ten-target workflow. SciPlex3's row-oriented metadata-plus-expression matrices must not be passed to the target-only loader; patient files and cell-line conditioning also require their scenario-specific protocol. This update packages the actual inputs and references without silently changing their scientific meaning or claiming that a new scenario runner has been implemented.
