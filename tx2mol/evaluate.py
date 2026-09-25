"""Evaluate all valid molecules and export each target's highest-scoring run.

Generation calls this module automatically. Use ``python -m tx2mol.evaluate``
to apply the same calculation to an existing raw_attempts.csv on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "name": "all_valid_best_run_maximum_tanimoto",
    "generated_scope": "All valid generated molecules, without novelty filtering.",
    "reference_scope": "Canonical, deduplicated known ligands excluding canonical training SMILES.",
    "fingerprint": {"type": "Morgan", "radius": 2, "bits": 2048, "useChirality": False},
    "run_score": "Maximum over generated molecule / eligible known-ligand pairs; empty sets give zero.",
    "selection": "Highest run maximum per target; ties choose the lowest zero-based run_idx.",
    "summary": "Arithmetic mean of the selected target maxima, after run selection.",
}
RESULT_FILES = ("run_max_tanimoto.csv", "best_max_tanimoto.csv", "best_run_attempts.csv", "evaluation_summary.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LigandScorer:
    """Cache ligand fingerprints and score every valid molecule with one rule."""

    def __init__(self, source_ligands):
        molecules = {}
        for smiles in source_ligands:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("LigandScorer requires valid source SMILES")
            molecules.setdefault(Chem.MolToSmiles(mol), mol)
        self.ligands = list(molecules)
        self.fingerprints = [self.fingerprint(mol) for mol in molecules.values()]
        self.cache = {}

    @staticmethod
    def fingerprint(mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)

    def score(self, smiles):
        if smiles not in self.cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Cannot score invalid SMILES")
            similarities = DataStructs.BulkTanimotoSimilarity(self.fingerprint(mol), self.fingerprints)
            if similarities:
                best = max(range(len(similarities)), key=similarities.__getitem__)
                self.cache[smiles] = (similarities[best], self.ligands[best])
            else:
                self.cache[smiles] = (0.0, "")
        return self.cache[smiles]


def validate_attempts(raw, runs, samples_per_run, targets=None):
    """Require complete attempt records before selecting any winning group."""
    required = {"target", "run_idx", "attempt_index", "canonical_smiles", "valid"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    if raw.empty or runs < 1 or samples_per_run < 1:
        raise ValueError("Nonempty attempts and positive run/sample counts are required")
    for column in ("run_idx", "attempt_index", "valid"):
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain integer values")
        raw[column] = values.astype(int)
    if not raw.valid.isin([0, 1]).all():
        raise ValueError("valid must contain only 0 or 1")
    if raw.duplicated(["target", "run_idx", "attempt_index"]).any():
        raise ValueError("Duplicate target/run/attempt identifiers")
    for target in raw.target.unique():
        if not isinstance(target, str) or not target.strip() or any(c in target for c in ("/", "\\")):
            raise ValueError("Target names must be nonempty filename stems")
    if targets is not None and set(raw.target) != set(targets):
        raise ValueError("Attempt records do not contain exactly the configured targets")
    for smiles in raw.loc[raw.valid == 1, "canonical_smiles"].unique():
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None or mol.GetNumAtoms() < 2:
            raise ValueError("Invalid or empty canonical SMILES marked valid")
    for target, group in raw.groupby("target", sort=False):
        if set(group.run_idx) != set(range(runs)):
            raise ValueError(f"{target}: expected exactly {runs} complete runs")
        for run_idx, attempts in group.groupby("run_idx", sort=True):
            if len(attempts) != samples_per_run or set(attempts.attempt_index) != set(range(samples_per_run)):
                raise ValueError(f"{target}, run {run_idx}: incomplete attempt records")


def evaluate_frame(raw, scorers, runs, samples_per_run, targets=None):
    """Return rescored attempts, all run maxima, and the selected runs."""
    raw = raw.copy()
    validate_attempts(raw, runs, samples_per_run, targets)
    values = [scorers[row.target].score(row.canonical_smiles) if row.valid else ("", "")
              for row in raw.itertuples()]
    raw["max_tanimoto"] = [value[0] for value in values]
    raw["closest_source_ligand"] = [value[1] for value in values]
    records = []
    for (target, run_idx), group in raw.groupby(["target", "run_idx"], sort=True):
        valid = group[group.valid == 1].sort_values("attempt_index")
        witness = max(valid.to_dict("records"), key=lambda r: r["max_tanimoto"], default=None)
        records.append({"target": target, "run_idx": int(run_idx), "total_generated": len(group),
                        "valid_num": len(valid), "source_ligand_count": len(scorers[target].ligands),
                        "max_tanimoto": witness["max_tanimoto"] if witness else 0.0,
                        "generated_smiles": witness["canonical_smiles"] if witness else "",
                        "known_ligand": witness["closest_source_ligand"] if witness else "",
                        "attempt_index": witness["attempt_index"] if witness else ""})
    per_run = pd.DataFrame(records)
    best = per_run.loc[per_run.groupby("target", sort=True).max_tanimoto.idxmax()].copy()
    best["best_group_1based"] = best.run_idx + 1
    return raw, per_run, best


def export_evaluation(attempts_path, output_dir, scorers, runs, samples_per_run,
                      train_path, sources, targets=None, export_rescored=False, expected_scores=None):
    """Shared finalization for fresh generation and CPU-only reevaluation."""
    output = Path(output_dir)
    for name in RESULT_FILES + (("raw_attempts.csv",) if export_rescored else ()):
        if (output / name).exists():
            raise FileExistsError(f"{output / name}: already exists; choose a new output directory")
    raw = pd.read_csv(attempts_path, keep_default_na=False, float_precision="round_trip")
    scored, per_run, best = evaluate_frame(raw, scorers, runs, samples_per_run, targets)
    if expected_scores is not None:
        observed = {(r.target, r.run_idx): r.max_tanimoto for r in per_run.itertuples()}
        expected = {(r["target"], r["run_idx"]): r["max_tanimoto"] for r in expected_scores}
        if observed != expected:
            raise ValueError("Generation and final evaluation disagree on per-run maxima")
    selected = scored.merge(best[["target", "run_idx"]], on=["target", "run_idx"],
                            how="inner", validate="many_to_one").sort_values(["target", "run_idx", "attempt_index"])
    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "run_max_tanimoto.csv", index=False)
    best.to_csv(output / "best_max_tanimoto.csv", index=False)
    selected.to_csv(output / "best_run_attempts.csv", index=False)
    if export_rescored:
        scored.to_csv(output / "raw_attempts.csv", index=False)
    summary = {
        "status": "complete", "protocol": PROTOCOL,
        "attempts": len(raw), "targets": len(best), "runs_per_target": runs,
        "attempts_per_run": samples_per_run, "selected_attempts": len(selected),
        "mean_of_selected_target_maxima": float(best.max_tanimoto.mean()),
        "raw_attempts_sha256": sha256(attempts_path), "train_sha256": sha256(train_path),
        "source_ligand_sha256": {target: sha256(Path(sources) / f"source_{target}.csv") for target in best.target},
        "evaluator_sha256": sha256(__file__),
        "result_sha256": {name: sha256(output / name) for name in RESULT_FILES[:-1]},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion},
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(best[["target", "best_group_1based", "max_tanimoto"]].to_string(index=False))
    print("Mean of selected target maxima:", summary["mean_of_selected_target_maxima"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv.gz")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/targets/known_ligands")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--samples-per-run", type=int, default=100)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    from .generate import load_ligands, load_reference

    raw = pd.read_csv(args.attempts, keep_default_na=False)
    try:
        validate_attempts(raw, args.runs, args.samples_per_run)
        training, _, _ = load_reference(args.train)
        scorers = {target: LigandScorer(load_ligands(args.sources / f"source_{target}.csv", training))
                   for target in raw.target.unique()}
        export_evaluation(args.attempts, args.output_dir, scorers, args.runs, args.samples_per_run,
                          args.train, args.sources, export_rescored=True)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
