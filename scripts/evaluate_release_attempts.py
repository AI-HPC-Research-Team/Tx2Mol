"""Select each target's highest run maximum using the recovered paper evaluator."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_records(raw, runs, samples_per_run):
    """Require all attempts before selecting any run; never fill missing records."""
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
    for smiles in raw.loc[raw.valid == 1, "canonical_smiles"].unique():
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None or mol.GetNumAtoms() < 2:
            raise ValueError("Invalid or empty canonical SMILES marked valid")
    records = []
    for target, target_rows in raw.groupby("target", sort=False):
        if set(target_rows.run_idx) != set(range(runs)):
            raise ValueError(f"{target}: expected exactly {runs} complete runs")
        for run_idx, group in target_rows.groupby("run_idx", sort=True):
            if len(group) != samples_per_run or set(group.attempt_index) != set(range(samples_per_run)):
                raise ValueError(f"{target}, run {run_idx}: incomplete attempt records")
            valid = group.sort_values("attempt_index").loc[lambda x: x.valid == 1, "canonical_smiles"].tolist()
            records.append({"protein_name": target, "run_idx": int(run_idx),
                            "total_generated": len(group), "valid_smiles": repr(valid)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv.gz")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/targets/known_ligands")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "scripts/evaluate_gxvaes_protocol.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--samples-per-run", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    raw = pd.read_csv(args.attempts, keep_default_na=False)
    try:
        records = prepare_records(raw, args.runs, args.samples_per_run)
    except ValueError as error:
        parser.error(str(error))
    source_paths = {target: args.sources / f"source_{target}.csv" for target in raw.target.unique()}
    for path in (args.train, args.evaluator, *source_paths.values()):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    adapter = args.output_dir / "Tx2Mol/generated_molecules"
    adapter.mkdir(parents=True)
    pd.DataFrame(records).to_csv(adapter / "all_runs_statistics.csv", index=False)
    evaluation = args.output_dir / "evaluation"
    command = [sys.executable, str(args.evaluator.resolve()), "--train", str(args.train.resolve()),
               "--sources", str(args.sources.resolve()), "--output-dir", str(evaluation.resolve()), str(adapter.resolve())]
    with (args.output_dir / "evaluator.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    best = pd.read_csv(evaluation / "best_of_10_max_tanimoto_gxvaes_protocol.csv", float_precision="round_trip")
    selected = best[["protein_name", "run_idx", "max_tanimoto_gxvaes"]].rename(
        columns={"protein_name": "target", "max_tanimoto_gxvaes": "max_tanimoto"})
    selected["best_group_1based"] = selected.run_idx + 1
    selected.to_csv(args.output_dir / "best_max_tanimoto.csv", index=False)
    winning_attempts = raw.merge(selected[["target", "run_idx"]], on=["target", "run_idx"],
                                how="inner", validate="many_to_one").sort_values(["target", "run_idx", "attempt_index"])
    winning_attempts = winning_attempts.rename(columns={
        "max_tanimoto": "generation_novel_max_tanimoto",
        "closest_source_ligand": "generation_novel_closest_source_ligand",
    })
    winning_attempts.to_csv(args.output_dir / "best_run_attempts.csv", index=False)
    provenance = {
        "protocol": "All valid generated molecules; known ligands canonicalized, deduplicated and filtered against canonical training SMILES from column 2; Morgan radius 2, 2048 bits, default useChirality=False; highest run maximum per target.",
        "selection": "Lowest run_idx breaks ties; all attempts in each selected run are exported.",
        "attempts": len(raw), "targets": len(selected), "runs_per_target": args.runs,
        "attempts_per_run": args.samples_per_run,
        "mean_of_selected_target_maxima": float(selected.max_tanimoto.mean()),
        "evaluator_sha256": sha256(args.evaluator),
        "adapter_sha256": sha256(Path(__file__)),
        "raw_attempts_sha256": sha256(args.attempts),
        "train_sha256": sha256(args.train),
        "source_ligand_sha256": {target: sha256(path) for target, path in source_paths.items()},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion},
        "command": command,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(selected.to_string(index=False))
    print("Mean of selected target maxima:", provenance["mean_of_selected_target_maxima"])


if __name__ == "__main__":
    main()
