"""Regression checks for the recovered all-valid, best-of-ten evaluation."""
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from scripts.evaluate_release_attempts import prepare_records


ROOT = Path(__file__).resolve().parents[1]


class PaperEvaluationTests(unittest.TestCase):
    def rows(self):
        rows = []
        for target in ("A", "B"):
            for run in range(2):
                for attempt in range(2):
                    smiles = "CCN" if target == "A" and run == 1 else "CCO"
                    valid = int(not (target == "A" and run == 1 and attempt == 1))
                    rows.append(dict(target=target, run_idx=run, attempt_index=attempt,
                                     raw_smiles=smiles if valid else "invalid", canonical_smiles=smiles if valid else "",
                                     valid=valid, novel=0, max_tanimoto="", closest_source_ligand=""))
        return pd.DataFrame(rows)

    def test_original_evaluator_is_preserved_byte_for_byte(self):
        digest = hashlib.sha256((ROOT / "scripts/evaluate_gxvaes_protocol.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "1c03c9dc21dae605db1a1920157e249788472d2487cae8cb85b91a822b6b5470")

    def test_incomplete_or_malformed_inputs_fail_before_selection(self):
        raw = self.rows()
        cases = [raw.iloc[:-1], pd.concat([raw, raw.iloc[:1]]), raw.iloc[:0],
                 raw.assign(valid=2), raw.assign(valid=0.5), raw.assign(run_idx=0.5),
                 raw.assign(canonical_smiles=""), raw.assign(target="../A"),
                 raw.drop(columns="valid"), raw.loc[raw.run_idx == 0]]
        for invalid in cases:
            with self.subTest(columns=invalid.columns.tolist(), rows=len(invalid)):
                with self.assertRaises(ValueError):
                    prepare_records(invalid.copy(), 2, 2)

    def test_all_invalid_run_is_retained(self):
        raw = self.rows().assign(valid=0, canonical_smiles="")
        records = prepare_records(raw, 2, 2)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(row["total_generated"] == 2 and row["valid_smiles"] == "[]" for row in records))

    def test_cli_all_valid_scope_training_exclusion_and_whole_winning_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.rows()
            # Reversed input order must not change the lowest-run tie break.
            raw.iloc[::-1].to_csv(root / "attempts.csv", index=False)
            with gzip.open(root / "train.csv.gz", "wt", encoding="utf-8") as stream:
                stream.write("MCF7,compound-id,OCC,0\n")
            (root / "source_A.csv").write_text("CCO\nCCN\nNCC\n", encoding="utf-8")
            (root / "source_B.csv").write_text("CCO\nCCC\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/evaluate_release_attempts.py"),
                       "--attempts", str(root / "attempts.csv"), "--train", str(root / "train.csv.gz"),
                       "--sources", str(root), "--output-dir", str(root / "out"),
                       "--runs", "2", "--samples-per-run", "2"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            best = pd.read_csv(root / "out/best_max_tanimoto.csv").set_index("target")
            fp = lambda s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
            expected_b = DataStructs.TanimotoSimilarity(fp("CCO"), fp("CCC"))
            self.assertEqual(best.loc["A", "run_idx"], 1)
            self.assertEqual(best.loc["A", "max_tanimoto"], 1.0)
            self.assertEqual(best.loc["B", "run_idx"], 0)
            self.assertAlmostEqual(best.loc["B", "max_tanimoto"], expected_b)
            self.assertLess(expected_b, 1.0)
            provenance = json.loads((root / "out/provenance.json").read_text())
            self.assertAlmostEqual(provenance["mean_of_selected_target_maxima"], (1 + expected_b) / 2)
            winning = pd.read_csv(root / "out/best_run_attempts.csv")
            self.assertEqual(len(winning), 4)
            self.assertEqual(winning.valid.sum(), 3)
            self.assertEqual(set(winning[winning.target == "A"].run_idx), {1})
            self.assertIn("generation_novel_max_tanimoto", winning.columns)
            self.assertNotIn("max_tanimoto", winning.columns)
            self.assertEqual(provenance["attempts"], 8)
            before = (root / "out/best_max_tanimoto.csv").read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)
            self.assertEqual((root / "out/best_max_tanimoto.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
