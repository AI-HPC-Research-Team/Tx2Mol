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

from tx2mol.evaluate import LigandScorer, evaluate_frame, validate_attempts


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
                    validate_attempts(invalid.copy(), 2, 2)
        with self.assertRaisesRegex(ValueError, "configured targets"):
            validate_attempts(raw, 2, 2, targets=["A", "B", "C"])

    def test_all_invalid_run_is_retained(self):
        raw = self.rows().assign(valid=0, canonical_smiles="")
        scored, per_run, best = evaluate_frame(raw, {t: LigandScorer([]) for t in ("A", "B")}, 2, 2)
        self.assertEqual(len(per_run), 4)
        self.assertTrue((per_run.total_generated == 2).all())
        self.assertTrue((per_run.max_tanimoto == 0).all())
        self.assertTrue((best.run_idx == 0).all())
        self.assertEqual(len(scored), len(raw))

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
            command = [sys.executable, "-m", "tx2mol.evaluate",
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
            provenance = json.loads((root / "out/evaluation_summary.json").read_text())
            self.assertAlmostEqual(provenance["mean_of_selected_target_maxima"], (1 + expected_b) / 2)
            winning = pd.read_csv(root / "out/best_run_attempts.csv")
            self.assertEqual(len(winning), 4)
            self.assertEqual(winning.valid.sum(), 3)
            self.assertEqual(set(winning[winning.target == "A"].run_idx), {1})
            self.assertIn("max_tanimoto", winning.columns)
            self.assertEqual(winning[winning.target == "A"].max_tanimoto.max(), 1.0)
            self.assertEqual(provenance["attempts"], 8)
            before = (root / "out/best_max_tanimoto.csv").read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)
            self.assertEqual((root / "out/best_max_tanimoto.csv").read_bytes(), before)

    def test_generation_automatically_exports_same_protocol(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from tx2mol.generate import main as generate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genes = [f"g{i}" for i in range(978)]
            for name in ("A", "B"):
                pd.DataFrame([[0] * 978], columns=genes).to_csv(root / f"{name}.csv", index=False)
                (root / f"source_{name}.csv").write_text("CCO\nCCN\nNCC\n")
            pd.DataFrame([["MCF7", "training-id", "OCC"] + [0] * 978]).to_csv(root / "train.csv", header=False, index=False)
            pd.DataFrame([["MCF7", "validation-id", "CCN"] + [0] * 978]).to_csv(root / "val.csv", header=False, index=False)
            (root / "vae.pt").write_bytes(b"model-loading-is-mocked")
            model = (None, None, None, None, None, torch.device("cpu"), {"cell_line_to_idx": {"MCF7": 0}})
            batches = [["CCO", "invalid"], ["CCN", "NCC"], ["CCN", "CCO"], ["CCC", "CCO"]]
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(__version__="mocked")}), patch(
                "tx2mol.generate.load_models", return_value=model
            ), patch(
                "tx2mol.finetune.generate_samples_with_ge", side_effect=batches
            ):
                generate(["--model_dir", str(root), "--saved_gene_vae", str(root / "vae.pt"),
                          "--target_dir", str(root), "--source_ligands_dir", str(root),
                          "--train_data_path", str(root / "train.csv"), "--val_data_path", str(root / "val.csv"),
                          "--targets", "A", "B", "--num_runs", "2", "--num_samples", "2", "--batch_size", "2",
                          "--device", "cpu", "--output_dir", str(root / "out")])
            best = pd.read_csv(root / "out/best_max_tanimoto.csv").set_index("target")
            self.assertEqual(best.loc["A", "run_idx"], 1)
            self.assertEqual(best.loc["B", "run_idx"], 0)
            self.assertTrue((best.max_tanimoto == 1.0).all())
            raw = pd.read_csv(root / "out/raw_attempts.csv")
            self.assertEqual(len(raw), 8)
            self.assertTrue((raw.loc[raw.canonical_smiles == "CCN", "novel"] == 0).all())
            self.assertTrue((raw.loc[raw.canonical_smiles == "CCN", "max_tanimoto"] == 1.0).all())
            self.assertEqual(len(pd.read_csv(root / "out/best_run_attempts.csv")), 4)
            aggregate = pd.read_csv(root / "out/aggregate_metrics.csv")
            self.assertNotIn("max_tanimoto_mean", aggregate.columns)
            self.assertTrue((aggregate.max_tanimoto == 1.0).all())
            metadata = json.loads((root / "out/metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["evaluation"]["mean_of_selected_target_maxima"], 1.0)


if __name__ == "__main__":
    unittest.main()
