"""Focused checks for denominator, gene-order and aggregation correctness."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tx2mol.generate import (aggregate_runs, infer_gene_vae_architecture, load_gene_order, load_reference,
                             load_target, parse_args, resolve_checkpoint, score_attempts,
                             sha256, verify_gene_vae_provenance)


class GenerateInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        return path

    def test_target_rejects_wrong_width_nonfinite_and_wrong_order(self):
        genes = [f"g{i}" for i in range(978)]
        values = list(range(978))
        path = self.csv("target.csv", [genes, values])
        loaded, order = load_target(path, genes)
        self.assertEqual(loaded, values)
        self.assertEqual(order, genes)
        with self.assertRaisesRegex(ValueError, "gene order"):
            load_target(path, genes[::-1])
        for invalid in (values[:-1], values[:-1] + ["nan"], values[:-1] + ["text"]):
            with self.subTest(invalid=invalid[-1]):
                path = self.csv("target.csv", [genes, invalid])
                with self.assertRaises(ValueError):
                    load_target(path)

    def test_first_data_row_and_manifest(self):
        genes = [f"hsa:{i}" for i in range(978)]
        path = self.csv("target.csv", [genes, [1] * 978, [2] * 978])
        loaded, _ = load_target(path)
        self.assertEqual(loaded, [1.0] * 978)
        manifest = self.root / "gene_order.json"
        manifest.write_text(json.dumps({"gene_ids": genes}), encoding="utf-8")
        self.assertEqual(load_gene_order(manifest), genes)

    def test_headerless_smiles_column_and_no_dropped_first_row(self):
        path = self.csv("train.csv", [["MCF7", "BRD-1", "CCO", 1], ["A549", "BRD-2", "CCN", 2]])
        smiles, cells, meta = load_reference(path)
        self.assertEqual(smiles, {"CCO", "CCN"})
        self.assertEqual(cells, ["A549", "MCF7"])
        self.assertEqual(meta["smiles_column"], 2)
        self.assertEqual(meta["rows"], 2)
        legacy, _, meta = load_reference(path, legacy_column=1)
        self.assertEqual(legacy, set())
        self.assertEqual(meta["smiles_column"], 1)

    def test_gzip_reference_and_target_csv(self):
        reference = self.root / "train.csv.gz"
        with gzip.open(reference, "wt", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([["MCF7", "BRD-1", "CCO", 1], ["A549", "BRD-2", "CCN", 2]])
        smiles, cells, meta = load_reference(reference)
        self.assertEqual(smiles, {"CCO", "CCN"})
        self.assertEqual(cells, ["A549", "MCF7"])
        self.assertEqual(meta["rows"], 2)
        target = self.root / "AKT1.csv.gz"
        genes = [f"g{i}" for i in range(978)]
        with gzip.open(target, "wt", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([genes, [0.5] * 978])
        values, order = load_target(target, genes)
        self.assertEqual(values, [0.5] * 978)
        self.assertEqual(order, genes)

    def test_vae_architecture_comes_from_weights(self):
        weights = {key: SimpleNamespace(shape=shape) for key, shape in {
            "encoder.encoding.0.weight": (512, 978),
            "encoder.encoding.3.weight": (256, 512),
            "encoder.encoding.6.weight": (128, 256),
            "encoder.encoding_to_mu.weight": (64, 128),
            "encoder.encoding_to_logvar.weight": (64, 128),
        }.items()}
        architecture, state = infer_gene_vae_architecture({"state_dict": weights})
        self.assertEqual(architecture["hidden_sizes"], [512, 256, 128])
        self.assertEqual(architecture["input_size"], 978)
        self.assertEqual(architecture["latent_size"], 64)
        self.assertIs(state, weights)
        weights["encoder.encoding_to_logvar.weight"].shape = (32, 128)
        with self.assertRaisesRegex(ValueError, "latent size"):
            infer_gene_vae_architecture(weights)

    def test_config_override_and_checkpoint_pointer(self):
        config = self.root / "generate.json"
        config.write_text(json.dumps({"num_samples": 12, "seed": 9}), encoding="utf-8")
        args = parse_args(["--config", str(config), "--num_samples", "1"])
        self.assertEqual(args.num_samples, 1)
        self.assertEqual(args.seed, 9)
        (self.root / "best_ep1").mkdir()
        (self.root / "best_checkpoint.json").write_text(json.dumps({"path": "best_ep1"}), encoding="utf-8")
        self.assertEqual(resolve_checkpoint(self.root), (self.root / "best_ep1").resolve())

    def test_vae_hash_matching_checks_checkpoint_and_archived_manifest(self):
        vae = self.root / "gene_vae.pt"
        vae.write_bytes(b"correct VAE weights")
        expected = sha256(vae)
        matching = verify_gene_vae_provenance(self.root, vae, {"gene_vae_sha256": expected})
        self.assertTrue(matching["gene_vae_match_verified"])
        model = self.root / "pytorch_model.bin"
        model.write_bytes(b"archived molecular weights")
        manifest = self.root / "release_manifest.json"
        manifest.write_text(json.dumps({"reference": {"files": {
            "checkpoints/reference/tx2mol/pytorch_model.bin": {"sha256": sha256(model)},
            "checkpoints/reference/gene_vae.pt": expected,
        }}}), encoding="utf-8")
        matching = verify_gene_vae_provenance(self.root, vae, {}, manifest)
        self.assertTrue(matching["gene_vae_match_verified"])
        self.assertEqual(matching["verified_model_weights_format"], "pytorch_bin")
        vae.write_bytes(b"a different VAE with the same architecture")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_gene_vae_provenance(self.root, vae, {"gene_vae_sha256": expected})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_gene_vae_provenance(self.root, vae, {}, manifest)


class GenerateMetricTests(unittest.TestCase):
    def test_raw_invalid_duplicates_and_denominators(self):
        scores, attempts = score_attempts(["CCO", "OCC", "CCN", "bad smiles", "C"], {"CCO"}, ["CCN"])
        self.assertEqual(len(attempts), 5)
        self.assertEqual([row["valid"] for row in attempts], [1, 1, 1, 0, 0])
        self.assertEqual([row["duplicate_in_run"] for row in attempts], [0, 1, 0, 0, 0])
        self.assertEqual((scores["valid_num"], scores["unique_num"], scores["novel_num"]), (3, 2, 1))
        self.assertAlmostEqual(scores["valid_rate"], 60)
        self.assertAlmostEqual(scores["unique_rate"], 200 / 3)
        self.assertAlmostEqual(scores["novel_rate"], 50)
        self.assertEqual(scores["max_tanimoto"], 1)
        self.assertEqual(attempts[2]["closest_source_ligand"], "CCN")
        self.assertGreater(attempts[0]["max_tanimoto"], 0)
        self.assertEqual(attempts[3]["max_tanimoto"], "")
        self.assertNotIn("mean_max_tanimoto", scores)
        self.assertNotIn("avg_tanimoto", scores)

    def test_non_novel_molecules_determine_primary_maximum(self):
        scores, attempts = score_attempts(["CCN", "CCO"], {"CCN"}, ["CCN", "NCC"])
        self.assertEqual(attempts[0]["novel"], 0)
        self.assertEqual(attempts[0]["max_tanimoto"], 1.0)
        self.assertEqual(scores["max_tanimoto"], 1.0)
        self.assertEqual(scores["source_ligand_count"], 1)

    def test_diversity_retains_duplicate_attempts(self):
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        a = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CCO"), 2, nBits=2048)
        b = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("CCN"), 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(a, b)
        scores, _ = score_attempts(["CCO", "CCO", "CCN"], set(), ["CCN"])
        self.assertAlmostEqual(scores["diversity"], 1 - (1 + 2 * sim) / 3)

    def test_aggregation_selects_maximum_and_averages_other_diagnostics(self):
        rows = [{"target": "AKT1", "cell_line": "MCF7", "run_idx": i, "seed": 42 + i,
                 "valid_rate": value, "max_tanimoto": score}
                for i, (value, score) in enumerate(((0, 0.4), (100, 1.0)))]
        output = aggregate_runs(rows)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["valid_rate_mean"], 50)
        self.assertAlmostEqual(output[0]["valid_rate_std"], 50 * 2**0.5)
        self.assertNotIn("seed_mean", output[0])
        self.assertEqual(output[0]["max_tanimoto"], 1.0)
        self.assertEqual(output[0]["best_run_idx"], 1)
        self.assertNotIn("max_tanimoto_mean", output[0])


if __name__ == "__main__":
    unittest.main()
