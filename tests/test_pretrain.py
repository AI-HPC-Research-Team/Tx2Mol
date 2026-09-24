"""Run with ``python -m unittest discover -s tests -p test_pretrain.py``."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tx2mol.gene_vae import GeneVAE, load_gene_vae
from tx2mol.pretrain import load_expression, parse_args, sha256_file, train


class GeneVAEContracts(unittest.TestCase):
    def test_archived_state_dict_keys_and_stochastic_interface(self):
        widths = [512, 256, 128]
        model = GeneVAE(hidden_sizes=widths).eval()
        self.assertEqual(widths, [512, 256, 128])
        expected = {
            "encoder.encoding.0.weight": (512, 978),
            "encoder.encoding.3.weight": (256, 512),
            "encoder.encoding.6.weight": (128, 256),
            "encoder.encoding_to_mu.weight": (64, 128),
            "encoder.encoding_to_logvar.weight": (64, 128),
            "decoder.decoding.0.weight": (128, 64),
            "decoder.decoding.3.weight": (256, 128),
            "decoder.decoding.6.weight": (512, 256),
            "decoder.decoding.9.weight": (978, 512),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(model.state_dict()[name].shape), shape)
        self.assertEqual(len(model.state_dict()), 2 * len(expected))
        inputs = torch.zeros(2, 978)
        torch.manual_seed(17)
        first = model(inputs)
        second = model(inputs)
        self.assertEqual([tuple(value.shape) for value in first], [(2, 64), (2, 64), (2, 978)])
        self.assertFalse(torch.equal(first[0], second[0]))
        torch.testing.assert_close(first[1], second[1])
        loss, reconstruction, kl = model.joint_loss(second[2], inputs, 0.7, 2)
        self.assertEqual(loss.dtype, torch.float64)
        torch.testing.assert_close(reconstruction, ((second[2] - inputs) ** 2).sum().double())
        expected_kl = -0.5 * (1 + model.logvar.double() - model.mu.double().square() - model.logvar.double().exp()).sum()
        torch.testing.assert_close(kl, expected_kl)
        torch.testing.assert_close(loss, 0.7 * reconstruction + 0.3 * 2 * kl)

    def test_headerless_and_headed_input_and_bad_values(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            values = np.arange(978, dtype=np.float32).reshape(1, -1) / 100
            for metadata in (2, 3):
                source = base / f"input{metadata}.csv.gz"
                frame = pd.concat([pd.DataFrame([["metadata"] * metadata]), pd.DataFrame(values)], axis=1)
                frame.to_csv(source, index=False, header=False)
                loaded, details = load_expression(source)
                np.testing.assert_array_equal(loaded.numpy(), values)
                self.assertEqual(details["metadata_columns"], metadata)
                self.assertFalse(details["header_present"])
            names = [f"hsa:{i}" for i in range(978)]
            manifest = base / "order.json"
            manifest.write_text(json.dumps(names), encoding="utf-8")
            frame.columns = ["cell_line", "inchi", "smiles"] + names
            source = base / "headed.csv"
            frame.to_csv(source, index=False)
            loaded, details = load_expression(source, gene_order_path=manifest)
            self.assertEqual(details["gene_columns"], names)
            frame.iloc[0, 3] = np.inf
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_expression(source)
            frame.iloc[0, 3] = "broken"
            frame.to_csv(source, index=False, header=False)
            with self.assertRaisesRegex(ValueError, "numeric"):
                load_expression(source)

    def test_cli_overrides_and_reproducible_one_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.csv"
            data = pd.concat([pd.DataFrame([["MCF7", "ID", "CCO"]] * 3),
                              pd.DataFrame(np.random.default_rng(9).normal(size=(3, 978)))], axis=1)
            data.to_csv(source, header=False, index=False)
            config = base / "config.json"
            config.write_text(json.dumps({"data_path": str(source), "epochs": 2, "batch_size": 3,
                                          "seed": 41, "hidden_sizes": [16, 8], "latent_size": 4}), encoding="utf-8")
            saved = []
            for index in range(2):
                output = base / f"run{index}"
                args = parse_args(["--config", str(config), "--output_dir", str(output), "--seed", "7",
                                   "--max_steps", "1", "--batch_size", "2", "--device", "cpu"])
                self.assertEqual(args.seed, 7)
                train(args)
                metadata = json.loads((output / "args.json").read_text())
                self.assertEqual(metadata["updates_completed"], 1)
                self.assertEqual(metadata["full_epochs_completed"], 0)
                self.assertEqual(metadata["status"], "completed")
                model = load_gene_vae(output / "gene_vae.pt", hidden_sizes=[16, 8], latent_size=4)
                saved.append(model.state_dict())
                checkpoint_hash = sha256_file(output / "gene_vae.pt")
                with self.assertRaisesRegex(FileExistsError, "fresh --output_dir"):
                    train(args)
                self.assertEqual(sha256_file(output / "gene_vae.pt"), checkpoint_hash)
            for name in saved[0]:
                torch.testing.assert_close(saved[0][name], saved[1][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
