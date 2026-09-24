"""CPU regression checks for prefix alignment, data schema and checkpoint APIs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from tx2mol.finetune import (GEPrefixProjection, SMILESDataset_WithGE,
                            compute_infonce_loss_cell_line, forward_with_ge_prefix,
                            generate_samples_with_ge, parse_args)


class FakeVAE(nn.Module):
    def forward(self, genes):
        return genes[:, :2], genes[:, :2], genes


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 4)
        self.head = nn.Linear(4, 5)
        self.raise_error = False
        self.return_tensor = False

    def forward(self, input_ids, labels=None, **kwargs):
        if self.raise_error:
            raise RuntimeError("intentional backbone error")
        hidden = self.embedding(input_ids).cumsum(dim=1)
        self.hidden = hidden
        self.labels = labels
        logits = self.head(hidden)
        # Keep a differentiable scalar for testing forward and backward.
        loss = logits.square().mean()
        return SimpleNamespace(loss=loss, logits=logits,
                               hidden_states=hidden if self.return_tensor else (hidden,))


class TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 3

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[0, 1, 4]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return "".join({1: "C", 2: "?"}.get(int(token), "") for token in ids)


class FinetuneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_first_smiles_position_alignment_and_hook_restoration(self):
        model = TinyBackbone()
        vae = FakeVAE()
        projection = GEPrefixProjection(2, 4, num_cell_lines=2)
        ids = torch.tensor([[0, 1, 2], [0, 2, 1]])
        labels = ids.clone()
        labels[:, -1] = -100
        original_forward = model.embedding.forward
        with patch("tx2mol.finetune.compute_infonce_loss_cell_line", return_value=torch.tensor(0.0)) as scorer:
            loss, _, _ = forward_with_ge_prefix(model, vae, projection, ids, torch.ones_like(ids),
                labels, torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([0, 0]), model.embedding)
        self.assertTrue(torch.equal(scorer.call_args.args[0], model.hidden[:, 1, :]))
        self.assertFalse(torch.equal(scorer.call_args.args[0], model.hidden[:, -1, :]))
        self.assertEqual(model.labels[:, 0].tolist(), [-100, -100])
        self.assertTrue(torch.equal(model.labels[:, 1:], labels))
        self.assertEqual(model.embedding.forward, original_forward)
        loss.backward()
        self.assertIsNotNone(projection.projection[0].weight.grad)
        model.raise_error = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            forward_with_ge_prefix(model, vae, projection, ids, torch.ones_like(ids), labels,
                                   torch.randn(2, 2), torch.tensor([0, 0]), model.embedding)
        self.assertEqual(model.embedding.forward, original_forward)

    def test_archived_tensor_return_preserves_bos_embedding_fallback_and_flag(self):
        model = TinyBackbone()
        model.return_tensor = True  # The released NovoMolGen returns a tensor, not a layer tuple.
        ids = torch.tensor([[0, 1, 2], [0, 2, 1]])
        expected = model.embedding(ids)[:, 0, :]
        projection = GEPrefixProjection(2, 4, num_cell_lines=2)
        with self.assertWarnsRegex(RuntimeWarning, "not molecule-specific"):
            with patch("tx2mol.finetune.compute_infonce_loss_cell_line", return_value=torch.tensor(0.0)) as scorer:
                forward_with_ge_prefix(model, FakeVAE(), projection, ids, torch.ones_like(ids),
                    ids, torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([0, 0]), model.embedding)
        captured = scorer.call_args.args[0]
        self.assertTrue(torch.equal(captured, expected))
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertEqual(model._tx2mol_alignment_representation, "first_token_input_embedding")
        self.assertEqual(model._tx2mol_hidden_states_container, "Tensor")

    def test_infonce_excludes_singleton_cells_and_retains_gradients(self):
        smiles = torch.randn(3, 4, requires_grad=True)
        genes = torch.randn(3, 4, requires_grad=True)
        actual = compute_infonce_loss_cell_line(smiles, genes, torch.tensor([0, 0, 1]))
        expected = compute_infonce_loss_cell_line(smiles[:2], genes[:2], torch.tensor([0, 0]))
        self.assertTrue(torch.allclose(actual, expected))
        zero = compute_infonce_loss_cell_line(smiles, genes, torch.tensor([0, 1, 2]))
        self.assertEqual(float(zero), 0.0)
        zero.backward()
        self.assertIsNotNone(smiles.grad)

    def test_headerless_data_preserves_first_row_and_rejects_unknown_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "train.csv.gz"
            pd.DataFrame([["A", "drug1", "CC", 1., 2.], ["B", "drug2", "CN", 3., 4.]]).to_csv(csv, index=False, header=False)
            dataset = SMILESDataset_WithGE(csv, TinyTokenizer(), gene_num=2,
                                           gene_columns=["hsa:1", "hsa:2"])
            self.assertEqual(len(dataset), 2)
            self.assertTrue(np.array_equal(dataset.gene_data[0], [1., 2.]))
            self.assertEqual(dataset.gene_columns, ["hsa:1", "hsa:2"])
            with self.assertRaisesRegex(ValueError, "absent from training mapping"):
                SMILESDataset_WithGE(csv, TinyTokenizer(), gene_num=2, cell_line_to_idx={"A": 0})

    def test_sampling_returns_each_attempt_including_duplicates(self):
        model = TinyBackbone().eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-100)
            model.head.bias[2] = 100  # deliberately invalid string '?', repeated across rows
        projection = GEPrefixProjection(2, 4, num_cell_lines=2).eval()
        result = generate_samples_with_ge(model, FakeVAE(), projection, TinyTokenizer(),
            torch.randn(3, 2), torch.tensor([0, 0, 0]), model.embedding,
            num_samples=3, max_length=2, top_k=1, device="cpu")
        self.assertEqual(result, ["??", "??", "??"])

    def test_config_overrides_and_hidden_sizes_are_not_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"batch_size": 64, "gene_hidden_sizes": [512, 256, 128]}))
            args = parse_args(["--config", str(config), "--batch_size", "2"])
            self.assertEqual(args.batch_size, 2)
            self.assertEqual(args.gene_hidden_sizes, [512, 256, 128])


if __name__ == "__main__":
    unittest.main()
