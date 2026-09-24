"""Portable Tx2Mol post-training and the historical GE-prefix sampling helpers.

The original objective is preserved: causal LM + cell-group bidirectional
InfoNCE. Historical tuple-style hidden-state indexing falls back to the first
token INPUT embedding with the archived NovoMolGen tensor-returning backend.
Since that token is BOS, this is not molecule-specific contrastive alignment.
The prefix-conditioned causal LM remains active. No pooling fix is applied to
the historical objective. Prefix injection uses a temporary embedding hook;
this implementation supports one training process and is not thread-safe.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import warnings

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="JSON defaults; explicitly provided CLI options override them.")
    for name, default in (("model_path", "pretrained/novomolgen"), ("data_path", "data/train.csv.gz"),
                          ("val_data_path", "data/val.csv.gz"), ("output_dir", "outputs/tx2mol"),
                          ("saved_gene_vae", "outputs/gene_vae/gene_vae.pt")):
        p.add_argument("--" + name, default=default)
    p.add_argument("--gene_order_path", help="JSON list of ordered gene identifiers, required for headerless production data.")
    p.add_argument("--csv_header", choices=("infer", "none"), default="none")
    for name, default in (("gene_num", 978), ("gene_latent_size", 64), ("num_prefix_tokens", 1),
                          ("num_cell_lines", 14), ("batch_size", 32), ("grad_accum", 1),
                          ("epochs", 20), ("max_len", 100), ("seed", 42), ("eval_epochs", 1),
                          ("num_samples", 256), ("patience", 3), ("max_eval_batches", 50)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--max_train_steps", type=int, help="Stop after this many optimizer updates; intended for smoke tests.")
    p.add_argument("--gene_hidden_sizes", nargs="+", type=int, default=[512, 256, 128])
    for name, default in (("gene_dropout", .1), ("infonce_weight", .2), ("infonce_temperature", .07),
                          ("lr", 1e-5), ("model_lr", 2e-5), ("prefix_lr", 5e-5),
                          ("warmup_ratio", .1), ("min_delta", .001), ("weight_decay", .01),
                          ("adam_epsilon", 1e-8), ("max_grad_norm", 1.0)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--use_tenfold_binary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip_generation_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Smoke mode: select checkpoint by validation LM loss, NOT the paper's composite criterion.")
    p.add_argument("--precision", choices=("no", "bf16", "fp16"), default="bf16")
    return p


def parse_args(argv=None):
    p = build_parser()
    initial, _ = p.parse_known_args(argv)
    if initial.config:
        config = json.loads(Path(initial.config).read_text(encoding="utf-8"))
        unknown = set(config) - {action.dest for action in p._actions}
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**config)
    args = p.parse_args(argv)
    if args.precision not in ("no", "bf16", "fp16") or args.csv_header not in ("infer", "none"):
        p.error("Invalid precision or csv_header in configuration")
    for key in ("epochs", "batch_size", "grad_accum", "max_len", "eval_epochs", "max_eval_batches", "num_samples"):
        if getattr(args, key) < 1:
            p.error(f"--{key} must be positive")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        p.error("--max_train_steps must be positive")
    if args.infonce_temperature <= 0 or args.max_grad_norm <= 0:
        p.error("InfoNCE temperature and gradient clipping norm must be positive")
    return args


if __name__ == "__main__" and any(arg in sys.argv[1:] for arg in ("-h", "--help")):
    build_parser().parse_args()


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .gene_vae import GeneVAE


def _get_token_embedding_layer(model: nn.Module) -> nn.Embedding:
    try:
        emb = model.get_input_embeddings()
        if isinstance(emb, nn.Embedding):
            return emb
    except (AttributeError, NotImplementedError):
        pass
    for path in (
        "model.embed_tokens", "model.model.embed_tokens", "backbone.embed_tokens",
        "transformer.wte", "transformer.embeddings.word_embeddings",
        "transformer.embeddings", "backbone.embeddings.word_embeddings", "gpt_neox.embed_in",
    ):
        cur = model
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, nn.Embedding):
            return cur
    raise ValueError("Cannot find the model's token embedding layer.")


class GEPrefixProjection(nn.Module):
    """Checkpoint-compatible projection of sampled GeneVAE z and cell encoding."""

    def __init__(self, ge_latent_size, model_hidden_size, num_prefix_tokens=1,
                 num_cell_lines=14, use_tenfold_binary=False):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_cell_lines = num_cell_lines
        self.use_tenfold_binary = use_tenfold_binary
        if use_tenfold_binary and num_cell_lines > 16:
            raise ValueError("The historical four-bit cell encoding supports at most 16 cells.")
        cell_dim = 4 if use_tenfold_binary else num_cell_lines
        self.projection = nn.Sequential(
            nn.Linear(ge_latent_size + cell_dim, model_hidden_size * 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(model_hidden_size * 2, model_hidden_size * num_prefix_tokens),
            nn.LayerNorm(model_hidden_size * num_prefix_tokens),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def _encode_cell_line(self, cell_line_idx):
        if self.use_tenfold_binary:
            return torch.stack([(cell_line_idx >> i) & 1 for i in range(4)], dim=1).float()
        return F.one_hot(cell_line_idx, num_classes=self.num_cell_lines).float()

    def forward(self, ge_latent, cell_line_idx):
        cell = self._encode_cell_line(cell_line_idx).to(ge_latent)
        projected = self.projection(torch.cat([ge_latent, cell], dim=-1))
        hidden = projected.size(-1) // self.num_prefix_tokens
        prefix = projected.view(ge_latent.size(0), self.num_prefix_tokens, hidden)
        return prefix / prefix.norm(dim=-1, keepdim=True).clamp(min=1e-6) * math.sqrt(hidden)


def compute_infonce_loss_cell_line(smiles_repr, ge_repr, cell_line_idx, temperature=0.07):
    """Sample-count-weighted, symmetric InfoNCE within each cell line."""
    smiles_repr = F.normalize(smiles_repr, p=2, dim=-1)
    ge_repr = F.normalize(ge_repr, p=2, dim=-1)
    total_loss, total_samples = 0.0, 0
    for cell in torch.unique(cell_line_idx):
        indices = torch.where(cell_line_idx == cell)[0]
        count = len(indices)
        if count < 2:
            continue
        similarity = smiles_repr[indices] @ ge_repr[indices].t() / temperature
        labels = torch.arange(count, device=smiles_repr.device)
        total_loss += (F.cross_entropy(similarity, labels) +
                       F.cross_entropy(similarity.t(), labels)) / 2 * count
        total_samples += count
    if total_samples == 0:
        return (smiles_repr - smiles_repr).mean() + (ge_repr - ge_repr).mean()
    return total_loss / total_samples


def forward_with_ge_prefix(model, gene_vae, ge_projection, input_ids, attention_mask,
                           labels, genes, cell_line_idx, token_embedding_layer,
                           compute_infonce=True, infonce_temperature=0.07):
    """Historical forward: prefix is visible to LM; its own label is ignored.

    As in the source training code, the backbone does not receive an attention
    mask. Inputs are right-padded and padded labels are -100. The released
    causal backend cannot let later padding affect earlier real tokens.
    """
    with torch.no_grad():
        ge_latent, _, _ = gene_vae(genes)
    dtype = next(ge_projection.parameters()).dtype
    ge_latent = ge_latent.to(dtype=dtype)
    if not torch.isfinite(ge_latent).all():
        raise ValueError("Non-finite GeneVAE latent vector.")
    prefix = ge_projection(ge_latent, cell_line_idx.to(input_ids.device))
    if not torch.isfinite(prefix).all():
        raise ValueError("Non-finite GE prefix.")
    # Accelerate returns projection outputs in FP32. Preserve the historical
    # explicit cast back to the backbone parameter dtype before normalization
    # in InfoNCE and concatenation with token embeddings.
    prefix = prefix.to(dtype=next(model.parameters()).dtype)
    token_embeddings = token_embedding_layer(input_ids).to(prefix.dtype)
    prefix_count = prefix.size(1)
    prefix_ids = torch.ones(input_ids.size(0), prefix_count, dtype=torch.long, device=input_ids.device)
    combined_ids = torch.cat([prefix_ids, input_ids], dim=1)
    prefix_labels = torch.full_like(prefix_ids, -100)
    combined_labels = torch.cat([prefix_labels, labels], dim=1)
    combined_embeddings = torch.cat([prefix, token_embeddings], dim=1)
    original_forward = token_embedding_layer.forward
    try:
        token_embedding_layer.forward = lambda input_ids: combined_embeddings
        outputs = model(input_ids=combined_ids, labels=combined_labels,
                        return_dict=True, output_hidden_states=True)
    finally:
        token_embedding_layer.forward = original_forward
    hidden_states = getattr(outputs, "hidden_states", None)
    last_hidden = hidden_states[-1] if hidden_states is not None else getattr(outputs, "last_hidden_state", None)
    model._tx2mol_hidden_states_container = type(hidden_states).__name__
    if last_hidden is not None and last_hidden.dim() == 3:
        smiles_repr = last_hidden[:, prefix_count, :]
        model._tx2mol_alignment_representation = "first_smiles_position_contextual_hidden_state"
    else:
        model._tx2mol_alignment_representation = "first_token_input_embedding"
        # Preserve the original fallback, but never conceal its limitation.
        if compute_infonce:
            warnings.warn("Historical hidden-state indexing did not yield a 3D layer state. "
                          "NovoMolGen returns a [batch, length, hidden] tensor; indexing it as a tuple "
                          "selects one batch row. Preserving the original InfoNCE fallback to the "
                          "first-token input embedding (BOS with the archived tokenizer), which is "
                          "not molecule-specific contextual alignment.",
                          RuntimeWarning, stacklevel=2)
        smiles_repr = token_embeddings[:, 0, :]
    infonce = None
    if compute_infonce:
        infonce = compute_infonce_loss_cell_line(smiles_repr, prefix[:, -1, :],
                                                 cell_line_idx, infonce_temperature)
    return outputs.loss, outputs.logits, infonce


def generate_samples_with_ge(model, gene_vae, ge_projection, tokenizer, genes_batch,
                             cell_line_idx_batch, token_embedding_layer, num_samples=8,
                             max_length=64, temperature=1.0, top_p=0.95, top_k=100,
                             device=torch.device("cuda"), verbose=False):
    """Return raw decoded strings, one per input condition, without filtering.

    Retains historical batch sampling/EOS behavior: finished rows continue
    sampling until all rows emit EOS together or max_length is reached. Each
    row is decoded only up to its FIRST EOS. num_samples is a legacy argument;
    the number of input rows determines the number of attempted generations.
    """
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0 or max_length < 1:
        raise ValueError("Require temperature>0, 0<top_p<=1, top_k>=0 and max_length>=1.")
    device = torch.device(device)
    model.eval()
    gene_vae.eval()
    ge_projection.eval()
    with torch.no_grad():
        z, _, _ = gene_vae(genes_batch.to(device))
        prefix_dtype = next(ge_projection.parameters()).dtype
        prefix = ge_projection(z.to(prefix_dtype), cell_line_idx_batch.to(device))
        batch_size = genes_batch.size(0)
        bos = tokenizer.bos_token_id
        if bos is None:
            input_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        else:
            input_ids = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)
        for _ in range(max_length):
            embeddings = torch.cat([prefix, token_embedding_layer(input_ids).to(prefix.dtype)], dim=1) if input_ids.size(1) else prefix
            dummy_ids = torch.full(embeddings.shape[:2], tokenizer.pad_token_id or 0,
                                   dtype=torch.long, device=device)
            original_forward = token_embedding_layer.forward
            amp = (torch.autocast("cuda", dtype=prefix.dtype)
                   if device.type == "cuda" and prefix.dtype in (torch.float16, torch.bfloat16)
                   else contextlib.nullcontext())
            try:
                token_embedding_layer.forward = lambda input_ids: embeddings
                with amp:
                    logits = model(input_ids=dummy_ids, return_dict=True).logits
            finally:
                token_embedding_layer.forward = original_forward
            next_logits = logits[:, -1, :] / temperature
            if top_k:
                k = min(top_k, next_logits.size(-1))
                remove = next_logits < torch.topk(next_logits, k)[0][..., -1, None]
                next_logits[remove] = -float("inf")
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                next_logits[remove.scatter(1, sorted_indices, remove)] = -float("inf")
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                break
        result = []
        for seq in input_ids:
            if tokenizer.eos_token_id is not None:
                positions = (seq == tokenizer.eos_token_id).nonzero()
                if positions.numel():
                    seq = seq[:positions[0].item() + 1]
            result.append(tokenizer.decode(seq, skip_special_tokens=True).replace(" ", ""))
        return result


class SMILESDataset_WithGE(Dataset):
    """CSV columns: cell line, compound ID, SMILES, then ordered gene values."""

    def __init__(self, csv_path, tokenizer, max_len=100, cell_line_to_idx=None,
                 csv_header="none", gene_num=978, gene_columns=None):
        self.df = pd.read_csv(csv_path, header="infer" if csv_header == "infer" else None)
        if self.df.shape[1] != gene_num + 3 or self.df.empty:
            raise ValueError(f"{csv_path}: expected nonempty CSV with {gene_num + 3} columns; got {self.df.shape}.")
        self.tokenizer, self.max_len = tokenizer, max_len
        if self.df.iloc[:, :3].isna().any().any():
            raise ValueError(f"{csv_path}: missing cell line, compound ID or SMILES.")
        self.cell_line_names = self.df.iloc[:, 0].astype(str).tolist()
        self.smiles_list = self.df.iloc[:, 2].astype(str).tolist()
        self.gene_data = self.df.iloc[:, 3:].to_numpy(dtype=np.float32)
        if not np.isfinite(self.gene_data).all():
            raise ValueError(f"{csv_path}: gene values contain NaN/Inf.")
        self.gene_columns = list(map(str, self.df.columns[3:])) if csv_header == "infer" else None
        if gene_columns is not None:
            if len(gene_columns) != gene_num or len(set(gene_columns)) != gene_num:
                raise ValueError("Gene order must contain exactly gene_num unique identifiers.")
            if self.gene_columns is not None and self.gene_columns != gene_columns:
                raise ValueError(f"{csv_path}: CSV gene header order differs from the supplied gene order.")
            self.gene_columns = list(gene_columns)
        self.cell_line_to_idx = (cell_line_to_idx if cell_line_to_idx is not None
                                 else {cell: i for i, cell in enumerate(sorted(set(self.cell_line_names)))})
        unknown = sorted(set(self.cell_line_names) - self.cell_line_to_idx.keys())
        if unknown:
            raise ValueError(f"{csv_path}: cell lines absent from training mapping: {unknown}")
        self.cell_line_indices = [self.cell_line_to_idx[cell] for cell in self.cell_line_names]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.smiles_list[idx], truncation=True, max_length=self.max_len,
                                 padding=False, add_special_tokens=True, return_tensors="pt")
        ids = encoded["input_ids"].squeeze(0)
        return {"input_ids": ids, "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": ids.clone(), "genes": torch.from_numpy(self.gene_data[idx]),
                "cell_line": self.cell_line_names[idx], "cell_line_idx": self.cell_line_indices[idx],
                "smiles": self.smiles_list[idx]}


def collate_fn_with_ge(batch, pad_token_id):
    result = {key: torch.nn.utils.rnn.pad_sequence([item[key] for item in batch],
               batch_first=True, padding_value=pad) for key, pad in
              (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}
    result.update(genes=torch.stack([item["genes"] for item in batch]),
                  cell_line_idx=torch.tensor([item["cell_line_idx"] for item in batch]),
                  smiles=[item["smiles"] for item in batch],
                  cell_lines=[item["cell_line"] for item in batch])
    return result


def load_gene_vae(args, device):
    vae = GeneVAE(input_size=args.gene_num, hidden_sizes=list(args.gene_hidden_sizes),
                  latent_size=args.gene_latent_size, output_size=args.gene_num,
                  activation_fn=nn.ReLU(), dropout=args.gene_dropout).to(device)
    vae.load_model(args.saved_gene_vae)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment():
    versions = {}
    for package in ("torch", "transformers", "accelerate", "numpy", "pandas", "rdkit", "flash-attn", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "command": sys.argv}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_metrics(strings, train_smiles):
    # Missing RDKit/SA scoring is an error, never a silently fabricated zero.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer
    valid = []
    for string in strings:
        try:
            mol = Chem.MolFromSmiles(string, sanitize=False)
            if mol is None:
                continue
            Chem.SanitizeMol(mol)
            canonical = Chem.MolToSmiles(mol).strip()
            if canonical and mol.GetNumAtoms() > 1:
                valid.append(canonical)
        except (ValueError, RuntimeError):
            continue
    unique = list(dict.fromkeys(valid))
    fps, properties = [], []
    for smi in valid:
        mol = Chem.MolFromSmiles(smi)
        fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
        lipinski = int(mw <= 500 and logp <= 5 and Descriptors.NumHDonors(mol) <= 5
                       and Descriptors.NumHAcceptors(mol) <= 10)
        properties.append([QED.qed(mol), sascorer.calculateScore(mol), logp, lipinski, mw])
    similarities = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    for i in range(len(fps)) for j in range(i + 1, len(fps))]
    # Historical diversity definition includes duplicates; <2 fingerprints => diversity 1.
    diversity = 1.0 - (float(np.mean(similarities)) if similarities else 0.0)
    means = np.mean(properties, axis=0).tolist() if properties else [0.0] * 5
    valid_rate = 100 * len(valid) / max(len(strings), 1)
    unique_rate = 100 * len(unique) / max(len(valid), 1)
    label_set = set(train_smiles)  # Preserve historical raw-reference novelty definition.
    novel = sum(smi not in label_set for smi in unique)
    metrics = dict(total=len(strings), valid=len(valid), valid_rate=valid_rate,
                   unique=len(unique), unique_rate=unique_rate, novel=novel,
                   novel_rate=100 * novel / max(len(unique), 1), diversity=diversity,
                   QED=means[0], SA=means[1], LogP=means[2], Lipinski=means[3], MW=means[4])
    metrics["composite_score"] = valid_rate / 100 * .35 + unique_rate / 100 * .25 + diversity * .25 + means[0] * .15
    return metrics, valid






def main(argv=None):
    args = parse_args(argv)
    from accelerate import Accelerator
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from tqdm.auto import tqdm

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    acc = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision=args.precision)
    if acc.num_processes != 1:
        raise ValueError("This embedding-hook implementation supports one process. Use one GPU.")
    if args.precision != "no" and acc.device.type != "cuda":
        raise ValueError("Use --precision no on CPU; the original NovoMolGen FlashAttention backend requires CUDA.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16. Use --precision fp16 on compatible CUDA GPUs.")
    if not args.skip_generation_eval:
        from rdkit.Contrib.SA_Score import sascorer  # fail before expensive training if unavailable
        from rdkit.Chem import QED
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists():
        raise FileExistsError(f"{output} already contains a training run; choose a fresh --output_dir.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define PAD or EOS.")
    gene_vae = load_gene_vae(args, acc.device)
    # bf16 parameter storage preserves the historical training configuration.
    # fp16 uses FP32 master parameters with Accelerate autocast/GradScaler.
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                torch_dtype=dtype, device_map=None).to(acc.device)
    hidden = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    if hidden is None:
        raise ValueError("Backbone config must define hidden_size or n_embd.")
    projection = GEPrefixProjection(args.gene_latent_size, hidden, args.num_prefix_tokens,
                                    args.num_cell_lines, args.use_tenfold_binary).to(acc.device, dtype=dtype)
    gene_columns = None
    if args.gene_order_path:
        gene_columns = json.loads(Path(args.gene_order_path).read_text(encoding="utf-8"))
    train = SMILESDataset_WithGE(args.data_path, tokenizer, args.max_len, csv_header=args.csv_header,
                                 gene_num=args.gene_num, gene_columns=gene_columns)
    if len(train.cell_line_to_idx) > args.num_cell_lines:
        raise ValueError("Training data contains more cell lines than --num_cell_lines.")
    val = SMILESDataset_WithGE(args.val_data_path, tokenizer, args.max_len, train.cell_line_to_idx,
                               args.csv_header, args.gene_num, train.gene_columns)
    if train.gene_columns is None:
        warnings.warn("Headerless data without gene_order_path: column order cannot be verified. "
                      "Provide the actual ordered gene identifiers for reproducibility.")
    collate = partial(collate_fn_with_ge, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.model_lr if args.model_lr is not None else args.lr},
        {"params": projection.parameters(), "lr": args.prefix_lr if args.prefix_lr is not None else args.lr * 5},
    ], weight_decay=args.weight_decay, eps=args.adam_epsilon)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if args.max_train_steps is not None:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)
    model, projection, optimizer, train_loader, val_loader, scheduler = acc.prepare(
        model, projection, optimizer, train_loader, val_loader, scheduler)
    embedding = _get_token_embedding_layer(acc.unwrap_model(model))
    environment = _environment()
    run_config = dict(vars(args), cell_line_to_idx=train.cell_line_to_idx,
                       gene_order_verified=args.csv_header == "infer" and train.gene_columns is not None,
                       gene_order_status=("verified_against_csv_headers" if args.csv_header == "infer" else
                                          "externally_supplied_expected_order" if train.gene_columns else "unverified_positional"),
                       gene_vae_sha256=_sha256(args.saved_gene_vae), model_hidden_size=hidden,
                       selection_criterion="validation_lm_loss" if args.skip_generation_eval else "historical_composite_score",
                       alignment_representation="not_observed_yet",
                       alignment_indexing="historical_hidden_states_minus_one_then_first_smiles_position_or_input_embedding_fallback",
                       bos_token_id=tokenizer.bos_token_id,
                       total_optimizer_steps=total_steps)
    _json(output / "run_config.json", run_config)
    _json(output / "environment.json", environment)
    _json(output / "cell_line_mapping.json", train.cell_line_to_idx)
    if train.gene_columns is not None:
        _json(output / "gene_columns.json", train.gene_columns)
    best_score, no_improve, global_step = -float("inf"), 0, 0
    history = []

    def forward(batch):
        return forward_with_ge_prefix(model, gene_vae, projection,
            batch["input_ids"], batch["attention_mask"], batch["labels"], batch["genes"],
            batch["cell_line_idx"], embedding, args.infonce_weight > 0, args.infonce_temperature)

    for epoch in range(1, args.epochs + 1):
        model.train()
        projection.train()
        gene_vae.eval()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            with acc.accumulate(model, projection):
                with acc.autocast():
                    lm, _, contrastive = forward(batch)
                    loss = lm + args.infonce_weight * contrastive if contrastive is not None else lm
                observed_alignment = model._tx2mol_alignment_representation
                if run_config["alignment_representation"] != observed_alignment:
                    run_config["alignment_representation"] = observed_alignment
                    run_config["hidden_states_container"] = model._tx2mol_hidden_states_container
                    run_config["first_token_ids_observed_in_batch"] = sorted(set(batch["input_ids"][:, 0].tolist()))
                    _json(output / "run_config.json", run_config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; stopping before optimizer update.")
                acc.backward(loss)
                if acc.sync_gradients:
                    # Unscale exactly once for fp16, then preserve separate historical clipping.
                    acc.unscale_gradients(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(projection.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach()))
                if acc.sync_gradients:
                    global_step += 1
            if global_step >= total_steps:
                break
        stop = global_step >= total_steps
        if epoch % args.eval_epochs != 0 and epoch != args.epochs and not stop:
            continue
        model.eval()
        projection.eval()
        val_losses, val_lm_losses = [], []
        with torch.no_grad(), acc.autocast():
            for batch in val_loader:
                lm, _, contrastive = forward(batch)
                total = lm + args.infonce_weight * contrastive if contrastive is not None else lm
                val_losses.append(float(total))
                val_lm_losses.append(float(lm))
                if len(val_losses) >= args.max_eval_batches:
                    break
        metrics = dict(epoch=epoch, global_step=global_step, train_loss=float(np.mean(losses)),
                       loss=float(np.mean(val_losses)), val_lm_loss=float(np.mean(val_lm_losses)),
                       selection_criterion=run_config["selection_criterion"],
                       smoke_test=bool(args.skip_generation_eval or args.max_train_steps is not None))
        if args.skip_generation_eval:
            score = -metrics["val_lm_loss"]
        else:
            generated = []
            while len(generated) < args.num_samples:
                count = min(args.batch_size, args.num_samples - len(generated), len(val))
                indices = np.random.choice(len(val), size=count, replace=False)
                genes = torch.stack([val[i]["genes"] for i in indices])
                cells = torch.tensor([val[i]["cell_line_idx"] for i in indices])
                with acc.autocast():
                    generated.extend(generate_samples_with_ge(acc.unwrap_model(model), gene_vae,
                        acc.unwrap_model(projection), tokenizer, genes, cells, embedding,
                        num_samples=count, max_length=args.max_len, device=acc.device))
            generation_metrics, valid_smiles = _generation_metrics(generated, train.smiles_list)
            metrics.update(generation_metrics)
            score = metrics["composite_score"]
            pd.DataFrame({"predict": valid_smiles}).to_csv(output / f"valid_gen_ep{epoch}.csv", index=False)
            pd.DataFrame({"raw_smiles": generated}).to_csv(output / f"valid_raw_ep{epoch}.csv", index=False)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation checkpoint selection score.")
        history.append(metrics)
        pd.DataFrame(history).to_csv(output / "training_log.csv", index=False)
        improved = score > best_score + args.min_delta
        if score > best_score:
            best_score = score
            checkpoint = output / f"best_ep{epoch}"
            checkpoint.mkdir(exist_ok=True)
            acc.unwrap_model(model).save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save(acc.unwrap_model(projection).state_dict(), checkpoint / "ge_projection.pt")
            _json(checkpoint / "args.json", run_config)
            _json(checkpoint / "cell_line_mapping.json", train.cell_line_to_idx)
            if train.gene_columns is not None:
                _json(checkpoint / "gene_columns.json", train.gene_columns)
            _json(checkpoint / "environment.json", environment)
            _json(checkpoint / "best_metrics.json", metrics)
            _json(output / "best_checkpoint.json", dict(path=checkpoint.name, epoch=epoch,
                  selection_criterion=run_config["selection_criterion"], score=score,
                  smoke_test=metrics["smoke_test"]))
            print(f"Saved checkpoint: {checkpoint}")
        no_improve = 0 if improved else no_improve + 1
        print(json.dumps(metrics))
        if stop or no_improve >= args.patience:
            break
    print(f"Finished. Checkpoint pointer: {output / 'best_checkpoint.json'}")


if __name__ == "__main__":
    main()
