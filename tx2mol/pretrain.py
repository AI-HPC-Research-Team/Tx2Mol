"""Reproducible GeneVAE pretraining: ``python -m tx2mol.pretrain --help``."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .gene_vae import GeneVAE


DEFAULTS = {
    "data_path": "data/train.csv.gz",
    "output_dir": "outputs/gene_vae",
    "epochs": 2000,
    "batch_size": 64,
    "lr": 1e-4,
    "seed": 0,
    "hidden_sizes": [512, 256, 128],
    "latent_size": 64,
    "gene_num": 978,
    "dropout": 0.2,
    "alpha_start": 0.99,
    "alpha_end": 0.5,
    "beta": 1.0,
    "device": "auto",
    "max_steps": None,
    "metadata_columns": "auto",
    "csv_header": "auto",
    "gene_order_path": None,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gene_order(path):
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        if isinstance(values, dict):
            for key in ("gene_columns", "gene_order", "genes"):
                if key in values:
                    values = values[key]
                    break
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("Gene order manifest must be a list of gene names or {'gene_columns': [...]}")
    if len(set(values)) != len(values):
        raise ValueError("Gene order manifest contains duplicate identifiers")
    return values


def load_expression(data_path, gene_num=978, metadata_columns="auto",
                    csv_header="auto", gene_order_path=None):
    """Keep supplied gene order and values; never standardize or drop rows.

    The original pretraining files have two metadata columns. The Tx2Mol files
    have three. Headerless and headed CSV/CSV.GZ are accepted. Explicit header
    selection is available for unusual numeric gene names.
    """
    first = pd.read_csv(data_path, header=None, nrows=1)
    actual_metadata = first.shape[1] - gene_num
    if actual_metadata not in (2, 3):
        raise ValueError(
            f"Expected 2 or 3 metadata columns followed by {gene_num} genes; "
            f"found {first.shape[1]} total columns"
        )
    if metadata_columns != "auto" and int(metadata_columns) != actual_metadata:
        raise ValueError(f"Expected {metadata_columns} metadata columns, found {actual_metadata}")
    manifest = read_gene_order(gene_order_path)
    if manifest is not None and len(manifest) != gene_num:
        raise ValueError(f"Gene order manifest must contain exactly {gene_num} names")
    if csv_header == "auto":
        initial_genes = first.iloc[0, actual_metadata:]
        numeric = pd.to_numeric(initial_genes, errors="coerce")
        # NaN/inf are data errors, not a reason to silently discard a row.
        conversion_failed = initial_genes.notna() & numeric.isna()
        header_present = bool(conversion_failed.all())
        if manifest is not None and initial_genes.astype(str).tolist() == manifest:
            header_present = True
    elif csv_header in ("present", "none"):
        header_present = csv_header == "present"
    else:
        raise ValueError("csv_header must be auto, present, or none")
    frame = pd.read_csv(data_path, header=0 if header_present else None)
    if frame.empty:
        raise ValueError("Training dataset is empty")
    if frame.shape[1] != actual_metadata + gene_num:
        raise ValueError("Inconsistent number of CSV columns")
    supplied_names = first.iloc[0, actual_metadata:].astype(str).tolist() if header_present else None
    if supplied_names is not None and len(set(supplied_names)) != gene_num:
        raise ValueError("Gene columns must have unique identifiers")
    if manifest is not None and supplied_names is not None and supplied_names != manifest:
        raise ValueError("CSV gene columns differ from the manifest (including order)")
    try:
        values = frame.iloc[:, actual_metadata:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("Every gene value must be numeric; check csv_header and metadata_columns") from error
    invalid = np.argwhere(~np.isfinite(values))
    if len(invalid):
        row, column = invalid[0]
        raise ValueError(f"Non-finite expression at data row {row + 1}, gene position {column + 1}")
    names = manifest or supplied_names or [f"gene{i}" for i in range(1, gene_num + 1)]
    return torch.from_numpy(values), {
        "rows": len(values),
        "gene_num": gene_num,
        "metadata_columns": actual_metadata,
        "header_present": header_present,
        "gene_columns": names,
        "gene_identifiers_source": "manifest" if manifest else "csv_header" if supplied_names else "positional_only",
        "preprocessing": "Use supplied numeric values unchanged; no normalization, row removal, or gene reordering.",
    }


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON object; explicit CLI flags override JSON settings")
    for name in ("data_path", "output_dir", "device", "gene_order_path"):
        parser.add_argument(f"--{name}", default=argparse.SUPPRESS)
    for name in ("epochs", "batch_size", "seed", "max_steps", "gene_num", "latent_size"):
        parser.add_argument(f"--{name}", type=int, default=argparse.SUPPRESS)
    for name in ("lr", "dropout", "alpha_start", "alpha_end", "beta"):
        parser.add_argument(f"--{name}", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--metadata_columns", choices=("auto", "2", "3"), default=argparse.SUPPRESS)
    parser.add_argument("--csv_header", choices=("auto", "present", "none"), default=argparse.SUPPRESS)
    cli = vars(parser.parse_args(argv))
    config_path = cli.pop("config")
    config = dict(DEFAULTS)
    if config_path:
        with open(config_path, encoding="utf-8") as handle:
            supplied = json.load(handle)
        if not isinstance(supplied, dict):
            parser.error("--config must contain a JSON object")
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(supplied)
    config.update(cli)
    for name in ("epochs", "batch_size", "gene_num", "latent_size"):
        if not isinstance(config[name], int) or config[name] < 1:
            parser.error(f"{name} must be a positive integer")
    if config["max_steps"] is not None and (not isinstance(config["max_steps"], int) or config["max_steps"] < 1):
        parser.error("max_steps must be a positive integer when provided")
    if not 0 <= config["seed"] < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if config["lr"] <= 0 or not 0 <= config["dropout"] < 1 or config["beta"] < 0:
        parser.error("lr must be positive; dropout in [0,1); beta nonnegative")
    if not 0 <= config["alpha_end"] <= config["alpha_start"] <= 1:
        parser.error("alpha weights must satisfy 0 <= alpha_end <= alpha_start <= 1")
    config["config_path"] = config_path
    return argparse.Namespace(**config)


def train(args):
    output_dir = Path(args.output_dir)
    if any((output_dir / name).exists() for name in ("args.json", "gene_vae.pt", "loss.csv")):
        raise FileExistsError(
            f"{output_dir} already contains a pretraining run; choose a fresh --output_dir."
        )
    seed_everything(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    data_path = Path(args.data_path)
    inputs, data_details = load_expression(data_path, args.gene_num, args.metadata_columns,
                                          args.csv_header, args.gene_order_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "gene_vae.pt"
    metadata_path = output_dir / "args.json"
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(inputs), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, generator=generator, drop_last=False)
    model = GeneVAE(args.gene_num, args.hidden_sizes, args.latent_size,
                    args.gene_num, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Exactly the original linspace/constant schedule, including float32→double.
    half = args.epochs // 2
    alphas = torch.cat([torch.linspace(args.alpha_start, args.alpha_end, half),
                        args.alpha_end * torch.ones(args.epochs - half)]).double().to(device)
    metadata = {
        **vars(args),
        "device_resolved": str(device),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "data_sha256": sha256_file(data_path),
        "data": data_details,
        "architecture": {"input_size": args.gene_num, "hidden_sizes": list(args.hidden_sizes),
                         "latent_size": args.latent_size, "output_size": args.gene_num,
                         "activation": "ReLU", "dropout": args.dropout},
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        "loss": "alpha * sum_squared_error + (1-alpha) * beta * summed_KL",
        "alpha_schedule": "linear over floor(epochs/2) epochs, then constant alpha_end",
        "seed_policy": "Python, NumPy, CPU/CUDA torch and shuffle generator seeded; deterministic algorithms enabled",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
        "provenance": {
            "source_project": "GxVAEs-main",
            "entry_point": "main.py -> train_gene_vae.py -> GeneVAE.py",
            "inspected_source_sha256": {
                "main.py": "8a5901b65fabe0c7e845c47f14b0e73b40e6bad3e945181da76198d6b77c65aa",
                "train_gene_vae.py": "c7450f3bb007267f7a170c704b4cdf5663d9cc8a86d43fe297ba8ac321dfb828",
                "GeneVAE.py": "9a067c0982f85c4fb6c33f3636f08006aef0cd516ac3bb5589f19e8c0cfba7d9",
                "utils.py": "e7badbf7aadac1ee94aca1adcbe9f3e6a129b084336bf234b362e17dcc1268d6",
            },
            "changes": ["Copied hidden widths before decoder reversal", "Explicit seed and device selection",
                        "Strict CSV validation and metadata handling", "Portable checkpoint and run metadata"],
            "reproduction_scope": "Reimplements the supplied training procedure. Newly trained weights are not claimed to be identical to archived study checkpoints.",
        },
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("pretrain.py", "gene_vae.py")},
        "weights_path": str(weights_path),
    }
    if args.config_path:
        metadata["config_sha256"] = sha256_file(args.config_path)
    if args.gene_order_path:
        metadata["gene_order_sha256"] = sha256_file(args.gene_order_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "gene_order.json").write_text(json.dumps(data_details["gene_columns"], indent=2) + "\n", encoding="utf-8")
    steps = 0
    completed_epochs = 0
    with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "updates", "samples_seen", "alpha", "joint_per_sample", "mse_per_gene", "kl_per_latent", "complete_epoch"])
        for epoch in range(args.epochs):
            model.train()
            totals = np.zeros(3, dtype=np.float64)
            seen = 0
            batches_done = 0
            for (genes,) in loader:
                genes = genes.to(device)
                _, _, reconstruction = model(genes)
                losses = model.joint_loss(reconstruction, genes, alphas[epoch], args.beta)
                if not all(torch.isfinite(loss).item() for loss in losses):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, update {steps + 1}")
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                optimizer.step()
                totals += [loss.item() for loss in losses]
                seen += len(genes)
                batches_done += 1
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    break
            means = [totals[0] / seen, totals[1] / (seen * args.gene_num), totals[2] / (seen * args.latent_size)]
            complete_epoch = batches_done == len(loader)
            completed_epochs += int(complete_epoch)
            writer.writerow([epoch + 1, steps, seen, alphas[epoch].item(), *means, complete_epoch])
            handle.flush()
            print(f"epoch={epoch + 1}/{args.epochs} updates={steps} joint={means[0]:.6f} mse={means[1]:.6f} kl={means[2]:.6f}", flush=True)
            if args.max_steps is not None and steps >= args.max_steps:
                break
    model.save_model(weights_path)
    metadata.update({"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "updates_completed": steps, "full_epochs_completed": completed_epochs,
                     "last_epoch": epoch + 1,
                     "stopped_at_max_steps": args.max_steps is not None and steps >= args.max_steps,
                     "weights_sha256": sha256_file(weights_path)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {weights_path}", flush=True)
    return model


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
