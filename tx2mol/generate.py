"""Generate phenotype-guided molecules and automatically evaluate all runs.

Run ``python -m tx2mol.generate --help`` for portable input paths. Configuration
files are JSON objects; explicit command-line arguments override their values.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import re
import statistics
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path


TARGETS = ["AKT1", "AKT2", "AURKB", "CTSK", "EGFR", "HDAC1", "MTOR", "PIK3CA", "SMAD3", "TP53"]


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    initial, _ = bootstrap.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument("--model_dir", default="outputs/tx2mol")
    parser.add_argument("--saved_gene_vae", default="outputs/gene_vae/gene_vae.pt")
    parser.add_argument("--target_dir", default="data/targets/test")
    parser.add_argument("--source_ligands_dir", default="data/targets/known_ligands")
    parser.add_argument("--train_data_path", default="data/train.csv.gz")
    parser.add_argument("--val_data_path", default="data/val.csv.gz")
    parser.add_argument("--gene_order_path", default=None,
                        help="Optional JSON gene-ID list, or one-row CSV, in target order")
    parser.add_argument("--output_dir", default="outputs/generated")
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--cell_line", default="MCF7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum sampled tokens, excluding initial BOS/prefix")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--legacy_reference_column", type=int, default=None,
                        help="Explicit zero-based train/val SMILES column; 1 reproduces the historical indexing error")
    if initial.config:
        with open(initial.config, encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            parser.error("--config must contain a JSON object")
        unknown = set(config) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for key in ("num_runs", "num_samples", "batch_size", "max_length"):
        if not isinstance(getattr(args, key), int) or getattr(args, key) < 1:
            parser.error(f"{key} must be a positive integer")
    if not isinstance(args.seed, int) or not 0 <= args.seed < 2**32:
        parser.error("seed must be an integer in [0, 2**32)")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Require temperature > 0, 0 < top_p <= 1, and top_k >= 0")
    if args.legacy_reference_column is not None and args.legacy_reference_column < 0:
        parser.error("legacy_reference_column must be nonnegative")
    if not isinstance(args.targets, list) or not args.targets or len(set(args.targets)) != len(args.targets):
        parser.error("targets must be a nonempty list of distinct names")
    if any(not isinstance(t, str) or not t or Path(t).name != t or "/" in t or "\\" in t for t in args.targets):
        parser.error("targets must be file stems, without directory components")
    if args.dtype not in ("auto", "bfloat16", "float16", "float32"):
        parser.error("Unsupported dtype")
    return args


def open_csv(path):
    """Open a plain or gzip-compressed CSV as UTF-8 text."""
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    return opener(path, "rt", newline="", encoding="utf-8-sig")


def read_rows(path):
    with open_csv(path) as stream:
        return [row for row in csv.reader(stream) if any(value.strip() for value in row)]


def load_gene_order(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        order = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(order, dict):
            order = next((order[key] for key in ("gene_order", "gene_ids", "genes") if key in order), None)
    else:
        rows = read_rows(path)
        order = rows[0] if len(rows) == 1 else [row[0] for row in rows if len(row) == 1]
    if not isinstance(order, list) or len(order) != 978:
        raise ValueError(f"{path}: gene-order manifest must contain exactly 978 IDs")
    order = [str(item).strip() for item in order]
    if not all(order) or len(set(order)) != 978:
        raise ValueError(f"{path}: gene IDs must be nonempty and unique")
    return order


def load_target(path, expected_order=None):
    """Read the first ordered 978-gene signature without padding/rescaling."""
    rows = read_rows(path)
    if len(rows) < 2 or any(len(row) != 978 for row in rows[:2]):
        raise ValueError(f"{path}: expected gene IDs and a first data row with exactly 978 columns each")
    order = [item.strip() for item in rows[0]]
    if not all(order) or len(set(order)) != 978:
        raise ValueError(f"{path}: target gene IDs must be nonempty and unique")
    if expected_order is not None and order != list(expected_order):
        raise ValueError(f"{path}: gene order does not match the supplied manifest or preceding target")
    try:
        values = [float(value) for value in rows[1]]
    except ValueError as error:
        raise ValueError(f"{path}: all 978 expression values must be numeric") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: expression values must be finite")
    return values, order


def canonical_smiles(smiles, minimum_atoms=1):
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol).strip()
        return canonical if canonical and mol.GetNumAtoms() >= minimum_atoms else None
    except (ValueError, RuntimeError):
        return None


def load_reference(path, legacy_column=None):
    """Read headerless [cell, drug, SMILES, genes...] or a named SMILES column."""
    canonical, cells, count = set(), set(), 0
    with open_csv(path) as stream:
        reader = (row for row in csv.reader(stream) if any(value.strip() for value in row))
        first = next(reader, None)
        if first is None:
            raise ValueError(f"{path}: empty reference dataset")
        header = [value.strip().lower() for value in first]
        named_smiles = "smiles" in header
        column = legacy_column if legacy_column is not None else (header.index("smiles") if named_smiles else 2)
        for row in reader if named_smiles else chain([first], reader):
            if len(row) <= column:
                raise ValueError(f"{path}: reference SMILES column {column} is missing")
            value = canonical_smiles(row[column])
            if value is not None:
                canonical.add(value)
            cells.add(row[0].strip())
            count += 1
    if count == 0:
        raise ValueError(f"{path}: no reference data rows")
    if not canonical and legacy_column is None:
        raise ValueError(f"{path}: no valid reference SMILES found in column {column}")
    return canonical, sorted(cells), {"path": str(Path(path)), "smiles_column": column,
                              "header": named_smiles, "rows": count,
                              "canonical_molecules": len(canonical)}


def load_ligands(path, train_smiles):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path}: empty ligand file")
    header = [value.strip().lower() for value in rows[0]]
    column = header.index("smiles") if "smiles" in header else 0
    data = rows[1:] if "smiles" in header else rows
    if any(len(row) <= column for row in data):
        raise ValueError(f"{path}: malformed ligand row")
    canonical = [s for row in data if (s := canonical_smiles(row[column])) is not None]
    return list(dict.fromkeys(s for s in canonical if s not in train_smiles))


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_checkpoint(model_dir):
    directory = Path(model_dir).expanduser().resolve()
    pointer = directory / "best_checkpoint.json"
    if pointer.is_file():
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        value = metadata.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{pointer}: missing checkpoint path")
        directory = Path(value) if Path(value).is_absolute() else directory / value
        directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    return directory


def infer_gene_vae_architecture(weights):
    """Infer layer widths from saved tensors; archived args may be reversed.

    The historical decoder reversed a caller-owned hidden_sizes list, so its
    serialized args are not reliable evidence of the encoder architecture.
    Dropout has no effect in eval mode; its presence fixes module numbering.
    Strict state_dict loading subsequently validates every decoder/bias tensor.
    """
    if not isinstance(weights, dict):
        raise ValueError("GeneVAE checkpoint must be a state_dict")
    if "state_dict" in weights:
        weights = weights["state_dict"]
    if not isinstance(weights, dict):
        raise ValueError("GeneVAE checkpoint state_dict must be a mapping")
    layers = []
    for key, value in weights.items():
        match = re.fullmatch(r"encoder\.encoding\.(\d+)\.weight", key)
        if match:
            if len(value.shape) != 2:
                raise ValueError(f"Invalid GeneVAE matrix: {key}")
            layers.append((int(match.group(1)), tuple(value.shape)))
    layers.sort()
    if not layers or layers[0][0] != 0:
        raise ValueError("GeneVAE checkpoint is missing encoder linear layers")
    indices = [index for index, _ in layers]
    if indices == [3 * i for i in range(len(layers))]:
        dropout = 0.2
    elif indices == [2 * i for i in range(len(layers))]:
        dropout = 0.0
    else:
        raise ValueError(f"Unrecognized GeneVAE layer numbering: {indices}")
    input_size = int(layers[0][1][1])
    hidden_sizes = [int(shape[0]) for _, shape in layers]
    if input_size != 978 or any(width < 1 for width in hidden_sizes):
        raise ValueError("GeneVAE must accept exactly 978 genes and have positive hidden widths")
    for i, (_, shape) in enumerate(layers[1:], start=1):
        if shape[1] != hidden_sizes[i - 1]:
            raise ValueError("Inconsistent GeneVAE encoder dimensions")
    expected_head = (64, hidden_sizes[-1])
    for key in ("encoder.encoding_to_mu.weight", "encoder.encoding_to_logvar.weight"):
        if key not in weights or tuple(weights[key].shape) != expected_head:
            raise ValueError(f"{key}: expected shape {expected_head}; study latent size is 64")
    return {"input_size": input_size, "hidden_sizes": hidden_sizes, "latent_size": 64,
            "output_size": 978, "dropout": dropout}, weights


def verify_gene_vae_provenance(checkpoint, vae_path, settings, manifest_path=None):
    """Reject a different VAE when checkpoint provenance specifies its hash."""
    checkpoint = Path(checkpoint)
    actual = sha256(vae_path)
    result = {"gene_vae_sha256": actual, "gene_vae_match_verified": False,
              "gene_vae_matching_source": "unverified: checkpoint does not record its conditioning GeneVAE hash"}
    expected = settings.get("gene_vae_sha256")
    source = "checkpoint args.json:gene_vae_sha256" if expected else None
    if not expected:
        manifest_path = (Path(manifest_path) if manifest_path is not None else
                         Path(__file__).resolve().parents[1] / "assets" / "release_manifest.json")
        model_bin = checkpoint / "pytorch_model.bin"
        if manifest_path.is_file() and model_bin.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("reference", {}).get("files", {})
            def entry_hash(key):
                entry = files.get(key)
                return entry.get("sha256") if isinstance(entry, dict) else entry
            model_expected = entry_hash("checkpoints/reference/tx2mol/pytorch_model.bin")
            vae_expected = entry_hash("checkpoints/reference/gene_vae.pt")
            if model_expected and vae_expected:
                model_actual = sha256(model_bin)
                result["reference_model_bin_sha256"] = model_actual
                if model_actual == model_expected:
                    expected = vae_expected
                    source = f"{manifest_path}: reference model binary hash match"
                    # If both formats exist, load the binary that was verified.
                    result["verified_model_weights_format"] = "pytorch_bin"
    if expected:
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"Invalid GeneVAE SHA256 in {source}")
        if actual.lower() != expected.lower():
            raise ValueError(f"GeneVAE checkpoint hash mismatch: {source} expects {expected}, got {actual}")
        result.update(gene_vae_match_verified=True, gene_vae_matching_source=source)
    return result


def load_models(args, checkpoint, train_cells):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .finetune import GEPrefixProjection, _get_token_embedding_layer
    from .gene_vae import GeneVAE

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)
    settings = json.loads((checkpoint / "args.json").read_text(encoding="utf-8"))
    vae_provenance = verify_gene_vae_provenance(checkpoint, args.saved_gene_vae, settings)
    vae_architecture, vae_weights = infer_gene_vae_architecture(
        torch.load(args.saved_gene_vae, map_location="cpu", weights_only=True))
    if int(settings.get("gene_latent_size", 64)) != vae_architecture["latent_size"]:
        raise ValueError("Molecular checkpoint and GeneVAE latent dimensions disagree")
    gene_vae = GeneVAE(**vae_architecture)
    gene_vae.load_state_dict(vae_weights, strict=True)
    gene_vae = gene_vae.to(device).eval()
    gene_vae.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a PAD or EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_options = {"use_safetensors": False} if vae_provenance.get("verified_model_weights_format") == "pytorch_bin" else {}
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True,
                                               local_files_only=True, torch_dtype=dtype, **model_options).to(device).eval()
    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    if hidden_size is None:
        raise ValueError("Cannot determine the molecular model hidden size")
    mapping_path = next((checkpoint / name for name in ("cell_line_mapping.json", "cell_line_to_idx.json")
                         if (checkpoint / name).is_file()), None)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path else settings.get("cell_line_to_idx")
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    if mapping is None:
        mapping = {cell: index for index, cell in enumerate(train_cells)}
        mapping_source = "sorted training-data cell lines (fallback)"
    else:
        mapping_source = str(mapping_path) if mapping_path else "checkpoint args.json"
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Invalid checkpoint cell-line mapping")
    num_cells = int(settings.get("num_cell_lines", len(mapping)))
    if any(type(i) is not int or not 0 <= i < num_cells for i in mapping.values()):
        raise ValueError("Invalid checkpoint cell-line mapping")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Checkpoint cell-line indices must be unique")
    if args.cell_line not in mapping:
        raise ValueError(f"Unknown cell line {args.cell_line!r}; available: {sorted(mapping)}")
    projection = GEPrefixProjection(
        ge_latent_size=int(settings.get("gene_latent_size", 64)), model_hidden_size=hidden_size,
        num_prefix_tokens=int(settings.get("num_prefix_tokens", 1)), num_cell_lines=num_cells,
        use_tenfold_binary=bool(settings.get("use_tenfold_binary", False)),
    ).to(device=device, dtype=dtype)
    projection.load_state_dict(torch.load(checkpoint / "ge_projection.pt", map_location=device, weights_only=True))
    projection.eval()
    details = {"device": str(device), "dtype": dtype_name, "cell_line_to_idx": mapping,
               "cell_mapping_source": mapping_source, "checkpoint_args": settings,
               **vae_provenance,
               "gene_vae_architecture": vae_architecture,
               "gene_vae_architecture_source": "checkpoint encoder tensor shapes; decoder validated by strict state_dict loading",
               "training_gene_order_verified": bool(settings.get("gene_order_verified", False))}
    return model, tokenizer, projection, gene_vae, _get_token_embedding_layer(model), device, details


def score_attempts(raw_smiles, reference_smiles, source_ligands):
    """All-valid maximum Tanimoto plus validity, novelty and property metrics."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, Descriptors, QED
    from rdkit.Contrib.SA_Score import sascorer
    from .evaluate import LigandScorer

    scorer = source_ligands if isinstance(source_ligands, LigandScorer) else LigandScorer(source_ligands)

    attempts, valid, seen = [], [], set()
    for index, raw in enumerate(raw_smiles):
        canonical = canonical_smiles(raw, minimum_atoms=2)
        duplicate = canonical is not None and canonical in seen
        novel = canonical is not None and canonical not in reference_smiles
        attempts.append({"attempt_index": index, "raw_smiles": raw, "canonical_smiles": canonical or "",
                         "valid": int(canonical is not None), "duplicate_in_run": int(duplicate),
                         "novel": int(novel)})
        if canonical is not None:
            valid.append(canonical)
            seen.add(canonical)
    unique = list(dict.fromkeys(valid))
    novel = [s for s in unique if s not in reference_smiles]
    molecules = {s: Chem.MolFromSmiles(s) for s in unique}
    morgan = {s: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for s, mol in molecules.items()}
    internal = [DataStructs.TanimotoSimilarity(morgan[left], morgan[right])
                for i, left in enumerate(valid) for right in valid[i + 1:]]
    path_fps = {s: Chem.RDKFingerprint(molecules[s]) for s in unique}
    suffix_diversities = [statistics.mean(1.0 - DataStructs.TanimotoSimilarity(path_fps[left], path_fps[right])
                                         for right in unique[i + 1:])
                         for i, left in enumerate(unique[:-1])]
    molecular_maxima = {smiles: scorer.score(smiles) for smiles in unique}
    for row in attempts:
        score, ligand = molecular_maxima.get(row["canonical_smiles"], ("", ""))
        row.update(max_tanimoto=score, closest_source_ligand=ligand)
    properties = {key: [] for key in ("qed", "sa", "logp", "lipinski", "mw")}
    for smiles in novel:
        mol = molecules[smiles]
        mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
        properties["qed"].append(float(QED.qed(mol)))
        properties["sa"].append(float(sascorer.calculateScore(mol)))
        properties["logp"].append(float(logp))
        properties["mw"].append(float(mw))
        properties["lipinski"].append(int(mw <= 500 and logp <= 5 and Descriptors.NumHDonors(mol) <= 5
                                          and Descriptors.NumHAcceptors(mol) <= 10))
    result = {"total_generated": len(raw_smiles), "valid_num": len(valid), "unique_num": len(unique),
              "novel_num": len(novel), "source_ligand_count": len(scorer.ligands),
              "valid_rate": 100.0 * len(valid) / max(len(raw_smiles), 1),
              "unique_rate": 100.0 * len(unique) / max(len(valid), 1),
              "novel_rate": 100.0 * len(novel) / max(len(unique), 1),
              "diversity": 1.0 - (statistics.mean(internal) if internal else 0.0),
              "intdivp": statistics.mean(suffix_diversities) if suffix_diversities else 0.0,
              "max_tanimoto": max((value[0] for value in molecular_maxima.values()), default=0.0),
              "tanimoto_pair_count": len(valid) * len(scorer.ligands)}
    result.update({key: statistics.mean(values) if values else 0.0 for key, values in properties.items()})
    return result, attempts


def aggregate_runs(rows):
    output = []
    for target in dict.fromkeys(row["target"] for row in rows):
        runs = [row for row in rows if row["target"] == target]
        summary = {"target": target, "cell_line": runs[0]["cell_line"], "num_runs": len(runs)}
        if "max_tanimoto" in runs[0]:
            best = min(runs, key=lambda row: (-row["max_tanimoto"], row["run_idx"]))
            summary.update(max_tanimoto=best["max_tanimoto"], best_run_idx=best["run_idx"],
                           best_group_1based=best["run_idx"] + 1)
        for key in runs[0]:
            if key in ("target", "cell_line", "run_idx", "seed", "cell_line_idx", "max_tanimoto"):
                continue
            values = [float(row[key]) for row in runs]
            summary[key + "_mean"] = statistics.mean(values)
            summary[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import rdkit
    import torch
    import transformers
    from rdkit import RDLogger
    from .finetune import generate_samples_with_ge
    from .evaluate import LigandScorer, PROTOCOL, RESULT_FILES, export_evaluation

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    checkpoint = resolve_checkpoint(args.model_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("raw_attempts.csv", "run_metrics.csv", "metadata.json") + RESULT_FILES):
        raise FileExistsError(f"{output}: generation results already exist; choose a fresh output_dir")
    train, train_cells, train_meta = load_reference(args.train_data_path, args.legacy_reference_column)
    validation, _, val_meta = load_reference(args.val_data_path, args.legacy_reference_column)
    scoring_train = train if args.legacy_reference_column is None else load_reference(args.train_data_path)[0]
    manifest_path = Path(args.gene_order_path) if args.gene_order_path else None
    checkpoint_order_path = checkpoint / "gene_columns.json"
    if manifest_path is None and checkpoint_order_path.is_file():
        manifest_path = checkpoint_order_path
    expected_order = load_gene_order(manifest_path) if manifest_path else None
    if checkpoint_order_path.is_file() and expected_order != load_gene_order(checkpoint_order_path):
        raise ValueError("Configured gene-order manifest disagrees with checkpoint gene_columns.json")
    target_values, ligands, inputs = {}, {}, [Path(args.saved_gene_vae), Path(args.train_data_path), Path(args.val_data_path)]
    if manifest_path:
        inputs.append(manifest_path)
    if args.config:
        inputs.append(Path(args.config))
    for target in args.targets:
        target_path = Path(args.target_dir) / f"{target}.csv"
        ligand_path = Path(args.source_ligands_dir) / f"source_{target}.csv"
        values, order = load_target(target_path, expected_order)
        if expected_order is None:
            expected_order = order
        target_values[target] = np.array(values, dtype=np.float32)
        if not np.isfinite(target_values[target]).all():
            raise ValueError(f"{target_path}: values overflow float32")
        ligands[target] = LigandScorer(load_ligands(ligand_path, scoring_train))
        inputs.extend([target_path, ligand_path])
    seed_everything(args.seed)
    model, tokenizer, projection, gene_vae, embedding, device, model_details = load_models(args, checkpoint, train_cells)
    inputs.extend(path for path in checkpoint.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    inputs.extend(Path(__file__).with_name(name) for name in ("generate.py", "evaluate.py", "finetune.py", "gene_vae.py"))
    metadata = {
        "status": "running", "metric_schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args), "checkpoint": str(checkpoint), **model_details,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "rdkit": rdkit.__version__, "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed_policy": "Each target/run resets Python, NumPy, Torch and CUDA to (seed + run_idx) mod 2**32; run_idx starts at 0.",
        "determinism": "cuDNN deterministic enabled, benchmark disabled; CUDA/FlashAttention results may depend on hardware/software/batch size.",
        "gene_order_manifest": str(manifest_path) if manifest_path else None,
        "target_gene_order_check": "All target headers match the manifest, when supplied, and one another. This does not independently verify upstream training gene IDs.",
        "gene_order": expected_order, "expression_transform": "none; first numeric data row used as supplied (historical protocol)",
        "references": {"train": train_meta, "validation": val_meta},
        "protocol": {
            "aggregation": "Maximum Tanimoto across runs per target, keeping the entire winning run; other diagnostics use per-run means and sample standard deviations (ddof=1).",
            "maximum_tanimoto": PROTOCOL,
            "validity": "RDKit sanitizes SMILES; canonical nonempty molecules must have at least two atoms.",
            "rates": "Percent: valid/attempted, unique/valid, novel/unique. Zero denominator gives zero.",
            "novelty": "Canonical SMILES absent from both training and validation sets.",
            "reference_column": "Named SMILES column, otherwise zero-based column 2; explicit legacy override recorded in config.",
            "source_ligands": "Canonical deduplicated source ligands absent from canonical TRAINING SMILES; legacy_reference_column does not alter scoring references.",
            "properties": "QED, SA, logP, Lipinski compliance (0/1), MW averaged over novel unique molecules; empty set gives zero.",
            "similarity": "Morgan radius 2, 2048 bits, useChirality=False; maximum over ALL VALID generated/source pairs, without a novelty filter; empty sets give zero.",
            "diversity": "1 minus mean Morgan Tanimoto over unordered pairs of valid attempts including duplicates; fewer than two gives 1 (historical convention).",
            "intdivp": "Historical RDKFingerprint distance, mean of suffix-wise means over unique valid molecules; deterministic first-occurrence order; fewer than two gives zero.",
            "latent": "Sampled GeneVAE z, including in evaluation mode.",
        },
        "sha256": {str(path): sha256(path) for path in dict.fromkeys(inputs)},
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "resolved_config.json", vars(args))
    raw_fields = ["target", "cell_line", "cell_line_idx", "run_idx", "seed", "attempt_index", "raw_smiles",
                  "canonical_smiles", "valid", "duplicate_in_run", "novel", "max_tanimoto", "closest_source_ligand"]
    all_runs = []
    with open(output / "raw_attempts.csv", "w", newline="", encoding="utf-8") as raw_stream:
        writer = csv.DictWriter(raw_stream, fieldnames=raw_fields)
        writer.writeheader()
        for target in args.targets:
            for run_idx in range(args.num_runs):
                seed = (args.seed + run_idx) % 2**32
                seed_everything(seed)
                cell_idx = model_details["cell_line_to_idx"][args.cell_line]
                context = {"target": target, "cell_line": args.cell_line, "cell_line_idx": cell_idx,
                           "run_idx": run_idx, "seed": seed}
                raw = []
                for start in range(0, args.num_samples, args.batch_size):
                    count = min(args.batch_size, args.num_samples - start)
                    genes = torch.from_numpy(target_values[target]).unsqueeze(0).repeat(count, 1).to(device)
                    cells = torch.full((count,), cell_idx, dtype=torch.long, device=device)
                    sampled = generate_samples_with_ge(
                        model=model, gene_vae=gene_vae, ge_projection=projection, tokenizer=tokenizer,
                        genes_batch=genes, cell_line_idx_batch=cells, token_embedding_layer=embedding,
                        num_samples=count, max_length=args.max_length, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k, device=device, verbose=False,
                    )
                    if len(sampled) != count or any(not isinstance(item, str) for item in sampled):
                        raise RuntimeError("Sampler must return exactly one raw SMILES string per requested attempt")
                    raw.extend(sampled)
                metrics, attempts = score_attempts(raw, train | validation, ligands[target])
                writer.writerows({**context, **attempt} for attempt in attempts)
                raw_stream.flush()
                all_runs.append({**context, **metrics})
                write_csv(output / "run_metrics.csv", all_runs)
                write_csv(output / "aggregate_metrics.csv", aggregate_runs(all_runs))
                print(f"{target} run {run_idx + 1}/{args.num_runs}: {metrics['valid_num']}/{len(raw)} valid, "
                      f"{metrics['novel_num']} novel; max Tanimoto={metrics['max_tanimoto']:.4f}", flush=True)
    metadata.update(status="evaluating", total_attempts=len(args.targets) * args.num_runs * args.num_samples)
    write_json(output / "metadata.json", metadata)
    evaluation = export_evaluation(output / "raw_attempts.csv", output, ligands, args.num_runs, args.num_samples,
                                   args.train_data_path, args.source_ligands_dir, targets=args.targets,
                                   expected_scores=all_runs)
    metadata.update(status="complete", evaluation=evaluation, completed_utc=datetime.now(timezone.utc).isoformat(),
                    total_attempts=len(args.targets) * args.num_runs * args.num_samples)
    write_json(output / "metadata.json", metadata)
    print(f"Saved all attempts, run scores, selected maxima and complete winning groups to {output}")


if __name__ == "__main__":
    main()
