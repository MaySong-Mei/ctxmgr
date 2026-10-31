"""Download and verify immutable official Qwen3.5 BF16 model snapshots.

The manifest's ``sha256`` is a hash of a canonical file inventory, not a hash
of one GGUF file. No model is loaded and no GPU is used by this module.
Run this utility on an allocated compute node when using Bouchet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

MODEL_SIZES = ("0.8B", "2B", "4B", "9B", "27B")
MODEL_REVISIONS = {
    "0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}
BACKEND = "vllm"
SERVER_VERSION = "0.30.0"
PRECISION = "bfloat16"
MODEL_SHA256_KIND = "hf-snapshot-tree-sha256"
INVENTORY_ALGORITHM = "sha256(json.dumps(files,sort_keys=True,separators=(',',':')))"
DOWNLOAD_PATTERNS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "*.safetensors",
    "model.safetensors.index.json",
)
_ARTIFACT_SUFFIXES = {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"}


def model_identity(size: str) -> tuple[str, str, str]:
    """Return ``(served_alias, Hugging_Face_repository, immutable_revision)``."""
    if size not in MODEL_REVISIONS:
        raise ValueError(f"Expected one of {MODEL_SIZES}, got {size!r}")
    return f"Qwen3.5-{size}-BF16", f"Qwen/Qwen3.5-{size}", MODEL_REVISIONS[size]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tree_hash(files: list[dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", name))
        and name not in {".", ".."}
        and not name.startswith(".")
    )


def _snapshot_path(raw_path: Any, repository: str, revision: str) -> Path:
    if not isinstance(raw_path, str) or any(char in raw_path for char in "\t\n\r\x00"):
        raise ValueError("Model path must be a plain absolute filesystem path")
    path = Path(raw_path)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("Model path must be an existing absolute directory")
    # A normal HF cache snapshot contains symlinks to its own repository's
    # blobs. Resolve enclosing cache-directory aliases, but keep this layout
    # invariant so a manifest cannot redirect a weight to an unrelated file.
    path = path.resolve(strict=True)
    if (
        path.name != revision
        or path.parent.name != "snapshots"
        or path.parent.parent.name != "models--" + repository.replace("/", "--")
    ):
        raise ValueError("Model path is not the pinned repository's HF cache snapshot")
    return path


def _safe_artifact(snapshot: Path, name: str) -> Path:
    if not _valid_name(name):
        raise ValueError(f"Invalid model artifact filename: {name!r}")
    artifact = snapshot / name
    if not artifact.is_file():
        raise ValueError(f"Missing model artifact: {name}")
    resolved = artifact.resolve(strict=True)
    blobs = snapshot.parent.parent / "blobs"
    if resolved.parent not in {snapshot, blobs}:
        raise ValueError(f"Model artifact points outside its snapshot/repository blobs: {name}")
    # Both materialized cache files and the usual HF blob symlinks are valid.
    return artifact


def _artifact_names(snapshot: Path) -> list[str]:
    names = []
    for item in snapshot.iterdir():
        if item.suffix in _ARTIFACT_SUFFIXES:
            _safe_artifact(snapshot, item.name)
            names.append(item.name)
        elif item.suffix in {".bin", ".gguf", ".pt", ".pth"}:
            raise ValueError(f"Unexpected alternative weight file in serving snapshot: {item.name}")
    return sorted(names)


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= 100 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        raw_header = handle.read(length)
        if len(raw_header) != length:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        header = json.loads(raw_header)
    if not isinstance(header, dict):
        # These checks validate serialized documents, not Python argument types.
        raise ValueError(f"Invalid safetensors header object: {path.name}")  # noqa: TRY004
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"Safetensors file has no tensors: {path.name}")
    data_size = path.stat().st_size - 8 - length
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"Invalid tensor metadata: {path.name}:{name}")  # noqa: TRY004
        dtype = tensor.get("dtype")
        if dtype not in {"BF16", "F32", "F64", "I64", "I32", "I16", "I8", "U8", "BOOL"}:
            raise ValueError(f"Unsupported or quantized weight dtype {dtype!r}: {path.name}:{name}")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise ValueError(f"Invalid tensor data offsets: {path.name}:{name}")
    return tensors


def _weight_structure(snapshot: Path) -> dict[str, Any]:
    config = _json(_safe_artifact(snapshot, "config.json"))
    if not isinstance(config, dict) or config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("Expected an official Qwen3.5 architecture config")
    text_config = config.get("text_config", {})
    if not isinstance(text_config, dict):
        raise ValueError("Invalid text_config")  # noqa: TRY004 - malformed JSON document
    if config.get("quantization_config") or text_config.get("quantization_config"):
        raise ValueError("Quantized models are not part of this BF16 experiment")
    tokenizer_config = _json(_safe_artifact(snapshot, "tokenizer_config.json"))
    if not isinstance(tokenizer_config, dict):
        raise ValueError("Invalid tokenizer config")  # noqa: TRY004 - malformed JSON document
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Tokenizer data is missing")
    if not (snapshot / "chat_template.jinja").is_file() and not tokenizer_config.get(
        "chat_template"
    ):
        raise ValueError("Model chat template is missing")

    index_path = snapshot / "model.safetensors.index.json"
    single_path = snapshot / "model.safetensors"
    if index_path.exists() and single_path.exists():
        raise ValueError("Ambiguous model weights: both monolithic and sharded weights are present")
    weight_map = None
    if index_path.exists():
        index = _json(_safe_artifact(snapshot, index_path.name))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no weight map")
        if any(
            not isinstance(key, str) or not _valid_name(value) for key, value in weight_map.items()
        ):
            raise ValueError("Unsafe filename in safetensors weight map")
        weight_files = sorted(set(weight_map.values()))
        if any(not name.endswith(".safetensors") for name in weight_files):
            raise ValueError("Weight map must refer only to safetensors shards")
    elif single_path.exists():
        weight_files = [single_path.name]
    else:
        raise ValueError("Model safetensors weights are missing")

    actual_weights = sorted(
        name for name in _artifact_names(snapshot) if name.endswith(".safetensors")
    )
    if actual_weights != weight_files:
        raise ValueError("Weight inventory does not match the complete safetensors index")
    seen: dict[str, str] = {}
    bf16_count = 0
    for name in weight_files:
        header = _safetensors_header(_safe_artifact(snapshot, name))
        for tensor_name, tensor in header.items():
            if tensor_name in seen:
                raise ValueError(f"Duplicate tensor across model shards: {tensor_name}")
            seen[tensor_name] = name
            bf16_count += tensor["dtype"] == "BF16"
    if weight_map is not None and weight_map != seen:
        raise ValueError("Safetensors tensors do not match the index weight map")
    if not bf16_count:
        raise ValueError("Model snapshot contains no BF16 tensors")
    return {
        "weight_files": weight_files,
        "tensor_count": len(seen),
        "bf16_tensor_count": bf16_count,
    }


def verify_model(
    manifest: str | Path | dict[str, Any], *, verify_hashes: bool = True
) -> dict[str, Any]:
    """Verify provenance, complete weights and optionally every recorded file byte.

    ``verify_hashes=False`` is only for inexpensive metadata queries. It still
    validates the snapshot layout, inventory, file sizes and tensor headers;
    it does not replace the full verification performed before serving.
    """
    record = _json(Path(manifest)) if isinstance(manifest, (str, Path)) else manifest
    if not isinstance(record, dict):
        raise ValueError("Model manifest must be an object")  # noqa: TRY004 - invalid manifest data
    size = record.get("size")
    alias, repository, revision = model_identity(size)
    expected = {
        "schema_version": 1,
        "model_alias": alias,
        "repository": repository,
        "repo_id": repository,
        "revision": revision,
        "backend": BACKEND,
        "server_version": SERVER_VERSION,
        "precision": PRECISION,
        "model_sha256_kind": MODEL_SHA256_KIND,
        "inventory_algorithm": INVENTORY_ALGORITHM,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Model manifest {key} does not match the pinned experiment")
    snapshot = _snapshot_path(record.get("model_path"), repository, revision)
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Model manifest has no file inventory")
    if any(not isinstance(entry, dict) for entry in files):
        raise ValueError("Invalid model file inventory")
    names = [entry.get("path") for entry in files]
    if any(not _valid_name(name) for name in names) or names != sorted(set(names)):
        raise ValueError("Model file inventory must be sorted, unique and safe")
    if names != _artifact_names(snapshot):
        raise ValueError("Serving snapshot files do not match the model manifest inventory")
    for entry in files:
        if set(entry) != {"path", "size", "sha256"}:
            raise ValueError("Invalid model file inventory fields")
        if type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError("Invalid model artifact size")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError("Malformed model artifact SHA-256")
        path = _safe_artifact(snapshot, entry["path"])
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Model file size mismatch: {entry['path']}")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise ValueError(f"Model file SHA-256 mismatch: {entry['path']}")
    if record.get("sha256") != _tree_hash(files):
        raise ValueError("Model snapshot inventory SHA-256 mismatch")
    if record.get("weight_structure") != _weight_structure(snapshot):
        raise ValueError("Model weight structure differs from its manifest")
    return record


def prepare_model(size: str, out: str | Path) -> dict[str, Any]:
    """Download one pinned snapshot and write ``out/model.json`` without overwriting."""
    alias, repository, revision = model_identity(size)
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "model.json"
    if manifest.exists():
        record = verify_model(manifest)
        if record["size"] != size:
            raise ValueError("Existing model manifest belongs to another model size")
        return record
    from huggingface_hub import snapshot_download

    # HF uses HF_HOME/HF_HUB_CACHE and reuses blobs already available there.
    # An immutable commit is passed directly; no branch name is resolved at run time.
    snapshot = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
    ).absolute()
    snapshot = _snapshot_path(str(snapshot), repository, revision)
    structure = _weight_structure(snapshot)
    files = [
        {"path": name, "size": (snapshot / name).stat().st_size, "sha256": _sha256(snapshot / name)}
        for name in _artifact_names(snapshot)
    ]
    record = {
        "schema_version": 1,
        "size": size,
        "model_alias": alias,
        "repository": repository,
        "repo_id": repository,
        "revision": revision,
        "model_path": str(snapshot),
        "sha256": _tree_hash(files),
        "model_sha256_kind": MODEL_SHA256_KIND,
        "inventory_algorithm": INVENTORY_ALGORITHM,
        "backend": BACKEND,
        "server_version": SERVER_VERSION,
        "precision": PRECISION,
        "files": files,
        "weight_structure": structure,
        "download_revision_policy": "immutable-pinned-commit",
    }
    verify_model(record, verify_hashes=False)
    # Exclusive creation refuses accidental replacement by a concurrent prepare.
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


prepare = prepare_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--size", choices=MODEL_SIZES, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    value_parser = commands.add_parser("value")
    value_parser.add_argument("--manifest", type=Path, required=True)
    value_parser.add_argument(
        "--key", choices=("model_path", "model_alias", "revision", "sha256"), required=True
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        record = prepare_model(args.size, args.out)
        print(
            json.dumps(
                {
                    "model": record["model_alias"],
                    "revision": record["revision"],
                    "sha256": record["sha256"],
                }
            )
        )
    elif args.command == "verify":
        record = verify_model(args.manifest)
        print(f"Verified pinned model snapshot: {record['model_alias']} {record['sha256']}")
    else:
        record = verify_model(args.manifest, verify_hashes=False)
        print(record[args.key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
