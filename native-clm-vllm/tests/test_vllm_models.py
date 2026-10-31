"""Small synthetic model files exercise provenance without downloading weights."""

import copy
import fnmatch
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/ctxmgr/harness/clm_vllm_models.py"
SPEC = importlib.util.spec_from_file_location("clm_vllm_models_test_target", MODULE_PATH)
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


def write_tensor(path, names=("model.weight",), dtype="BF16"):
    width = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
    header = {
        name: {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [index * width, (index + 1) * width],
        }
        for index, name in enumerate(names)
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * width * len(names))


class ModelManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.size = "27B"
        self.alias, self.repository, self.revision = models.model_identity(self.size)
        self.snapshot = (
            self.root / "hub" / "models--Qwen--Qwen3.5-27B" / "snapshots" / self.revision
        )
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "text_config": {"dtype": "bfloat16"}})
        )
        (self.snapshot / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"})
        )
        (self.snapshot / "tokenizer.json").write_text('{"version":"1.0"}')
        write_tensor(self.snapshot / "model.safetensors")
        self.requests = []

        def download(**kwargs):
            self.requests.append(kwargs)
            return str(self.snapshot)

        self.hub = types.SimpleNamespace(snapshot_download=download)
        self.patch_hub = patch.dict(sys.modules, {"huggingface_hub": self.hub})
        self.patch_hub.start()
        self.addCleanup(self.patch_hub.stop)
        self.out = self.root / "manifest"

    def prepare(self):
        return models.prepare_model(self.size, self.out)

    def test_five_models_include_9b_and_27b_with_immutable_revisions(self):
        self.assertEqual(models.MODEL_SIZES, ("0.8B", "2B", "4B", "9B", "27B"))
        for size in models.MODEL_SIZES:
            alias, repository, revision = models.model_identity(size)
            self.assertEqual(alias, f"Qwen3.5-{size}-BF16")
            self.assertEqual(repository, f"Qwen/Qwen3.5-{size}")
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            models.model_identity("9B")[2],
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        )
        with self.assertRaises(ValueError):
            models.model_identity("14B")

    def test_monolithic_snapshot_and_manifest_validate(self):
        record = self.prepare()
        verified = models.verify_model(self.out / "model.json")
        self.assertEqual(record, verified)
        self.assertEqual(record["precision"], "bfloat16")
        self.assertEqual(record["model_sha256_kind"], "hf-snapshot-tree-sha256")
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.requests[0]["revision"], self.revision)
        self.assertEqual(self.requests[0]["repo_id"], self.repository)
        self.assertNotIn("*.bin", self.requests[0]["allow_patterns"])

    def test_existing_manifest_is_verified_and_not_replaced(self):
        self.prepare()
        manifest = self.out / "model.json"
        original = manifest.read_bytes()
        models.prepare_model(self.size, self.out)
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(len(self.requests), 1)

    def test_same_length_file_corruption_is_detected_by_hash(self):
        self.prepare()
        path = self.snapshot / "model.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[:-2] + b"\x01\x01")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            models.verify_model(self.out / "model.json")
        # Metadata access explicitly opts out of reading every model byte.
        models.verify_model(self.out / "model.json", verify_hashes=False)

    def test_hash_or_revision_tampering_is_rejected(self):
        original = self.prepare()
        for key, value in (("sha256", "0" * 64), ("revision", "0" * 40), ("precision", "fp8")):
            with self.subTest(key=key):
                record = copy.deepcopy(original)
                record[key] = value
                with self.assertRaises(ValueError):
                    models.verify_model(record)

    def test_hf_blob_symlinks_are_supported(self):
        weight = self.snapshot / "model.safetensors"
        content = weight.read_bytes()
        blobs = self.snapshot.parent.parent / "blobs"
        blobs.mkdir()
        blob = blobs / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        weight.unlink()
        weight.symlink_to(Path("..") / ".." / "blobs" / blob.name)
        self.prepare()
        models.verify_model(self.out / "model.json")

    def test_symlink_outside_repository_is_rejected(self):
        weight = self.snapshot / "model.safetensors"
        external = self.root / "external.safetensors"
        external.write_bytes(weight.read_bytes())
        weight.unlink()
        weight.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.prepare()

    def make_shards(self):
        (self.snapshot / "model.safetensors").unlink()
        names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        write_tensor(self.snapshot / names[0], ("model.first",))
        write_tensor(self.snapshot / names[1], ("model.second",))
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.first": names[0], "model.second": names[1]}})
        )
        return names

    def test_sharded_weights_are_complete_and_match_index(self):
        names = self.make_shards()
        record = self.prepare()
        self.assertEqual(record["weight_structure"]["weight_files"], list(names))
        self.assertEqual(record["weight_structure"]["tensor_count"], 2)

    def test_pinned_qwen_shard_names_and_processor_files_are_downloaded(self):
        # The official Qwen3.5 repos use this prefix, including the extra
        # '.safetensors' before '-00001', for every size in this experiment.
        published = [
            "model.safetensors-00001-of-00001.safetensors",
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00011-of-00011.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ]
        for filename in published:
            with self.subTest(filename=filename):
                self.assertTrue(
                    any(fnmatch.fnmatch(filename, pattern) for pattern in models.DOWNLOAD_PATTERNS)
                )
        (self.snapshot / "model.safetensors").unlink()
        shard = "model.safetensors-00001-of-00001.safetensors"
        write_tensor(self.snapshot / shard)
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": shard}})
        )
        for filename in published[-3:]:
            (self.snapshot / filename).write_text("{}")
        record = self.prepare()
        self.assertEqual(record["weight_structure"]["weight_files"], [shard])
        self.assertTrue(set(published[-3:]) <= {entry["path"] for entry in record["files"]})

    def test_official_bf16_weights_with_fp32_auxiliary_tensors_are_accepted(self):
        names = self.make_shards()
        write_tensor(self.snapshot / names[1], ("model.second",), dtype="F32")
        record = self.prepare()
        self.assertEqual(record["weight_structure"]["bf16_tensor_count"], 1)
        self.assertEqual(record["weight_structure"]["tensor_count"], 2)

    def test_missing_shard_is_rejected(self):
        names = self.make_shards()
        (self.snapshot / names[1]).unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.prepare()

    def test_wrong_shard_tensor_is_rejected(self):
        names = self.make_shards()
        write_tensor(self.snapshot / names[1], ("model.wrong",))
        with self.assertRaisesRegex(ValueError, "index weight map"):
            self.prepare()

    def test_unsafe_index_path_is_rejected(self):
        self.make_shards()
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.first": "../../outside.safetensors"}})
        )
        with self.assertRaisesRegex(ValueError, "Unsafe filename"):
            self.prepare()

    def test_unrecorded_artifact_is_rejected(self):
        self.prepare()
        (self.snapshot / "generation_config.json").write_text('{"temperature":0.1}')
        with self.assertRaisesRegex(ValueError, "manifest inventory"):
            models.verify_model(self.out / "model.json")

    def test_missing_tokenizer_or_template_is_rejected(self):
        (self.snapshot / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "Tokenizer data"):
            self.prepare()
        (self.snapshot / "tokenizer.json").write_text("{}")
        (self.snapshot / "tokenizer_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "chat template"):
            self.prepare()

    def test_quantized_weights_are_rejected(self):
        write_tensor(self.snapshot / "model.safetensors", dtype="F8_E4M3")
        with self.assertRaisesRegex(ValueError, "quantized weight dtype"):
            self.prepare()

    def test_truncated_weight_data_is_rejected(self):
        path = self.snapshot / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "data offsets"):
            self.prepare()

    def test_cli_value_outputs_only_value(self):
        self.prepare()
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = models.main(
                ["value", "--manifest", str(self.out / "model.json"), "--key", "model_alias"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(stream.getvalue(), self.alias + "\n")


if __name__ == "__main__":
    unittest.main()
