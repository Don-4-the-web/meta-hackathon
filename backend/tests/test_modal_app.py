"""Deployable Modal app tests. Importing the app must not spend credits."""
import base64
import io
import unittest
import zipfile
from pathlib import Path

from PIL import Image

import modal_lora_app as lora_app


def png_bytes(color=(10, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(buf, format="PNG")
    return buf.getvalue()


def make_zip(*names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if name.endswith(".txt"):
                archive.writestr(name, "norman_tok, a garden gnome")
            else:
                archive.writestr(name, png_bytes())
    return buf.getvalue()


class ModalAppTests(unittest.TestCase):
    def test_validates_dataset_pairing_without_touching_modal(self):
        files = dict(lora_app.safe_dataset_files(make_zip("000.png", "000.txt")))
        self.assertEqual(set(files), {"000.png", "000.txt"})
        with self.assertRaises(ValueError):
            lora_app.safe_dataset_files(make_zip("000.png"))

    def test_rejects_path_traversal_but_keeps_safe_stem(self):
        blob = make_zip("../evil.png", "../evil.txt", "ok.png", "ok.txt")
        files = dict(lora_app.safe_dataset_files(blob))
        self.assertEqual(set(files), {"evil.png", "evil.txt", "ok.png", "ok.txt"})

    def test_training_config_is_sdxl_and_cost_capped(self):
        config = lora_app.training_config(
            "norman", "norman_tok", "/vol/datasets/norman", {"steps": 5000}
        )
        process = config["config"]["process"][0]
        self.assertEqual(process["type"], "sd_trainer")
        self.assertEqual(process["model"]["name_or_path"], lora_app.BASE_MODEL)
        self.assertTrue(process["model"]["is_xl"])
        self.assertEqual(process["train"]["steps"], lora_app.MAX_TRAIN_STEPS)
        self.assertTrue(process["train"]["disable_sampling"])
        self.assertNotIn("sample", process)

    def test_training_config_supports_one_validation_sample(self):
        config = lora_app.training_config(
            "norman",
            "norman_tok",
            "/vol/datasets/norman",
            {"steps": 200, "validation_prompt": "watering plants"},
        )
        process = config["config"]["process"][0]
        self.assertEqual(
            process["sample"]["prompts"], ["[trigger], watering plants"]
        )
        self.assertNotIn("disable_sampling", process["train"])

    def test_find_adapter_prefers_newest_checkpoint(self):
        import tempfile

        keeper = tempfile.TemporaryDirectory(prefix="lora-app-test-")
        self.addCleanup(keeper.cleanup)
        root = Path(keeper.name) / "adapter"
        root.mkdir(parents=True)
        first = root / "first.safetensors"
        second = root / "nested" / "second.safetensors"
        second.parent.mkdir()
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        self.assertEqual(lora_app.find_adapter_file(root), second)


if __name__ == "__main__":
    unittest.main()
