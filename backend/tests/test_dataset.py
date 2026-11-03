"""Dataset builder + training gate tests. No Modal account needed."""
import base64
import io
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image

from app import dataset, training


def png_data_url(color=(180, 60, 30), size=(64, 64)) -> str:
    from PIL import ImageDraw
    buf = io.BytesIO()
    img = Image.new("RGB", size, (250, 250, 250))
    ImageDraw.Draw(img).rectangle(
        [size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 4],
        fill=color)
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

ASSET = {"id": "g1", "kind": "character", "name": "Norman The Good Gnome",
         "description": "A cheerful garden gnome.",
         "dna": {"accessory": "red pointed hat", "facial_hair": "beard"},
         "reference_image": "data:image/png;base64,AAA",
         "reference_images": ["data:image/png;base64,AAA",
                              "data:image/png;base64,BBB"]}


class DatasetTests(unittest.TestCase):
    def test_trigger_is_stable_slug(self):
        self.assertEqual(dataset.trigger_for("Norman The Good Gnome"),
                         "norman_the_good_gnome_tok")
        self.assertEqual(dataset.trigger_for("  "), "character_tok")

    def test_dataset_uses_all_reference_images_with_trigger_captions(self):
        ds = dataset.build_dataset(ASSET)
        self.assertEqual(ds["count"], 2)
        for item in ds["items"]:
            self.assertIn("norman_the_good_gnome_tok", item["caption"])
            self.assertIn("red pointed hat", item["caption"])

    def test_dataset_requires_images(self):
        with self.assertRaises(ValueError):
            dataset.build_dataset({**ASSET, "reference_image": "",
                                   "reference_images": []})

    def test_training_zip_contains_image_caption_pairs(self):
        # tiny valid PNGs are needed for decode; reuse imagegen helper
        from tests.test_imagegen import ref_image
        asset = {**ASSET, "reference_image": "",
                 "reference_images": [ref_image(), ref_image()]}
        blob = dataset.training_zip(asset)
        names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
        self.assertIn("000.png", names)
        self.assertIn("000.txt", names)
        self.assertIn("001.png", names)

    def test_training_blocked_without_gcp_credentials(self):
        with patch.object(training, "training_provider", return_value="gcp"):
            with patch.object(training, "gcp_configured", return_value=False):
                with self.assertRaises(training.TrainingNotConfigured):
                    training.submit_training(ASSET, b"zip")

    def test_namespaced_trigger_isolates_users(self):
        owned = {**ASSET, "user_id": "abcdefab-1234-5678-9abc-def012345678"}
        self.assertEqual(dataset.namespaced_trigger(ASSET),
                         "norman_the_good_gnome_tok")
        self.assertEqual(dataset.namespaced_trigger(owned),
                         "uabcdefab_norman_the_good_gnome_tok")
        ds = dataset.build_dataset(owned)
        for item in ds["items"]:
            self.assertIn("uabcdefab_norman_the_good_gnome_tok", item["caption"])

    def test_caption_overrides_apply_by_fingerprint(self):
        asset = {**ASSET, "reference_image": png_data_url(color=(180, 60, 30)),
                 "reference_images": [png_data_url(color=(60, 120, 200))]}
        ds = dataset.build_dataset(asset)
        fp = ds["items"][0]["fingerprint"]
        custom = "norman_the_good_gnome_tok, custom caption here"
        ds2 = dataset.build_dataset({**asset, "caption_overrides": {fp: custom}})
        self.assertEqual(ds2["items"][0]["caption"], custom)
        self.assertTrue(ds2["items"][0]["custom"])
        self.assertFalse(ds2["items"][1]["custom"])
        self.assertIn("red pointed hat", ds2["items"][1]["caption"])

    def test_grade_flags_duplicates_blank_and_size(self):
        dup = png_data_url()
        asset = {**ASSET, "reference_image": dup, "reference_images": [dup]}
        graded = dataset.grade_dataset(asset)
        self.assertFalse(graded["ready"])
        codes = {c["code"] for c in graded["checks"]}
        self.assertIn("duplicate", codes)
        self.assertIn("too_few", codes)

    def test_grade_passes_varied_set(self):
        colors = [(180, 60, 30), (60, 120, 200), (90, 180, 90), (200, 120, 40)]
        imgs = [png_data_url(color=c, size=(300, 300)) for c in colors]
        asset = {**ASSET, "reference_image": imgs[0],
                 "reference_images": imgs[1:]}
        graded = dataset.grade_dataset(asset)
        codes = {c["code"] for c in graded["checks"] if c["level"] == "error"}
        self.assertEqual(codes, set())
        self.assertTrue(graded["ready"])


if __name__ == "__main__":
    unittest.main()
