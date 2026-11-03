"""Acceptance-gate tests: probe checks and the validate flow.

No network: adapter inference is faked at the `validation.training` seam
and segmentation at the `validation.imagegen` seam.
"""
import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app import validation
from app.config import settings

DNA = {"outfit_top": "#dfa92e", "outfit_bottom": "#5da06a",
       "accent": "#d84e2a", "accessory": "red pointed hat"}
ASSET = {"id": "v1", "kind": "character", "name": "Norman",
         "description": "A cheerful garden gnome.",
         "dna": DNA, "version": 1}
ADAPTER = "modal://lora-studio/adapters/norman/x.safetensors"


def png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def good_figure(size=(300, 300), box=(60, 30, 240, 280),
                hat=(216, 78, 42)):
    """Full-body-ish figure: red hat, mustard shirt, green bottoms."""
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    hatline = box[1] + (box[3] - box[1]) // 4
    waist = (box[1] + box[3]) // 2
    d.rectangle([box[0], box[1], box[2], hatline], fill=hat)
    d.rectangle([box[0], hatline, box[2], waist], fill=(223, 169, 46))
    d.rectangle([box[0], waist, box[2], box[3]], fill=(93, 160, 106))
    return img


def mask_for(box, size=(300, 300), extra=()):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rectangle(box, fill=255)
    for b in extra:
        d.rectangle(b, fill=255)
    return mask


class ProbeCheckTests(unittest.TestCase):
    def test_clean_probe_passes(self):
        box = (60, 30, 240, 280)
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            reasons = validation.check_probe_image(
                good_figure(box=box), DNA, needs_body=True)
        self.assertEqual(reasons, [])

    def test_twin_figures_fail_single_subject(self):
        mask = mask_for((20, 60, 130, 250), extra=[(170, 60, 280, 250)])
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask):
            reasons = validation.check_probe_image(
                good_figure(), DNA, needs_body=False)
        self.assertIn("multiple figures instead of one character", reasons)

    def test_bust_fails_body_probe(self):
        box = (100, 60, 200, 150)
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            reasons = validation.check_probe_image(
                good_figure(box=box), DNA, needs_body=True)
        self.assertIn("full body not visible (bust/close-up)", reasons)

    def test_missing_outfit_colour_fails(self):
        box = (60, 30, 240, 280)
        grey = Image.new("RGB", (300, 300), (255, 255, 255))
        ImageDraw.Draw(grey).rectangle(box, fill=(150, 150, 150))
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            reasons = validation.check_probe_image(
                grey, DNA, needs_body=False)
        self.assertIn("outfit colour missing: shirt", reasons)
        self.assertIn("outfit colour missing: bottoms", reasons)
        self.assertIn("headwear colour wrong or missing", reasons)

    def test_fringe_pixels_do_not_count_as_outfit(self):
        # A bust with the right hat but only highlight-fringe where the
        # shirt and overalls should be must still fail both colours.
        box = (60, 30, 240, 280)
        img = good_figure(box=box)
        hatline = box[1] + (box[3] - box[1]) // 4
        ImageDraw.Draw(img).rectangle(
            [box[0], hatline, box[2], box[3]], fill=(150, 150, 150))
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            reasons = validation.check_probe_image(
                img, DNA, needs_body=False)
        self.assertIn("outfit colour missing: shirt", reasons)
        self.assertIn("outfit colour missing: bottoms", reasons)
        self.assertNotIn("headwear colour wrong or missing", reasons)

    def test_wrong_hat_colour_fails(self):
        # The user's exact failure: a green cap where the red hat belongs.
        box = (60, 30, 240, 280)
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            reasons = validation.check_probe_image(
                good_figure(box=box, hat=(93, 160, 106)),
                DNA, needs_body=False)
        self.assertIn("headwear colour wrong or missing", reasons)
        self.assertNotIn("outfit colour missing: shirt", reasons)
        self.assertNotIn("outfit colour missing: bottoms", reasons)

    def test_no_subject_fails(self):
        with patch.object(validation.imagegen, "_segment_mask",
                          return_value=None):
            reasons = validation.check_probe_image(
                good_figure(), DNA, needs_body=True)
        self.assertEqual(reasons, ["no coherent subject isolated"])


class ValidateFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._key = settings.meta_api_key

    def tearDown(self):
        settings.meta_api_key = self._key

    async def test_passing_adapter_returns_passed(self):
        box = (60, 30, 240, 280)
        generated = {"image": png_url(good_figure(box=box)),
                     "model": "base", "source": "modal-lora"}
        prompts = []

        def fake(ref, prompt, *a, **kw):
            prompts.append(prompt)
            return generated

        with patch.object(validation.training, "generate_with_adapter",
                          side_effect=fake), \
             patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            verdict = await validation.validate_adapter(ASSET, ADAPTER)
        self.assertTrue(verdict["passed"])
        self.assertEqual(len(verdict["probes"]), 3)
        self.assertTrue(all(p["passed"] for p in verdict["probes"]))
        self.assertEqual(len(prompts), 3)
        self.assertIn("norman", prompts[0].split(",")[0])
        self.assertIn("full body", prompts[0])
        self.assertIn("close-up portrait", prompts[2])

    async def test_failing_probe_parks_verdict(self):
        box = (100, 60, 200, 150)
        generated = {"image": png_url(good_figure(box=box)),
                     "model": "base", "source": "modal-lora"}
        with patch.object(validation.training, "generate_with_adapter",
                          return_value=generated), \
             patch.object(validation.imagegen, "_segment_mask",
                          return_value=mask_for(box)):
            verdict = await validation.validate_adapter(ASSET, ADAPTER)
        self.assertFalse(verdict["passed"])
        neutral = next(p for p in verdict["probes"] if p["name"] == "neutral")
        self.assertFalse(neutral["passed"])
        self.assertTrue(neutral["reasons"])


if __name__ == "__main__":
    unittest.main()
