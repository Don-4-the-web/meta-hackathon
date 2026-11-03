"""Reference-image -> DNA encoding tests.

Runs with the stdlib + Pillow:  python -m unittest discover -s tests -v
No network: vision is faked at the `encode.call_meta_vision_json` seam.
"""
import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

from app import encode, llm
from app.config import settings


def band_image() -> str:
    """Synthetic 'figure': white head band, mustard torso, green legs."""
    img = Image.new("RGB", (60, 90))
    px = img.load()
    for y in range(90):
        colour = (242, 239, 230) if y < 30 else (201, 143, 27) if y < 60 else (62, 107, 47)
        for x in range(60):
            px[x, y] = colour
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class EncodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._key = settings.meta_api_key

    def tearDown(self):
        settings.meta_api_key = self._key

    async def test_vision_result_is_coerced_to_valid_dna(self):
        settings.meta_api_key = "test-key"

        async def fake(system, user, image, temperature=0.2, **kw):
            return {"description": "A gnome with a white beard.",
                    "dna": {"form": "HUMAN", "build": "stocky",
                            "skin": "#E8B08A", "hair": "fffef7",
                            "hair_style": "long", "eyes": "#222222",
                            "outfit_top": "#E0AA3E", "outfit_bottom": "#5FA271",
                            "accent": "#B91C1C", "accessory": "red pointed hat",
                            "marking": "none", "surprise": "drop me"}}

        with patch.object(encode, "call_meta_vision_json", fake):
            out = await encode.encode_reference(band_image(), "character")
        dna = out["dna"]
        self.assertFalse(out["fallback"])
        self.assertEqual(out["source"], "meta-vision")
        self.assertEqual(dna["form"], "human")          # enum, case-folded
        self.assertEqual(dna["build"], "average")       # invalid enum -> default
        self.assertEqual(dna["skin"], "#e8b08a")        # hex normalised
        self.assertEqual(dna["hair"], "#fffef7")        # missing '#' repaired
        self.assertEqual(dna["accessory"], "red pointed hat")
        self.assertNotIn("surprise", dna)               # unknown fields dropped
        self.assertEqual(out["description"], "A gnome with a white beard.")

    async def test_vision_failure_falls_back_to_palette_bands(self):
        settings.meta_api_key = "test-key"

        async def boom(system, user, image, temperature=0.2, **kw):
            raise llm.LLMError("vision unsupported")

        with patch.object(encode, "call_meta_vision_json", boom):
            out = await encode.encode_reference(band_image(), "character")
        self.assertTrue(out["fallback"])
        self.assertEqual(out["source"], "palette")
        self.assertEqual(out["dna"]["hair"], "#f2efe6")
        self.assertEqual(out["dna"]["outfit_top"], "#c98f1b")
        self.assertEqual(out["dna"]["outfit_bottom"], "#3e6b2f")
        self.assertTrue(out["palette"])

    async def test_rejects_non_image_payload(self):
        with self.assertRaises(ValueError):
            await encode.encode_reference("not an image", "character")
        with self.assertRaises(ValueError):
            await encode.encode_reference("data:text/html;base64,PGI+", "character")


class VisionCallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._key = settings.meta_api_key

    def tearDown(self):
        settings.meta_api_key = self._key

    async def test_sends_image_url_part_in_json_mode(self):
        settings.meta_api_key = "test-key"
        seen = {}

        class Resp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({"dna": {}})}}]}

        class Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                seen.update(url=url, payload=json)
                return Resp()

        with patch.object(llm.httpx, "AsyncClient", Client):
            out = await llm.call_meta_vision_json("sys", "describe",
                                                  "data:image/png;base64,AAAA")
        self.assertEqual(out, {"dna": {}})
        content = seen["payload"]["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "describe"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,AAAA")
        self.assertEqual(seen["payload"]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
