"""Likeness generation tests (Meta muse-image, reference-conditioned).

No network: the Meta API is faked at the `imagegen.httpx` seam.
"""
import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app import imagegen, llm
from app.config import settings

WEBPISH = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 ").decode()

ASSET = {"id": "g1", "kind": "character", "name": "Norman", "version": 1,
         "description": "A cheerful garden gnome.",
         "dna": {"hair": "#ffffff", "facial_hair": "beard",
                 "outfit_top": "#dfa92e", "outfit_bottom": "#5da06a",
                 "accent": "#d63d1e", "accessory": "red pointed hat",
                 "headwear_shape": "pointed", "marking": "rosy pink cheeks"}}


def ref_image() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 40, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def subject_png_b64() -> str:
    img = Image.new("RGB", (80, 80), (255, 255, 255))
    for y in range(20, 60):
        for x in range(20, 60):
            img.putpixel((x, y), (200, 40, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fake_image_client(captured):
    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"b64_json": subject_png_b64()}]}

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None, files=None, data=None):
            captured.update(url=url, json=json, files=files, data=data)
            return Resp()

    return Client


def fake_client(status=200, captured=None):
    class Resp:
        text = "boom"

        def __init__(self):
            self.status_code = status

        def json(self):
            return {"data": [{"b64_json": WEBPISH}]}

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None, files=None, data=None):
            if captured is not None:
                captured.update(url=url, json=json, files=files, data=data)
            return Resp()

    return Client


def flaky_client(statuses, calls):
    """Fail with each status in order, then succeed; counts every attempt."""
    class Resp:
        text = "server_error"

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"data": [{"b64_json": WEBPISH}]}

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None, files=None, data=None):
            calls.append(url)
            status = statuses[len(calls) - 1] if len(calls) <= len(statuses) else 200
            return Resp(status)

    return Client


class ImageGenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._key = settings.meta_api_key

    def tearDown(self):
        settings.meta_api_key = self._key

    async def test_reference_image_conditions_the_edit_call(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_client(200, seen)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [], seed=7)
        self.assertEqual(out["source"], "meta-image-edit")
        self.assertEqual(out["model"], "muse-image-1.0")
        self.assertTrue(out["image"].startswith("data:image/webp;base64,"))
        self.assertTrue(seen["url"].endswith("/images/edits"))
        self.assertIn("image", seen["files"])
        self.assertEqual(seen["data"]["model"], "muse-image-1.0")
        self.assertIn("Norman", seen["data"]["prompt"])
        self.assertIn("red pointed hat", seen["data"]["prompt"])

    async def test_without_reference_falls_back_to_prompt_generation(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_client(200, seen)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ""}, [], seed=7)
        self.assertEqual(out["source"], "meta-image-generation")
        self.assertTrue(seen["url"].endswith("/images/generations"))
        self.assertIn("#dfa92e", seen["json"]["prompt"])

    def test_segmentation_mask_preferred_over_floodfill(self):
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle([20, 20, 60, 60], fill=255)
        with patch.object(imagegen, "_segment_mask", return_value=mask):
            out, checks = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        result = imagegen._decode_image(out).convert("RGBA")
        self.assertEqual(result.size, (1024, 1024))
        self.assertEqual(checks, [])
        # Segmentation bbox is centred: subject pixels must be central,
        # with transparent margins on every side.
        alpha = result.getchannel("A")
        bbox = alpha.getbbox()
        self.assertIsNotNone(bbox)
        self.assertGreater(bbox[0], 1024 * 0.10)
        self.assertGreater(bbox[1], 1024 * 0.10)
        self.assertLess(bbox[2], 1024 * 0.90)
        self.assertLess(bbox[3], 1024 * 0.90)

    def test_fail_open_mask_falls_back_to_floodfill(self):
        with patch.object(imagegen, "_segment_mask",
                           return_value=Image.new("L", (300, 300), 255)):
            out, _ = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        result = imagegen._decode_image(out).convert("RGBA")
        # Flood-fill on the plain test background still isolates cleanly.
        self.assertEqual(imagegen._tshirt_checks(result), [])

    def test_missing_segmentation_backend_falls_back(self):
        with patch.object(imagegen, "_segment_mask", return_value=None):
            out, _ = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        result = imagegen._decode_image(out).convert("RGBA")
        self.assertEqual(imagegen._tshirt_checks(result), [])

    def test_holey_segmentation_mask_is_filled(self):
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle([20, 20, 60, 60], fill=255)
        ImageDraw.Draw(mask).rectangle([35, 35, 45, 45], fill=0)
        with patch.object(imagegen, "_segment_mask", return_value=mask):
            out, checks = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        self.assertEqual(checks, [])
        alpha = imagegen._decode_image(out).convert("RGBA").getchannel("A")
        bbox = alpha.getbbox()
        self.assertIsNotNone(bbox)
        # The punched hole sat at the mask centre, so the output centre —
        # which would be transparent if the hole survived — must be solid.
        # (Overall fill is no longer near 1.0: feathering correctly softens
        # every edge, including the filled hole's rim.)
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.assertEqual(alpha.getpixel((cx, cy)), 255)

    def test_cut_edge_is_feathered_not_hard(self):
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([50, 50, 250, 250], fill=(200, 40, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = ("data:image/png;base64,"
               + base64.b64encode(buf.getvalue()).decode())
        mask = Image.new("L", (300, 300), 0)
        ImageDraw.Draw(mask).rectangle([50, 50, 250, 250], fill=255)
        with patch.object(imagegen, "_segment_mask", return_value=mask):
            out, _ = imagegen.isolate_subject(url)
        alpha = imagegen._decode_image(out).convert("RGBA").getchannel("A")
        fringe = sum(1 for v in alpha.getdata() if 0 < v < 255)
        # A hard mask would leave zero intermediate values: the subject is
        # small enough that no resampling softens it, so any fringe comes
        # from the feathering step.
        self.assertGreater(fringe, 500)
        self.assertEqual(alpha.getpixel((512, 512)), 255)
        self.assertEqual(alpha.getpixel((2, 2)), 0)

    def test_stray_speckle_is_dropped_from_mask(self):
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle([20, 20, 60, 60], fill=255)
        ImageDraw.Draw(mask).rectangle([0, 0, 3, 3], fill=255)
        cleaned = imagegen._fill_mask_holes(
            imagegen._significant_components(mask))
        # The corner speck is gone but the subject is untouched.
        self.assertEqual(cleaned.getpixel((1, 1)), 0)
        self.assertEqual(cleaned.getpixel((40, 40)), 255)
        with patch.object(imagegen, "_segment_mask", return_value=mask):
            out, checks = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        # Without speckle removal the corner blob would stretch the subject
        # past the safe margin and trip the containment check.
        self.assertEqual(checks, [])

    def test_thin_detached_handle_is_kept_not_sliced(self):
        # A thin tool handle, detached by a few background pixels, must
        # survive: coarse labelling plus strict gating used to erase it,
        # leaving a cleanly sliced stump (fragmentation).
        mask = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask).rectangle([100, 100, 400, 700], fill=255)
        ImageDraw.Draw(mask).rectangle([410, 398, 900, 400], fill=255)
        cleaned = imagegen._fill_mask_holes(
            imagegen._significant_components(mask))
        self.assertEqual(cleaned.getpixel((800, 399)), 255)
        self.assertEqual(cleaned.getpixel((250, 400)), 255)

    def test_second_design_part_is_kept_not_amputated(self):
        # A held tool or dirt pile is a separate, smaller, but significant
        # mask region: it must survive isolation, not be cut off.
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle([20, 20, 60, 60], fill=255)
        ImageDraw.Draw(mask).rectangle([66, 66, 78, 78], fill=255)
        cleaned = imagegen._fill_mask_holes(
            imagegen._significant_components(mask))
        self.assertEqual(cleaned.getpixel((72, 72)), 255)
        self.assertEqual(cleaned.getpixel((40, 40)), 255)

    def test_sliver_mask_falls_back_to_floodfill(self):
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle([38, 10, 40, 70], fill=255)
        with patch.object(imagegen, "_segment_mask", return_value=mask):
            out, checks = imagegen.isolate_subject(
                "data:image/png;base64," + subject_png_b64())
        result = imagegen._decode_image(out).convert("RGBA")
        self.assertEqual(imagegen._tshirt_checks(result), [])
        self.assertNotIn("subject isolation is uncertain", checks)

    def test_textured_backdrop_reports_uncertain_isolation(self):
        img = Image.effect_noise((80, 80), 64).convert("RGB")
        ImageDraw.Draw(img).rectangle([20, 20, 60, 60], fill=(200, 40, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = ("data:image/png;base64,"
               + base64.b64encode(buf.getvalue()).decode())
        sliver = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(sliver).rectangle([0, 0, 9, 9], fill=255)
        with patch.object(imagegen, "_segment_mask", return_value=sliver):
            out, checks = imagegen.isolate_subject(url)
        result = imagegen._decode_image(out).convert("RGBA")
        self.assertEqual(result.size, (1024, 1024))
        self.assertIn("subject isolation is uncertain", checks)

    async def test_adapter_generation_passes_lora_scale(self):
        settings.meta_api_key = ""
        generated = {
            "image": "data:image/png;base64," + subject_png_b64(),
            "model": "base",
            "source": "modal-lora",
        }
        with patch("app.imagegen.training.generate_with_adapter",
                   return_value=generated) as generate, \
             patch("app.validation.check_probe_image", return_value=[]):
            await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", adapter_ref="modal://x/y.safetensors",
                lora_scale=0.6)
        self.assertEqual(generate.call_args.args[5], 0.6)

    async def test_adapter_generation_uses_trigger_without_meta_key(self):
        settings.meta_api_key = ""
        generated = {
            "image": "data:image/png;base64," + subject_png_b64(),
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "source": "modal-lora",
        }
        with patch("app.imagegen.training.generate_with_adapter",
                   return_value=generated) as generate, \
             patch("app.validation.check_probe_image", return_value=[]):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", adapter_ref="modal://lora-studio/adapters/norman/x.safetensors")
        self.assertEqual(out["source"], "modal-lora")
        self.assertIn("norman_tok,", generate.call_args.args[1])
        self.assertIn("white background", generate.call_args.args[1])
        self.assertTrue(out["image"].startswith("data:image/png;base64,"))
        self.assertEqual(out["model"], generated["model"])
        self.assertEqual(out["checks"], [])

    async def test_adapter_prompt_is_compact_and_action_led(self):
        settings.meta_api_key = ""
        generated = {
            "image": "data:image/png;base64," + subject_png_b64(),
            "model": "base",
            "source": "modal-lora",
        }
        with patch("app.imagegen.training.generate_with_adapter",
                   return_value=generated) as generate, \
             patch("app.validation.check_probe_image", return_value=[]):
            await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt",
                adapter_ref="modal://lora-studio/adapters/norman/x.safetensors",
                action="shoveling a pile of dirt with a shovel")
        prompt = generate.call_args.args[1]
        self.assertTrue(prompt.startswith("norman_tok,"))
        self.assertIn("shoveling a pile of dirt with a shovel", prompt)
        self.assertIn("no duplicates", prompt)
        self.assertIn("full body, head to toe", prompt)
        self.assertNotIn("standing pose", prompt)
        # SDXL truncates past ~77 tokens: the whole prompt must fit.
        self.assertLess(len(prompt.split()), 70)

    async def test_panel_prompt_carries_action_without_forced_style(self):
        settings.meta_api_key = "test-key"
        seen = {}
        panel = {"action": "Norman waves at a neighbour.",
                 "layout": {"pose": "stand", "expression": "calm"},
                 "camera": {"shot": "medium"}}
        with patch.object(imagegen.httpx, "AsyncClient", fake_client(200, seen)):
            out = await imagegen.generate_panel_graphic(
                {**ASSET, "reference_image": ref_image()}, [], panel)
        self.assertEqual(out["source"], "meta-image-edit")
        self.assertIn("Norman waves at a neighbour.", seen["data"]["prompt"])
        self.assertNotIn("storybook", seen["data"]["prompt"])
        self.assertIn("exact art style", seen["data"]["prompt"])

    async def test_likeness_score_compares_encoded_traits(self):
        async def fake_encode(image, kind):
            return {"dna": {"skin": "#f9c48f", "hair": "#fefefe",
                            "outfit_top": "#dfa92e", "outfit_bottom": "#5da06a",
                            "accent": "#d63d1e", "accessory": "red pointed hat",
                            "facial_hair": "beard"},
                    "fallback": False}

        bible = {"skin": "#f9c48f", "hair": "#ffffff",
                 "outfit_top": "#dfa92e", "outfit_bottom": "#5da06a",
                 "accent": "#d63d1e", "accessory": "red pointed hat",
                 "facial_hair": "beard"}
        with patch.object(imagegen, "encode_reference", fake_encode):
            score, issues = await imagegen.score_likeness("data:image/webp;base64,x", bible)
        self.assertGreaterEqual(score, 80.0)
        self.assertEqual(issues, [])

    async def test_tshirt_profile_isolates_and_contains_subject(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_image_client(seen)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [], profile="tshirt")
        self.assertEqual(out["profile"], "tshirt")
        self.assertIn("plain solid white background", seen["data"]["prompt"])
        self.assertIn("fully inside", seen["data"]["prompt"])
        self.assertEqual(out["checks"], [])
        raw = base64.b64decode(out["image"].split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(img.getpixel((2, 2))[3], 0)  # transparent corner
        bbox = img.getchannel("A").getbbox()
        self.assertGreaterEqual(bbox[0], img.width * 0.10)
        self.assertGreaterEqual(bbox[1], img.height * 0.10)
        self.assertLessEqual(bbox[2], img.width * 0.90)
        self.assertLessEqual(bbox[3], img.height * 0.90)

    async def test_video_profile_asks_for_widescreen_headroom(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_image_client(seen)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [], profile="video")
        self.assertEqual(out["profile"], "video")
        self.assertIn("16:9", seen["data"]["prompt"])
        self.assertIn("headroom", seen["data"]["prompt"])

    def test_isolate_subject_rejects_blank_image(self):
        buf = io.BytesIO()
        Image.new("RGB", (40, 40), (255, 255, 255)).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        with self.assertRaises(llm.LLMError):
            imagegen.isolate_subject(url)

    async def test_tshirt_design_prompt_and_local_phrase(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_image_client(seen)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", composition="vignette",
                elements="mushrooms and a small sun",
                phrase="GNOME SWEET GNOME", phrase_placement="below")
        prompt = seen["data"]["prompt"]
        self.assertIn("mushrooms and a small sun", prompt)
        self.assertIn("vignette", prompt)
        self.assertIn("richly detailed", prompt)  # dense, not a bare figure
        self.assertNotIn("sunburst", prompt)  # halo only if the user asks
        self.assertNotIn("GNOME SWEET GNOME", prompt)  # text stays local
        self.assertEqual(out["checks"], [])
        raw = base64.b64decode(out["image"].split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
        self.assertEqual(img.mode, "RGBA")
        # phrase band under the subject carries visible pixels
        bottom = img.crop((0, int(img.height * 0.84), img.width, img.height))
        self.assertGreater(
            sum(1 for px in bottom.getchannel("A").getdata() if px > 0), 500)

    def test_compose_phrase_keeps_spelling_exact(self):
        buf = io.BytesIO()
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        for y in range(60, 140):
            for x in range(70, 130):
                img.putpixel((x, y), (200, 40, 30, 255))
        img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        out_url, checks = imagegen.compose_phrase(url, "GNOME SWEET GNOME", "below")
        self.assertEqual(checks, [])
        raw = base64.b64decode(out_url.split(",", 1)[1])
        result = Image.open(io.BytesIO(raw))
        self.assertEqual(result.mode, "RGBA")
        # styled lettering in the phrase band: coloured fill sampled from
        # the artwork, not plain black — a warm pixel must be present
        band = result.crop((0, result.height - 110, result.width, result.height))
        pixels = [px for px in band.getdata() if px[3] > 0]
        self.assertTrue(any(p[0] > 120 and p[2] < 130 for p in pixels))

    async def test_action_beats_stock_pose_in_prompt(self):
        prompt = imagegen.build_likeness_prompt(
            ASSET, [], pose="stand", expression="calm", shot="medium",
            action="shoveling a pile of dirt with a shovel")
        self.assertIn("Action (depict this clearly", prompt)
        self.assertIn("shoveling a pile of dirt with a shovel", prompt)
        self.assertNotIn("standing pose", prompt)

    async def test_no_action_keeps_pose_word(self):
        prompt = imagegen.build_likeness_prompt(
            ASSET, [], pose="stand", expression="calm", shot="medium")
        self.assertIn("standing pose", prompt)
        self.assertNotIn("Action (depict this clearly", prompt)

    async def test_action_renders_compact_and_reposable(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_image_client(seen)):
            await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", action="shoveling a pile of dirt")
        prompt = seen["data"]["prompt"]
        self.assertIn("one connected group", prompt)
        self.assertIn("change the pose as needed", prompt)
        self.assertNotIn("change only the pose", prompt)

    def _dna_probe_png(self):
        img = Image.new("RGB", (80, 80), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 10, 60, 25], fill=(216, 78, 42))
        d.rectangle([20, 25, 60, 45], fill=(223, 169, 46))
        d.rectangle([20, 45, 60, 70], fill=(93, 160, 106))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return ("data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode())

    async def test_adapter_retries_seeds_until_direction_passes(self):
        settings.meta_api_key = ""
        bust = {"image": self._dna_probe_png(),
                "model": "base", "source": "modal-lora"}
        full = {"image": self._dna_probe_png(),
                "model": "base", "source": "modal-lora"}
        mask_ok = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask_ok).rectangle([20, 10, 60, 70], fill=255)
        mask_bust = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask_bust).rectangle([25, 10, 55, 40], fill=255)
        # Two probe checks plus the final isolation cutout each consume
        # one segmentation call.
        masks = [mask_bust, mask_ok, mask_ok]
        with patch("app.imagegen.training.generate_with_adapter",
                   side_effect=[bust, full]) as generate, \
             patch.object(imagegen, "_segment_mask",
                          side_effect=lambda img: masks.pop(0)):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", seed=7,
                adapter_ref="modal://lora-studio/adapters/norman/x.safetensors")
        self.assertEqual(generate.call_count, 2)
        # First seed busted, retry won: honest winning seed is reported.
        self.assertEqual(out["seed"], 7 + 1009)
        self.assertEqual(out["checks"], [])

    async def test_adapter_gives_up_after_three_tries(self):
        settings.meta_api_key = ""
        bust = {"image": "data:image/png;base64," + subject_png_b64(),
                "model": "base", "source": "modal-lora"}
        mask_bust = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask_bust).rectangle([25, 10, 55, 40], fill=255)
        with patch("app.imagegen.training.generate_with_adapter",
                   return_value=bust) as generate, \
             patch.object(imagegen, "_segment_mask", return_value=mask_bust):
            out = await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt", seed=7,
                adapter_ref="modal://lora-studio/adapters/norman/x.safetensors")
        self.assertEqual(generate.call_count, 3)
        self.assertTrue(any("direction not fully followed" in c
                            for c in out["checks"]))

    async def test_no_action_keeps_change_only_pose(self):
        settings.meta_api_key = "test-key"
        seen = {}
        with patch.object(imagegen.httpx, "AsyncClient", fake_image_client(seen)):
            await imagegen.generate_graphic(
                {**ASSET, "reference_image": ref_image()}, [],
                profile="tshirt")
        prompt = seen["data"]["prompt"]
        self.assertIn("change only the pose", prompt)
        self.assertNotIn("one connected group", prompt)

    async def test_transient_500_is_retried_then_succeeds(self):
        settings.meta_api_key = "test-key"
        calls = []
        with patch.object(imagegen.httpx, "AsyncClient",
                          flaky_client([500, 500], calls)):
            with patch.object(imagegen.asyncio, "sleep", return_value=None):
                out = await imagegen.generate_graphic(
                    {**ASSET, "reference_image": ""}, [])
        self.assertTrue(out["image"].startswith("data:image"))
        self.assertEqual(len(calls), 3)

    async def test_permanent_400_raises_without_retry(self):
        settings.meta_api_key = "test-key"
        calls = []
        with patch.object(imagegen.httpx, "AsyncClient",
                          flaky_client([400, 400, 400], calls)):
            with self.assertRaises(llm.LLMError):
                await imagegen.generate_graphic(
                    {**ASSET, "reference_image": ""}, [])
        self.assertEqual(len(calls), 1)

    async def test_upstream_error_raises_llmerror(self):
        settings.meta_api_key = "test-key"
        with patch.object(imagegen.httpx, "AsyncClient", fake_client(500)):
            with patch.object(imagegen.asyncio, "sleep", return_value=None):
                with self.assertRaises(llm.LLMError):
                    await imagegen.generate_graphic(
                        {**ASSET, "reference_image": ref_image()}, [], seed=7)


if __name__ == "__main__":
    unittest.main()
