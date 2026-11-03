"""Meta-integration tests for the story pipeline.

Runs with the stdlib only:  python -m unittest discover -s tests -v
No network: Meta is faked at the `pipeline.call_meta_json` / `llm.httpx` seam,
and the offline paths run with an empty API key.
"""
import json
import unittest
from unittest.mock import patch

from app import llm, pipeline
from app.config import settings

ASSETS = [
    {"id": "c1", "kind": "character", "name": "Mira", "description": "", "version": 1,
     "dna": {"form": "fox", "skin": "#e8b08a", "hair": "#b3502e",
             "outfit_top": "#256d5a", "outfit_bottom": "#33404f",
             "accent": "#d9a03c", "accessory": "brass goggles", "marking": "accent stripe"}},
    {"id": "e1", "kind": "environment", "name": "Docks", "description": "", "version": 1,
     "dna": {"biome": "sky docks", "sky": "#a8c8e0", "ground": "#8a7a5a",
             "secondary": "#3c4c5e", "landmark": "towers", "time": "day"}},
]
PREMISE = "Mira finds a storm compass. She races the storm."


class KeyGuard(unittest.TestCase):
    def setUp(self):
        self._key = settings.meta_api_key

    def tearDown(self):
        settings.meta_api_key = self._key


class CallMetaJsonTests(KeyGuard, unittest.IsolatedAsyncioTestCase):
    async def test_parses_json_object_and_sends_json_mode(self):
        settings.meta_api_key = "test-key"
        seen = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({"beats": []})}}]}

            @property
            def text(self):
                return ""

        class Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                seen.update(url=url, payload=json, headers=headers)
                return Resp()

        with patch.object(llm.httpx, "AsyncClient", Client):
            out = await llm.call_meta_json("sys", "user")
        self.assertEqual(out, {"beats": []})
        self.assertTrue(seen["url"].endswith("/chat/completions"))
        self.assertEqual(seen["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(seen["payload"]["temperature"], 0.6)
        self.assertEqual(seen["payload"]["messages"][0], {"role": "system", "content": "sys"})

    async def test_http_error_and_empty_content_raise_llmerror(self):
        settings.meta_api_key = "test-key"

        def make_client(status, content):
            class Resp:
                status_code = status
                text = "boom"

                def json(self):
                    return {"choices": [{"message": {"content": content}}]}

            class Client:
                def __init__(self, *a, **kw):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return False

                async def post(self, url, json=None, headers=None):
                    return Resp()

            return Client

        with patch.object(llm.httpx, "AsyncClient", make_client(500, "{}")):
            with self.assertRaises(llm.LLMError):
                await llm.call_meta_json("s", "u")
        with patch.object(llm.httpx, "AsyncClient", make_client(200, "  ")):
            with self.assertRaises(llm.LLMError):
                await llm.call_meta_json("s", "u")
        with patch.object(llm.httpx, "AsyncClient", make_client(200, "not json")):
            with self.assertRaises(llm.LLMError):
                await llm.call_meta_json("s", "u")

    async def test_missing_key_raises_llmerror(self):
        settings.meta_api_key = ""
        with self.assertRaises(llm.LLMError):
            await llm.call_meta_json("s", "u")


class PipelineMetaTests(KeyGuard, unittest.IsolatedAsyncioTestCase):
    async def test_offline_fallback_is_deterministic_and_labelled(self):
        settings.meta_api_key = ""
        r1 = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        r2 = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        self.assertEqual(r1["output_hash"], r2["output_hash"])
        self.assertEqual([p["svg"] for p in r1["panels"]],
                         [p["svg"] for p in r2["panels"]])
        self.assertEqual(r1["consistency_avg"], 100.0)
        for agent in ("story", "layout"):
            detail = next(s["detail"] for s in r1["stages"] if s["agent"] == agent)
            self.assertTrue(detail["fallback"], detail)
            self.assertIn("model", detail)

    async def test_meta_beats_and_layout_are_used_and_labelled(self):
        settings.meta_api_key = "test-key"

        async def fake(system, user, temperature=0.6, **kw):
            if "Story Architect" in system:
                return {"beats": [
                    {"index": i, "beat_title": f"Meta Beat {i}",
                     "action": f"Mira acts in panel {i}.",
                     "dialogue": "\"Onward.\"" if i % 2 else "",
                     "narration": "Meta narration." if i == 0 else ""}
                    for i in range(6)]}
            return {"layouts": [
                {"index": i, "shot": "close", "angle": "low",
                 "pose": "reach", "expression": "determined"} for i in range(6)]}

        with patch.object(pipeline, "call_meta_json", fake):
            result = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        self.assertEqual([p["beat_title"] for p in result["panels"]],
                         [f"Meta Beat {i}" for i in range(6)])
        self.assertTrue(all(p["camera"]["shot"] == "close" for p in result["panels"]))
        self.assertTrue(all(p["layout"]["pose"] == "reach" for p in result["panels"]))
        for agent in ("story", "layout"):
            detail = next(s["detail"] for s in result["stages"] if s["agent"] == agent)
            self.assertFalse(detail["fallback"], detail)
            self.assertEqual(detail["model"], settings.meta_model)
        self.assertEqual(result["consistency_avg"], 100.0)

    async def test_invalid_layout_enum_falls_back_but_story_stays_meta(self):
        settings.meta_api_key = "test-key"

        async def fake(system, user, temperature=0.6, **kw):
            if "Story Architect" in system:
                return {"beats": [
                    {"index": i, "beat_title": f"Meta Beat {i}",
                     "action": f"Mira acts in panel {i}.", "dialogue": "", "narration": ""}
                    for i in range(6)]}
            return {"layouts": [
                {"index": i, "shot": "aerial", "angle": "low",
                 "pose": "reach", "expression": "determined"} for i in range(6)]}

        with patch.object(pipeline, "call_meta_json", fake):
            result = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        story = next(s["detail"] for s in result["stages"] if s["agent"] == "story")
        layout = next(s["detail"] for s in result["stages"] if s["agent"] == "layout")
        self.assertFalse(story["fallback"])
        self.assertTrue(layout["fallback"])
        self.assertNotIn("aerial", layout["shots"][0])

    async def test_critic_does_not_fake_repair_on_identical_rerender(self):
        # Bald characters draw no hair, so the hair-colour check legitimately
        # fails. A deterministic re-render cannot fix that: the panel must
        # keep its score/issues and must not be flagged as repaired.
        settings.meta_api_key = ""
        assets = [dict(ASSETS[0], dna={**ASSETS[0]["dna"], "form": "human",
                                        "hair": "#123456", "hair_style": "bald"}),
                  ASSETS[1]]
        result = await pipeline.run_pipeline(PREMISE, 6, 7, assets)
        self.assertTrue(all(p["issues"] for p in result["panels"]))
        self.assertTrue(all(p["consistency_score"] < 100.0 for p in result["panels"]))
        self.assertFalse(any(p["repaired"] for p in result["panels"]))
        critic = next(s for s in result["stages"] if s["agent"] == "critic")
        self.assertNotIn("repair pass", critic["summary"])

    async def test_gnome_traits_render_without_drift(self):
        # A gnome encoded from a reference image must actually draw as one:
        # pointed hat and beard in every panel, no animal ears, rosy cheeks
        # on the face (not a stripe on the torso).
        settings.meta_api_key = ""
        gnome = {"id": "g1", "kind": "character", "name": "Norman",
                 "description": "", "version": 1,
                 "dna": {"form": "creature", "build": "broad", "skin": "#f5bd8a",
                         "hair": "#fefefe", "hair_style": "long",
                         "facial_hair": "beard", "eyes": "#3f2d1a",
                         "outfit_top": "#e2aa2c", "outfit_bottom": "#5f9f6d",
                         "accent": "#d63e1d", "accessory": "tall red pointed hat",
                         "marking": "rosy pink cheeks"}}
        result = await pipeline.run_pipeline("Norman tends his garden.", 6, 42,
                                             [gnome, ASSETS[1]])
        for p in result["panels"]:
            self.assertIn('data-hat="pointed"', p["svg"])
            self.assertIn('data-beard="true"', p["svg"])
            self.assertIn('data-cheeks="true"', p["svg"])
            self.assertNotIn("-70 L -2 -54", p["svg"])  # animal ears
        self.assertEqual(result["consistency_avg"], 100.0)
        solo = await pipeline.run_pipeline("Norman tends his garden.", 6, 42, [gnome])
        self.assertEqual(solo["consistency_avg"], 100.0)  # no environment picked

    async def test_single_graphic_renders_one_image_without_story(self):
        # The default product output: ONE standalone graphic of the figure,
        # no premise, beats, or storyboard. Same seed -> identical image.
        settings.meta_api_key = ""
        gnome = {"id": "g1", "kind": "character", "name": "Norman",
                 "description": "", "version": 1,
                 "dna": {"form": "creature", "build": "broad", "skin": "#f5bd8a",
                         "hair": "#fefefe", "hair_style": "long",
                         "facial_hair": "beard", "eyes": "#3f2d1a",
                         "outfit_top": "#e2aa2c", "outfit_bottom": "#5f9f6d",
                         "accent": "#d63e1d", "accessory": "tall red pointed hat",
                         "marking": "rosy pink cheeks"}}
        g1 = await pipeline.render_single_graphic([gnome, ASSETS[1]], seed=42)
        g2 = await pipeline.render_single_graphic([gnome, ASSETS[1]], seed=42)
        self.assertEqual(g1["output_hash"], g2["output_hash"])
        self.assertEqual(g1["svg"], g2["svg"])
        self.assertEqual(g1["consistency_score"], 100.0)
        self.assertIn('data-hat="pointed"', g1["svg"])
        self.assertIn('data-beard="true"', g1["svg"])
        self.assertNotIn("beat", g1)  # a graphic, not a storyboard panel

    async def test_traits_are_generic_not_species_coded(self):
        # "gnome" in user data must not trigger anything by itself: shape
        # comes from the generic headwear_shape trait, ears from ear_shape.
        settings.meta_api_key = ""
        base = {"id": "g1", "kind": "character", "name": "Figure",
                "description": "", "version": 1,
                "dna": {"form": "creature", "skin": "#f5bd8a", "hair": "#fefefe",
                        "outfit_top": "#e2aa2c", "outfit_bottom": "#5f9f6d",
                        "accent": "#d63e1d", "accessory": "gnome hat",
                        "marking": "none"}}
        plain = await pipeline.render_single_graphic([dict(base)], seed=1)
        self.assertIn('data-hat="brim"', plain["svg"])
        self.assertNotIn('data-hat="pointed"', plain["svg"])
        self.assertNotIn("data-ears", plain["svg"])
        shaped = dict(base, dna={**base["dna"], "accessory": "red hat",
                                 "headwear_shape": "pointed",
                                 "ear_shape": "pointed"})
        fancy = await pipeline.render_single_graphic([shaped], seed=1)
        self.assertIn('data-hat="pointed"', fancy["svg"])
        self.assertIn('data-ears="pointed"', fancy["svg"])

    async def test_api_failure_and_invalid_enums_fall_back(self):
        settings.meta_api_key = "test-key"

        async def boom(system, user, temperature=0.6, **kw):
            raise llm.LLMError("upstream down")

        with patch.object(pipeline, "call_meta_json", boom):
            failed = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        settings.meta_api_key = ""
        offline = await pipeline.run_pipeline(PREMISE, 6, 7, ASSETS)
        self.assertEqual(failed["output_hash"], offline["output_hash"])
        for agent in ("story", "layout"):
            detail = next(s["detail"] for s in failed["stages"] if s["agent"] == agent)
            self.assertTrue(detail["fallback"], detail)


if __name__ == "__main__":
    unittest.main()
