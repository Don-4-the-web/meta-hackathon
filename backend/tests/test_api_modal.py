"""Modal endpoint integration tests.

Uses an isolated temporary SQLite database; no network, credentials, or
Modal spend. This exercises train submission, status polling, and adapter
routing through FastAPI.
"""
import base64
import io
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

_temp_fd, _temp_db = tempfile.mkstemp(prefix="modal-api-test-", suffix=".db")
os.close(_temp_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_temp_db}"

from fastapi.testclient import TestClient
from PIL import Image

from app import main


def image_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (24, 24), (180, 60, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def authed_client():
    client = TestClient(main.app)
    r = client.post("/api/auth/register",
                    json={"username": "tester", "password": "secret123"})
    assert r.status_code == 201, r.text
    client.headers.update({"Authorization": f"Bearer {r.json()['token']}"})
    return client


class ModalApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = authed_client()
        cls.addClassCleanup(os.unlink, _temp_db)
        cls.addClassCleanup(main.engine.dispose)

    def test_train_status_and_adapter_render_flow(self):
        created = self.client.post("/api/assets", json={
            "kind": "character",
            "name": "Norman Test",
            "description": "A cheerful garden gnome.",
            "dna": {"facial_hair": "beard", "accessory": "red pointed hat"},
            "reference_images": [image_url(), image_url()],
        })
        self.assertEqual(created.status_code, 201, created.text)
        aid = created.json()["id"]
        self.addCleanup(lambda: self.client.delete(f"/api/assets/{aid}"))

        with patch.object(main.training, "submit_training", return_value={
            "status": "training",
            "adapter_ref": "modal://lora-studio/adapters/norman_test/",
            "call_id": "call-123",
        }):
            trained = self.client.post(f"/api/assets/{aid}/train")
        self.assertEqual(trained.status_code, 200, trained.text)
        self.assertEqual(trained.json()["call_id"], "call-123")
        asset = self.client.get(f"/api/assets/{aid}").json()
        self.assertEqual(asset["train_status"], "training")
        self.assertEqual(asset["train_call_id"], "call-123")

        with patch.object(main.training, "training_status", return_value={
            "status": "ready",
            "call_id": "call-123",
            "adapter_ref": "modal://lora-studio/adapters/norman_test/final.safetensors",
            "trigger": "norman_test_tok",
            "steps": 400,
        }), patch.object(main.validation, "validate_adapter",
                          new=AsyncMock(return_value={
                              "passed": True, "adapter_ref": "modal://x",
                              "trigger": "norman_test_tok", "probes": []})):
            status = self.client.get(f"/api/assets/{aid}/train/status")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["status"], "ready")
        asset = self.client.get(f"/api/assets/{aid}").json()
        self.assertEqual(asset["train_status"], "ready")
        self.assertTrue(asset["adapter_ref"].endswith("final.safetensors"))

        rendered_output = {
            "image": "data:image/png;base64,AAA",
            "prompt": "norman_test_tok, test",
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "source": "modal-lora",
            "profile": "tshirt",
            "checks": [],
            "bible_hash": "bible",
            "output_hash": "output",
            "seed": 7,
        }
        with patch.object(main, "generate_graphic",
                           new=AsyncMock(return_value=rendered_output)) as generate:
            rendered = self.client.post(
                f"/api/assets/{aid}/render",
                json={"profile": "tshirt", "seed": 7},
            )
        self.assertEqual(rendered.status_code, 200, rendered.text)
        self.assertEqual(rendered.json()["source"], "modal-lora")
        self.assertEqual(generate.await_args.kwargs["adapter_ref"],
                         asset["adapter_ref"])

    def test_explicit_adapter_request_requires_ready_lora(self):
        created = self.client.post("/api/assets", json={
            "kind": "character",
            "name": "Untrained Test",
            "description": "No LoRA yet.",
            "reference_images": [image_url()],
        })
        self.assertEqual(created.status_code, 201, created.text)
        aid = created.json()["id"]
        self.addCleanup(lambda: self.client.delete(f"/api/assets/{aid}"))
        denied = self.client.post(
            f"/api/assets/{aid}/render",
            json={"profile": "tshirt", "adapter": True},
        )
        self.assertEqual(denied.status_code, 409)


class AuthApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.addClassCleanup(main.engine.dispose)

    def _register(self, username):
        r = self.client.post("/api/auth/register", json={
            "username": username, "password": "secret123"})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def _authed(self, token):
        client = TestClient(main.app)
        client.headers.update({"Authorization": f"Bearer {token}"})
        return client

    def test_unauthenticated_api_is_rejected(self):
        self.assertEqual(self.client.get("/api/assets").status_code, 401)
        self.assertEqual(self.client.get("/api/conversations").status_code, 401)

    def test_register_login_me_and_duplicates(self):
        creds = self._register("alice")
        self.assertEqual(creds["user"]["username"], "alice")
        self.assertEqual(creds["user"]["trainings_used"], 0)
        dup = self.client.post("/api/auth/register", json={
            "username": "alice", "password": "otherpass"})
        self.assertEqual(dup.status_code, 409)
        bad = self.client.post("/api/auth/login", json={
            "username": "alice", "password": "wrongpass"})
        self.assertEqual(bad.status_code, 401)
        logged = self.client.post("/api/auth/login", json={
            "username": "alice", "password": "secret123"})
        self.assertEqual(logged.status_code, 200, logged.text)
        me = self._authed(logged.json()["token"]).get("/api/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "alice")

    def test_users_cannot_see_each_others_assets(self):
        alice = self._authed(self._register("alice-assets")["token"])
        bob = self._authed(self._register("bob-assets")["token"])
        created = alice.post("/api/assets", json={
            "kind": "character", "name": "Alice Private",
            "reference_images": [image_url()]})
        self.assertEqual(created.status_code, 201, created.text)
        aid = created.json()["id"]
        self.assertEqual(bob.get(f"/api/assets/{aid}").status_code, 404)
        self.assertNotIn("Alice Private",
                         [a["name"] for a in bob.get("/api/assets").json()])
        self.assertEqual(alice.get(f"/api/assets/{aid}").status_code, 200)

    def test_legacy_unowned_assets_stay_visible(self):
        from app import models
        from app.db import SessionLocal
        with SessionLocal() as s:
            legacy = models.Asset(kind="character", name="Legacy Shared",
                                  reference_image=image_url(),
                                  reference_images=[image_url()])
            s.add(legacy)
            s.commit()
            legacy_id = legacy.id
        try:
            carol = self._authed(self._register("carol-legacy")["token"])
            self.assertEqual(
                carol.get(f"/api/assets/{legacy_id}").status_code, 200)
        finally:
            with SessionLocal() as s:
                obj = s.get(models.Asset, legacy_id)
                if obj:
                    s.delete(obj)
                    s.commit()

    def test_training_quota_blocks_and_counts(self):
        token = self._register("dave-quota")["token"]
        client = self._authed(token)
        created = client.post("/api/assets", json={
            "kind": "character", "name": "Dave Char",
            "reference_images": [image_url(), image_url()]})
        aid = created.json()["id"]
        from app import models
        from app.db import SessionLocal
        with SessionLocal() as s:
            user = s.query(models.User).filter_by(username="dave-quota").one()
            user.max_trainings = 0
            s.commit()
        denied = client.post(f"/api/assets/{aid}/train", json={})
        self.assertEqual(denied.status_code, 402)
        with SessionLocal() as s:
            user = s.query(models.User).filter_by(username="dave-quota").one()
            user.max_trainings = 5
            s.commit()
        with patch.object(main.training, "submit_training", return_value={
            "status": "training", "adapter_ref": "modal://x/",
            "call_id": "call-9",
        }):
            trained = client.post(f"/api/assets/{aid}/train",
                                  json={"steps": 123})
        self.assertEqual(trained.status_code, 200, trained.text)
        me = client.get("/api/auth/me").json()
        self.assertEqual(me["trainings_used"], 1)

    def test_captions_patch_and_grade_endpoints(self):
        token = self._register("erin-caps")["token"]
        client = self._authed(token)
        created = client.post("/api/assets", json={
            "kind": "character", "name": "Erin Char",
            "reference_images": [image_url(), image_url()]})
        aid = created.json()["id"]
        ds = client.get(f"/api/assets/{aid}/dataset").json()
        self.assertIn("fingerprints", ds)
        patched = client.patch(f"/api/assets/{aid}/captions", json={
            "overrides": {ds["fingerprints"][0]: "custom caption one",
                          "deadbeefdeadbeef": "stale caption"}})
        self.assertEqual(patched.json(), {"updated": 1})
        ds2 = client.get(f"/api/assets/{aid}/dataset").json()
        self.assertEqual(ds2["captions"][0], "custom caption one")
        self.assertTrue(ds2["custom"][0])
        grade = client.get(f"/api/assets/{aid}/grade").json()
        self.assertIn("checks", grade)
        self.assertIn("distinct", grade)


if __name__ == "__main__":
    unittest.main()
