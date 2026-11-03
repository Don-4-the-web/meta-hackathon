"""Modal client tests. No network, credentials, or GPU spend."""
import base64
import io
import os
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from app import modal_backend, training
from tests.test_imagegen import ref_image


def dataset_zip_bytes() -> bytes:
    buf = io.BytesIO()
    raw = base64.b64decode(ref_image().split(",", 1)[1])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("000.png", raw)
        archive.writestr("000.txt", "norman_the_good_gnome_tok, a garden gnome")
    return buf.getvalue()


class FakeCall:
    def __init__(self, result=None, error=None, object_id="call-123"):
        self._result = result
        self._error = error
        self.object_id = object_id

    def get(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._result


class FakeFunction:
    def __init__(self, call=None, remote_result=None):
        self._call = call
        self.remote_result = remote_result
        self.spawned = []
        self.remoted = []

    def spawn(self, *args):
        self.spawned.append(args)
        return self._call

    def remote(self, *args):
        self.remoted.append(args)
        return self.remote_result


def fake_modal(starter=None, generate=None):
    exceptions = SimpleNamespace(
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthError=type("AuthError", (Exception,), {}),
        RequestSizeError=type("RequestSizeError", (Exception,), {}),
        OutputExpiredError=type("ExpiredError", (Exception,), {}),
    )
    functions = {}

    class Functions:
        @staticmethod
        def from_name(app_name, function_name):
            assert app_name == "meta-lora-studio"
            return functions[function_name]

        @staticmethod
        def from_id(call_id):
            assert call_id == "call-123"
            return starter_call_holder["call"]

    functions["start_training"] = starter
    functions["generate_character"] = generate
    starter_call_holder = {"call": starter._call if starter else FakeCall()}
    return SimpleNamespace(
        Function=Functions, FunctionCall=Functions, exception=exceptions
    )


class ModalBackendTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_submit_uses_namespaced_trigger_for_owned_assets(self):
        queued = {
            "status": "training",
            "train_call_id": "call-123",
            "adapter_dir": "modal://lora-studio/adapters/uabcdefab_norman/",
            "image_count": 1,
        }
        starter = FakeFunction(call=FakeCall(result=queued))
        asset = {"kind": "character", "name": "Norman",
                 "user_id": "abcdefab-1234-5678-9abc-def012345678"}
        env = {
            "TRAINING_PROVIDER": "modal",
            "MODAL_TOKEN_ID": "token-id",
            "MODAL_TOKEN_SECRET": "token-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                modal_backend, "_require_modal", return_value=fake_modal(starter)
            ):
                out = modal_backend.submit_training(
                    asset, dataset_zip_bytes(), {"steps": 1000}
                )
        self.assertEqual(out["slug"], "uabcdefab_norman")
        self.assertEqual(out["trigger"], "uabcdefab_norman_tok")
        self.assertEqual(starter.spawned[0][2], "uabcdefab_norman_tok")

    def test_submit_queues_gpu_run_and_returns_call_id(self):
        queued = {
            "status": "training",
            "train_call_id": "call-123",
            "adapter_dir": "modal://lora-studio/adapters/norman/",
            "image_count": 1,
        }
        starter = FakeFunction(call=FakeCall(result=queued))
        asset = {"kind": "character", "name": "Norman"}
        env = {
            "TRAINING_PROVIDER": "modal",
            "MODAL_TOKEN_ID": "token-id",
            "MODAL_TOKEN_SECRET": "token-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                modal_backend, "_require_modal", return_value=fake_modal(starter)
            ):
                out = modal_backend.submit_training(
                    asset, dataset_zip_bytes(), {"steps": 1000}
                )
        self.assertEqual(out["status"], "training")
        self.assertEqual(out["call_id"], "call-123")
        self.assertEqual(out["steps"], 400)  # clamped to the cost-safe default
        args = starter.spawned[0]
        self.assertEqual(args[1], "norman")
        self.assertEqual(args[2], "norman_tok")

    def test_submit_requires_credentials_and_spends_nothing(self):
        with patch.dict(os.environ, {"TRAINING_PROVIDER": "modal"}, clear=True):
            with self.assertRaises(modal_backend.ModalNotConfigured):
                modal_backend.submit_training(
                    {"kind": "character", "name": "Norman"}, b"zip"
                )

    def test_training_status_pending_and_ready(self):
        pending_modal = fake_modal(FakeFunction(call=FakeCall(error=TimeoutError())))
        ready_modal = fake_modal(
            FakeFunction(
                call=FakeCall(
                    result={
                        "status": "ready",
                        "adapter_ref": "modal://lora-studio/adapters/norman/x.safetensors",
                        "trigger": "norman_tok",
                        "steps": 400,
                    }
                )
            )
        )
        env = {
            "TRAINING_PROVIDER": "modal",
            "MODAL_TOKEN_ID": "token-id",
            "MODAL_TOKEN_SECRET": "token-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                modal_backend, "_require_modal", return_value=pending_modal
            ):
                self.assertEqual(
                    modal_backend.training_status("call-123")["status"], "training"
                )
            with patch.object(
                modal_backend, "_require_modal", return_value=ready_modal
            ):
                ready = modal_backend.training_status("call-123")
        self.assertEqual(ready["status"], "ready")
        self.assertTrue(ready["adapter_ref"].endswith("x.safetensors"))

    def test_generate_with_adapter_returns_png_data_url(self):
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
        generate = FakeFunction(
            remote_result={"image_base64": payload, "model": "base", "source": "modal-lora"}
        )
        env = {
            "TRAINING_PROVIDER": "modal",
            "MODAL_TOKEN_ID": "token-id",
            "MODAL_TOKEN_SECRET": "token-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                modal_backend, "_require_modal",
                return_value=fake_modal(generate=generate),
            ):
                out = modal_backend.generate_with_adapter(
                    "modal://lora-studio/adapters/norman/x.safetensors",
                    "norman_tok, a garden gnome",
                    seed=7,
                )
        self.assertEqual(out["source"], "modal-lora")
        self.assertTrue(out["image"].startswith("data:image/png;base64,"))
        self.assertEqual(generate.remoted[0][1], "norman_tok, a garden gnome")

    def test_credentials_detect_username_named_toml_profile(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(
                "[don-martirez]\n"
                'token_id = "id"\n'
                'token_secret = "secret"\n'
            )
            path = handle.name
        self.addCleanup(os.unlink, path)
        with patch.dict(
            os.environ,
            {"MODAL_CONFIG_PATH": path, "TRAINING_PROVIDER": "modal"},
            clear=True,
        ):
            self.assertTrue(modal_backend.modal_credentials_present())

    def test_provider_dispatch_keeps_gcp_path(self):
        with patch.object(training, "training_provider", return_value="gcp"):
            with patch.object(training, "gcp_configured", return_value=False):
                with self.assertRaises(training.TrainingNotConfigured):
                    training.submit_training({"kind": "character"}, b"zip")

    def test_provider_dispatch_delegates_to_modal(self):
        sentinel = {"status": "training"}
        with patch.object(training, "training_provider", return_value="modal"):
            with patch.object(
                training.modal_backend, "submit_training", return_value=sentinel
            ) as submit:
                out = training.submit_training(
                    {"kind": "character"}, b"zip", {"steps": 123}
                )
        self.assertIs(out, sentinel)
        submit.assert_called_once_with({"kind": "character"}, b"zip", {"steps": 123})

    def test_non_modal_status_and_generation_are_unavailable(self):
        with patch.object(training, "training_provider", return_value="gcp"):
            self.assertEqual(
                training.training_status({"train_status": "training"}, "call"),
                {"status": "training", "call_id": "call"},
            )
            with self.assertRaises(training.TrainingNotConfigured):
                training.generate_with_adapter("modal://x", "prompt")


if __name__ == "__main__":
    unittest.main()
