"""META Model API client (OpenAI-compatible, streaming).

Calls {META_API_BASE_URL}/chat/completions with the META key. There is no
demo/placeholder fallback: upstream failures raise LLMError so the API can
surface a real error to the user instead of faking a reply.
"""
import json
from collections.abc import AsyncIterator

import httpx

from .config import settings

SYSTEM_PROMPT = "You are a helpful assistant powered by a META model."


class LLMError(RuntimeError):
    pass


async def call_meta_json(system_prompt: str, user_prompt: str,
                         temperature: float = 0.6,
                         model: str | None = None) -> dict:
    """Non-streaming Meta call in JSON mode. Returns the parsed object.

    Raises LLMError on a missing key, any upstream/API error, an empty
    response, or content that is not a JSON object — callers (the story
    pipeline) treat that as the signal to use their offline fallback.
    """
    if not settings.llm_live:
        raise LLMError("META_API_KEY is not set in backend/.env")
    chosen = model or settings.meta_model
    payload = {"model": chosen,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}],
               "response_format": {"type": "json_object"},
               "temperature": temperature, "stream": False}
    headers = {"Authorization": f"Bearer {settings.meta_api_key}",
               "Content-Type": "application/json"}
    url = settings.meta_api_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            body = (resp.text or "")[:300]
            raise LLMError(f"META API returned {resp.status_code}: {body}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"META API returned an unexpected response shape: {e}") from e
        if isinstance(content, dict):
            return content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("META API returned an empty response")
        try:
            parsed = json.loads(content)
        except Exception as e:
            raise LLMError(f"META API did not return valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise LLMError("META API returned JSON that is not an object")
        return parsed
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Could not reach META API: {type(e).__name__}: {e}") from e


async def call_meta_vision_json(system_prompt: str, user_prompt: str,
                                image_data_url: str, temperature: float = 0.2,
                                model: str | None = None) -> dict:
    """Non-streaming Meta vision call in JSON mode. Returns the parsed object.

    Sends the image as an OpenAI-style `image_url` content part holding a
    data URL. Raises LLMError exactly like call_meta_json so callers can
    fall back (the asset encoder falls back to local colour extraction).
    """
    if not settings.llm_live:
        raise LLMError("META_API_KEY is not set in backend/.env")
    if not image_data_url.startswith("data:image/"):
        raise LLMError("Reference image must be an image data URL")
    chosen = model or settings.meta_model
    payload = {"model": chosen,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url",
                                 "image_url": {"url": image_data_url}}]}],
               "response_format": {"type": "json_object"},
               "temperature": temperature, "stream": False}
    headers = {"Authorization": f"Bearer {settings.meta_api_key}",
               "Content-Type": "application/json"}
    url = settings.meta_api_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            body = (resp.text or "")[:300]
            raise LLMError(f"META API returned {resp.status_code}: {body}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"META API returned an unexpected response shape: {e}") from e
        if isinstance(content, dict):
            return content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("META API returned an empty response")
        try:
            parsed = json.loads(content)
        except Exception as e:
            raise LLMError(f"META API did not return valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise LLMError("META API returned JSON that is not an object")
        return parsed
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Could not reach META API: {type(e).__name__}: {e}") from e


async def stream_chat(messages: list[dict], model: str | None = None) -> AsyncIterator[str]:
    """Yield text deltas. Raise LLMError on any upstream failure."""
    if not settings.llm_live:
        raise LLMError("META_API_KEY is not set in backend/.env")
    chosen = model or settings.meta_model
    payload = {"model": chosen, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
               "stream": True, "temperature": 0.7}
    headers = {"Authorization": f"Bearer {settings.meta_api_key}", "Content-Type": "application/json"}
    url = settings.meta_api_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"META API returned {resp.status_code}: {body}")
                got_any = False
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        continue
                    if delta:
                        got_any = True
                        yield delta
                if not got_any:
                    raise LLMError("META API returned an empty response")
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Could not reach META API: {type(e).__name__}: {e}") from e
