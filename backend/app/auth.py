"""Minimal local auth for the single-host consumer app.

Username + password (PBKDF2, stdlib only), one bearer token per user. There
are no roles: whoever has an account owns their own assets, projects, and
chats. Assets created before auth existed (user_id NULL) stay visible to
everyone as legacy shared data.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import models
from .db import get_db

bearer = HTTPBearer(auto_error=False)

_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2${_ITERATIONS}${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iters))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest)


def issue_token() -> tuple[str, str]:
    """Return (bearer_token, sha256_hex) — the raw token is shown once."""
    raw = secrets.token_urlsafe(32)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def get_current_user(db: Session = Depends(get_db),
                     creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                     ) -> models.User:
    if not creds or not creds.credentials:
        raise HTTPException(401, "Login required")
    digest = hashlib.sha256(creds.credentials.encode()).hexdigest()
    user = db.query(models.User).filter(
        models.User.token_hash == digest).first()
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user
