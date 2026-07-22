from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
except ImportError:  # pragma: no cover - dependency is optional in the local profile
    PasswordHasher = None  # type: ignore[assignment,misc]
    VerifyMismatchError = ValueError  # type: ignore[assignment,misc]


class AuthRepositoryMixin:
    """User and server-side session persistence for the dataset-store facade."""

    def login_or_create_user(self, *, username: str, password: str) -> dict[str, Any]:
        user_id = normalize_user_id(username)
        if not user_id:
            raise RuntimeError("Username is required.")
        if not password:
            raise RuntimeError("Password is required.")
        now = _now_iso()
        with self._connect() as connection:  # type: ignore[attr-defined]
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                password_hash, salt = _new_password_hash(password)
                cursor = connection.execute(
                    """
                    INSERT INTO users (
                        user_id, display_name, password_hash, salt, created_at, last_login_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    (user_id, username.strip(), password_hash, salt, now, now),
                )
                if cursor.rowcount > 0:
                    return {
                        "user_id": user_id,
                        "display_name": username.strip(),
                        "created": True,
                    }
                row = connection.execute(
                    "SELECT * FROM users WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("User creation failed.")

            expected_hash = str(row["password_hash"])
            salt = str(row["salt"])
            if not _verify_password(password, expected_hash, salt):
                raise RuntimeError("Invalid username or password.")
            replacement_hash, replacement_salt = _upgraded_password_hash(
                password,
                expected_hash,
                salt,
            )
            connection.execute(
                """
                UPDATE users
                SET last_login_at = ?, password_hash = ?, salt = ?
                WHERE user_id = ?
                """,
                (now, replacement_hash, replacement_salt, user_id),
            )
            return {
                "user_id": user_id,
                "display_name": str(row["display_name"]),
                "created": False,
            }

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:  # type: ignore[attr-defined]
            row = connection.execute(
                "SELECT user_id, display_name FROM users WHERE user_id = ?",
                (normalize_user_id(user_id),),
            ).fetchone()
        if row is None:
            raise RuntimeError("User was not found.")
        return {"user_id": str(row["user_id"]), "display_name": str(row["display_name"])}

    def create_user_session(
        self,
        *,
        user_id: str,
        ttl_seconds: int,
        absolute_ttl_seconds: int,
    ) -> dict[str, str]:
        self.get_user(user_id)
        now = datetime.now(UTC)
        token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        session_id = str(uuid4())
        expires_at = now + timedelta(seconds=max(300, ttl_seconds))
        absolute_expires_at = now + timedelta(seconds=max(ttl_seconds, absolute_ttl_seconds))
        with self._connect() as connection:  # type: ignore[attr-defined]
            connection.execute(
                """
                INSERT INTO user_sessions (
                    id, user_id, token_hash, csrf_hash, created_at, last_seen_at,
                    expires_at, absolute_expires_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    normalize_user_id(user_id),
                    _token_hash(token),
                    _token_hash(csrf_token),
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                    absolute_expires_at.isoformat(),
                ),
            )
        return {
            "session_id": session_id,
            "token": token,
            "csrf_token": csrf_token,
            "expires_at": expires_at.isoformat(),
        }

    def validate_user_session(
        self,
        token: str,
        *,
        csrf_token: str | None = None,
        require_csrf: bool = False,
        ttl_seconds: int = 43200,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        with self._connect() as connection:  # type: ignore[attr-defined]
            row = connection.execute(
                """
                SELECT * FROM user_sessions
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (_token_hash(token),),
            ).fetchone()
            if row is None:
                raise RuntimeError("Session is invalid or expired.")
            if now >= datetime.fromisoformat(
                str(row["expires_at"])
            ) or now >= datetime.fromisoformat(str(row["absolute_expires_at"])):
                connection.execute(
                    "UPDATE user_sessions SET revoked_at = ? WHERE id = ?",
                    (now.isoformat(), str(row["id"])),
                )
                raise RuntimeError("Session is invalid or expired.")
            if require_csrf and (
                not csrf_token
                or not hmac.compare_digest(str(row["csrf_hash"]), _token_hash(csrf_token))
            ):
                raise RuntimeError("CSRF validation failed.")
            absolute = datetime.fromisoformat(str(row["absolute_expires_at"]))
            next_expiry = min(now + timedelta(seconds=max(300, ttl_seconds)), absolute)
            connection.execute(
                "UPDATE user_sessions SET last_seen_at = ?, expires_at = ? WHERE id = ?",
                (now.isoformat(), next_expiry.isoformat(), str(row["id"])),
            )
        user = self.get_user(str(row["user_id"]))
        return user | {"session_id": str(row["id"]), "expires_at": next_expiry.isoformat()}

    def revoke_user_session(self, token: str) -> None:
        with self._connect() as connection:  # type: ignore[attr-defined]
            connection.execute(
                "UPDATE user_sessions SET revoked_at = ? WHERE token_hash = ?",
                (_now_iso(), _token_hash(token)),
            )


def normalize_user_id(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_" for char in value.strip()
    ).strip("_")
    return normalized or "default"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()


def _new_password_hash(password: str) -> tuple[str, str]:
    if PasswordHasher is not None:
        return PasswordHasher().hash(password), ""
    salt = os.urandom(16).hex()
    return _hash_password(password, salt), salt


def _verify_password(password: str, expected_hash: str, salt: str) -> bool:
    if expected_hash.startswith("$argon2") and PasswordHasher is not None:
        try:
            return bool(PasswordHasher().verify(expected_hash, password))
        except VerifyMismatchError:
            return False
    if not salt:
        return False
    return hmac.compare_digest(expected_hash, _hash_password(password, salt))


def _upgraded_password_hash(
    password: str,
    expected_hash: str,
    salt: str,
) -> tuple[str, str]:
    if PasswordHasher is None or expected_hash.startswith("$argon2"):
        return expected_hash, salt
    return PasswordHasher().hash(password), ""


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
