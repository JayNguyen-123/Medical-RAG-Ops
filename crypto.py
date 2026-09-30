# crypto.py
"""
AES-256-GCM application-level encryption for PHI-bearing cache fields.

Envelope (base64):  version(1) | key_id(4) | nonce(12) | ciphertext+tag
  * key_id lets us rotate keys: new writes use the current key, reads fall back to
    MEDICAL_ENCRYPTION_PREVIOUS_KEYS until old entries expire.
  * Every call takes an `aad` (associated data) string. It is authenticated but not
    stored, so a ciphertext copied into another tenant's / field's record fails to decrypt.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from functools import lru_cache

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from config import get_settings

_VERSION = b"\x01"
_KID_LEN = 4
_NONCE_LEN = 12


class DecryptionError(Exception):
    """Raised when a payload cannot be authenticated/decrypted (tampered, wrong key or wrong AAD)."""


def _key_id(key: bytes) -> bytes:
    return hashlib.sha256(b"kid:" + key).digest()[:_KID_LEN]


class MedicalCryptoEngine:
    def __init__(self, keys: list[bytes]):
        if not keys:
            raise ValueError("at least one encryption key is required")
        for k in keys:
            if len(k) != 32:
                raise ValueError("AES-256 keys must be 32 bytes")
        self._current_kid = _key_id(keys[0])
        self._ciphers = {_key_id(k): AESGCM(k) for k in keys}
        # Separate sub-key for keyed hashing -- never reuse the AES key for HMAC.
        self._hmac_key = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=b"medical-rag/cache-key/v1"
        ).derive(keys[0])

    def encrypt(self, plain_text: str, aad: str) -> str:
        nonce = os.urandom(_NONCE_LEN)
        ct = self._ciphers[self._current_kid].encrypt(nonce, plain_text.encode("utf-8"), aad.encode("utf-8"))
        return base64.b64encode(_VERSION + self._current_kid + nonce + ct).decode("ascii")

    def decrypt(self, blob: str | bytes, aad: str) -> str:
        try:
            raw = base64.b64decode(blob, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise DecryptionError("payload is not valid base64") from exc
        header = 1 + _KID_LEN + _NONCE_LEN
        if len(raw) <= header or raw[:1] != _VERSION:
            raise DecryptionError("unsupported or truncated envelope")
        kid, nonce, ct = raw[1:1 + _KID_LEN], raw[1 + _KID_LEN:header], raw[header:]
        cipher = self._ciphers.get(kid)
        if cipher is None:
            raise DecryptionError("payload encrypted with an unknown/retired key")
        try:
            return cipher.decrypt(nonce, ct, aad.encode("utf-8")).decode("utf-8")
        except InvalidTag as exc:
            raise DecryptionError("authentication failed (tampered payload or AAD mismatch)") from exc

    def keyed_hash(self, value: str) -> str:
        """Deterministic, non-reversible identifier (HMAC-SHA256) -- stable across processes."""
        return hmac.new(self._hmac_key, value.encode("utf-8"), hashlib.sha256).hexdigest()


@lru_cache(maxsize=1)
def get_crypto_engine() -> MedicalCryptoEngine:
    return MedicalCryptoEngine(get_settings().encryption_keys)
