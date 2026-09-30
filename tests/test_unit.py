"""Offline unit tests: no network, no external services."""
import base64
import os

import pytest

import config
from crypto import DecryptionError, MedicalCryptoEngine

KEY_A = os.urandom(32)
KEY_B = os.urandom(32)


# ------------------------------------------------------------------ config
def test_rejects_short_encryption_key():
    with pytest.raises(ValueError):
        config.decode_aes_key(base64.urlsafe_b64encode(os.urandom(16)).decode())


def test_rejects_non_base64_key():
    with pytest.raises(ValueError):
        config.decode_aes_key("not base64 at all!!")


def test_production_requires_redis_password(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    with pytest.raises(ValueError):
        config.AppSettings(_env_file=None)


def test_rejects_bad_slack_url(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://slack.com")
    with pytest.raises(ValueError):
        config.AppSettings(_env_file=None)


def test_redis_url_escapes_password(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "p@ss/word")
    s = config.AppSettings(_env_file=None)
    assert s.redis_url(1).endswith("@redis:6379/1")
    assert "p%40ss%2Fword" in s.redis_url(1)


def test_secrets_not_in_repr():
    s = config.AppSettings(_env_file=None)
    assert "sk-test-dummy" not in repr(s)


# ------------------------------------------------------------------ crypto
def test_crypto_round_trip():
    eng = MedicalCryptoEngine([KEY_A])
    blob = eng.encrypt("Labetalol 20 mg IV", aad="k|response")
    assert "Labetalol" not in blob
    assert eng.decrypt(blob, aad="k|response") == "Labetalol 20 mg IV"


def test_crypto_unique_nonce():
    eng = MedicalCryptoEngine([KEY_A])
    assert eng.encrypt("x", "a") != eng.encrypt("x", "a")


def test_crypto_aad_binding_blocks_swapping():
    eng = MedicalCryptoEngine([KEY_A])
    blob = eng.encrypt("secret", aad="cache:v1:cardiology:abc|response")
    with pytest.raises(DecryptionError):
        eng.decrypt(blob, aad="cache:v1:oncology:abc|response")


def test_crypto_tamper_detection():
    eng = MedicalCryptoEngine([KEY_A])
    raw = bytearray(base64.b64decode(eng.encrypt("secret", "a")))
    raw[-1] ^= 0x01
    with pytest.raises(DecryptionError):
        eng.decrypt(base64.b64encode(bytes(raw)).decode(), "a")


def test_crypto_key_rotation():
    old = MedicalCryptoEngine([KEY_A])
    blob = old.encrypt("legacy", "a")
    rotated = MedicalCryptoEngine([KEY_B, KEY_A])
    assert rotated.decrypt(blob, "a") == "legacy"
    with pytest.raises(DecryptionError):
        MedicalCryptoEngine([KEY_B]).decrypt(blob, "a")


def test_crypto_accepts_bytes_from_redis():
    eng = MedicalCryptoEngine([KEY_A])
    assert eng.decrypt(eng.encrypt("x", "a").encode(), "a") == "x"


def test_keyed_hash_is_deterministic_and_keyed():
    a1, a2, b = MedicalCryptoEngine([KEY_A]), MedicalCryptoEngine([KEY_A]), MedicalCryptoEngine([KEY_B])
    assert a1.keyed_hash("q") == a2.keyed_hash("q")
    assert a1.keyed_hash("q") != b.keyed_hash("q")


# ------------------------------------------------------------------ cache helpers
def test_cache_key_is_tenant_scoped_and_normalised():
    import cache
    k1 = cache.make_cache_key("What is  the dose?", "cardiology")
    k2 = cache.make_cache_key("what is the dose?", "cardiology")
    k3 = cache.make_cache_key("What is the dose?", "oncology")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("cache:v1:cardiology:")
    assert "dose" not in k1


@pytest.mark.parametrize("bad", ["", "Cardiology", "cardio logy", "x}|@*", "a", "cardiology)", None])
def test_invalid_clearance_group_rejected(bad):
    import cache
    with pytest.raises(ValueError):
        cache.validate_clearance_group(bad)


def test_knn_query_uses_tag_filter_and_knn():
    import cache
    q = cache.build_knn_query_string("cardiology")
    assert q == "(@clearance_group:{cardiology})=>[KNN 1 @query_embedding $vec AS vector_score]"


# ------------------------------------------------------------------ PHI detectors
@pytest.mark.parametrize("text,expected", [
    ("SSN 123-45-6789 on file", "ssn"),
    ("MRN: A1234567", "mrn"),
    ("DOB 11/12/1984", "dob"),
    ("call 555-123-4567", "phone"),
    ("email jane.doe@example.org", "email"),
])
def test_regex_detectors_flag_identifiers(text, expected):
    from tasks import regex_findings
    assert expected in regex_findings(text)


def test_regex_detectors_ignore_clinical_guidance():
    from tasks import regex_findings
    assert regex_findings("Labetalol 20 mg IV over 2 minutes, then 40-80 mg every 10 minutes (max 300 mg).") == []
