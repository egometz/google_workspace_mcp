"""
Tests for auth.oauth_state_store (persistent per-file OAuth state store).

These tests run without the fastmcp / google-auth dependencies and exercise
the full store/consume/TTL/replay-protection/cleanup surface.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from auth.oauth_state_store import (
    OAuthStateStore,
    _deserialize_state_entry,
    _serialize_state_entry,
    _state_file_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEX_STATE = "aabbccddeeff0011"
_HEX_STATE_2 = "1122334455667788"
_HEX_STATE_3 = "cafebabecafebabe"


def _make_store(tmp_path) -> OAuthStateStore:
    return OAuthStateStore(state_dir=str(tmp_path / "states"))


# ---------------------------------------------------------------------------
# Basic store / consume round-trip
# ---------------------------------------------------------------------------


def test_store_and_consume(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, session_id="s1", code_verifier="cv1")
    result = store.consume(_HEX_STATE)
    assert result is not None
    assert result["session_id"] == "s1"
    assert result["code_verifier"] == "cv1"


def test_state_file_is_created_with_correct_name(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv")
    state_dir = str(tmp_path / "states")
    assert os.path.exists(os.path.join(state_dir, f"{_HEX_STATE}.json"))


def test_state_file_removed_after_consume(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv")
    store.consume(_HEX_STATE)
    state_dir = str(tmp_path / "states")
    assert not os.path.exists(os.path.join(state_dir, f"{_HEX_STATE}.json"))


def test_consume_nonexistent_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.consume(_HEX_STATE) is None


# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------


def test_double_consume_returns_none(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv")
    assert store.consume(_HEX_STATE) is not None
    assert store.consume(_HEX_STATE) is None


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_expired_state_returns_none(tmp_path):
    """State that was stored with expires_in_seconds=0 is immediately expired."""
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv", expires_in_seconds=0)
    # expires_at is set to now+0; by the time consume() runs it is past
    result = store.consume(_HEX_STATE)
    assert result is None


def test_expired_file_is_removed_on_consume(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv", expires_in_seconds=0)
    store.consume(_HEX_STATE)
    state_dir = str(tmp_path / "states")
    assert not os.path.exists(os.path.join(state_dir, f"{_HEX_STATE}.json"))


def test_cleanup_removes_expired_on_next_write(tmp_path):
    """Expired files are cleaned up opportunistically when a new state is stored."""
    store = _make_store(tmp_path)
    # Write an already-expired state manually
    state_dir = str(tmp_path / "states")
    os.makedirs(state_dir, mode=0o700, exist_ok=True)
    past = datetime.now(timezone.utc) - timedelta(minutes=15)
    expired_data = {
        "session_id": None,
        "code_verifier": "old",
        "created_at": (past - timedelta(seconds=1)).isoformat(),
        "expires_at": past.isoformat(),
    }
    with open(os.path.join(state_dir, f"{_HEX_STATE}.json"), "w") as f:
        json.dump(expired_data, f)

    # A new write triggers cleanup
    store.store(_HEX_STATE_2, code_verifier="new")
    assert not os.path.exists(os.path.join(state_dir, f"{_HEX_STATE}.json"))
    assert os.path.exists(os.path.join(state_dir, f"{_HEX_STATE_2}.json"))


# ---------------------------------------------------------------------------
# Cross-instance persistence (simulates replica swap / container restart)
# ---------------------------------------------------------------------------


def test_state_readable_by_different_store_instance(tmp_path):
    """Two OAuthStateStore instances pointed at the same dir share state."""
    state_dir = str(tmp_path / "states")
    store_a = OAuthStateStore(state_dir=state_dir)
    store_b = OAuthStateStore(state_dir=state_dir)

    store_a.store(_HEX_STATE, session_id="sess", code_verifier="cv-cross")
    result = store_b.consume(_HEX_STATE)
    assert result is not None
    assert result["code_verifier"] == "cv-cross"


# ---------------------------------------------------------------------------
# consume_latest
# ---------------------------------------------------------------------------


def test_consume_latest_returns_most_recent(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, session_id="s", code_verifier="first")
    # Slight delay so created_at differs
    time.sleep(0.05)
    store.store(_HEX_STATE_2, session_id="s", code_verifier="second")
    result = store.consume_latest(session_id="s")
    assert result is not None
    assert result["code_verifier"] == "second"


def test_consume_latest_filters_by_session_id(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, session_id="sessA", code_verifier="cvA")
    store.store(_HEX_STATE_2, session_id="sessB", code_verifier="cvB")
    result = store.consume_latest(session_id="sessA")
    assert result is not None
    assert result["code_verifier"] == "cvA"
    # sessB state is untouched
    result_b = store.consume_latest(session_id="sessB")
    assert result_b is not None
    assert result_b["code_verifier"] == "cvB"


def test_consume_latest_no_session_filter_returns_any(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, session_id="sessX", code_verifier="cvX")
    result = store.consume_latest(session_id=None)
    assert result is not None


def test_consume_latest_empty_dir_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.consume_latest() is None


def test_consume_latest_removes_winner_file(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv")
    store.consume_latest()
    state_dir = str(tmp_path / "states")
    assert not os.path.exists(os.path.join(state_dir, f"{_HEX_STATE}.json"))


# ---------------------------------------------------------------------------
# Atomic write: no .tmp files left behind on success
# ---------------------------------------------------------------------------


def test_no_tmp_files_left_after_store(tmp_path):
    store = _make_store(tmp_path)
    store.store(_HEX_STATE, code_verifier="cv")
    state_dir = str(tmp_path / "states")
    tmp_files = [f for f in os.listdir(state_dir) if f.endswith(".tmp")]
    assert tmp_files == []


# ---------------------------------------------------------------------------
# Serialise / deserialise helpers
# ---------------------------------------------------------------------------


def test_serialize_round_trip():
    now = datetime.now(timezone.utc)
    original = {
        "session_id": "s",
        "code_verifier": "cv",
        "created_at": now,
        "expires_at": now + timedelta(minutes=10),
    }
    serialised = _serialize_state_entry(original)
    # Should be strings now
    assert isinstance(serialised["created_at"], str)
    assert isinstance(serialised["expires_at"], str)

    restored = _deserialize_state_entry(serialised)
    # Round-trips should be close (within 1 second)
    assert abs((restored["created_at"] - now).total_seconds()) < 1
    assert restored["session_id"] == "s"
    assert restored["code_verifier"] == "cv"


def test_deserialize_handles_invalid_timestamp():
    result = _deserialize_state_entry(
        {"created_at": "not-a-date", "expires_at": "also-bad"}
    )
    assert result["created_at"] is None
    assert result["expires_at"] is None


def test_deserialize_handles_naive_timestamp():
    """Naive ISO strings are treated as UTC."""
    result = _deserialize_state_entry({"created_at": "2026-04-21T12:00:00"})
    assert result["created_at"].tzinfo is not None


# ---------------------------------------------------------------------------
# Security: invalid state token format is rejected
# ---------------------------------------------------------------------------


def test_invalid_state_token_raises_on_file_path():
    with pytest.raises(ValueError, match="unexpected characters"):
        _state_file_path("../evil/path", "/tmp/states")


def test_consume_invalid_state_returns_none(tmp_path):
    store = _make_store(tmp_path)
    # Non-hex state is caught and returns None, not a crash
    assert store.consume("not-hex-at-all!") is None


# ---------------------------------------------------------------------------
# Directory permissions
# ---------------------------------------------------------------------------


def test_state_dir_created_with_0700(tmp_path):
    state_dir = str(tmp_path / "newdir" / "states")
    store = OAuthStateStore(state_dir=state_dir)
    store.store(_HEX_STATE, code_verifier="cv")
    mode = oct(os.stat(state_dir).st_mode)[-3:]
    assert mode == "700"
