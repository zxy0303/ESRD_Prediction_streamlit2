"""Server-only usage tracking. Never render UI or send patient inputs.

Successful writes survive app restarts. Failed writes are retried on a later
interaction in the same session; closing that session can lose pending events.
"""

import logging
import re
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

import requests
import streamlit as st

_LOG = logging.getLogger("esrd_usage")
_STATE_KEY = "_esrd_usage_d1_v1"
_RETRY_SECONDS = 15
_MAX_PENDING = 1000
_BATCH_SIZE = 30  # 3 parameters per event, below D1's 100-parameter limit.


def _state():
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = {
            "visit_queued": False,
            "pending": [],
            "retry_after": 0.0,
            "config_warned": False,
        }
    return st.session_state[_STATE_KEY]


def _config():
    # Secrets remain in Python on the server; the API host is fixed.
    config = st.secrets["usage_tracking"]
    account_id = str(config["cloudflare_account_id"]).strip()
    database_id = str(config["d1_database_id"]).strip()
    token = str(config["cloudflare_api_token"]).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{32}", account_id):
        raise ValueError("Expected a 32-character Cloudflare account ID")
    database_id = str(UUID(database_id))
    if not token or any(c.isspace() for c in token):
        raise ValueError("Expected a Cloudflare API token")
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + account_id + "/d1/database/" + database_id + "/query"
    )
    return url, token


def _query(batch):
    # Values are bound parameters, never interpolated SQL. The INSERT trigger
    # updates the two totals in the same transaction as the event insertion.
    placeholders = ", ".join("(?, ?, ?)" for _ in batch)
    return {
        "sql": (
            "INSERT INTO esrd_usage_events (event_id, event_type, created_at) "
            "VALUES " + placeholders + " ON CONFLICT(event_id) DO NOTHING;"
        ),
        "params": [
            event[field]
            for event in batch
            for field in ("event_id", "event_type", "created_at")
        ],
    }


def _confirmed(body):
    if not isinstance(body, dict) or body.get("success") is not True or body.get("errors"):
        return False
    results = body.get("result")
    return (
        isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        and results[0].get("success") is True
    )


def _enqueue(state, event_type):
    if len(state["pending"]) >= _MAX_PENDING:
        _LOG.warning("[usage_tracking] Pending queue full; new event not recorded.")
        return
    state["pending"].append({
        "event_id": str(uuid4()),
        "event_type": event_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def _flush(state):
    if not state["pending"] or time.monotonic() < state["retry_after"]:
        return
    try:
        url, token = _config()
    except Exception:
        if not state["config_warned"]:
            _LOG.warning(
                "[usage_tracking] Not configured: check usage_tracking in "
                "Streamlit Secrets. Events are not saved yet."
            )
            state["config_warned"] = True
        state["retry_after"] = time.monotonic() + _RETRY_SECONDS
        return

    batch = list(state["pending"][:_BATCH_SIZE])
    try:
        # UUID primary keys + ignore-duplicates prevent double counting if the
        # database committed a request but its response was lost in transit.
        with requests.post(
            url,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
            json=_query(batch),
            timeout=(1.0, 2.0),
            allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                # Do not log response bodies, headers, secret keys or inputs.
                _LOG.warning(
                    "[usage_tracking] Save failed (HTTP %s); pending=%s. "
                    "Will retry on a later interaction.",
                    response.status_code, len(batch),
                )
                state["retry_after"] = time.monotonic() + _RETRY_SECONDS
                return
            if not _confirmed(response.json()):
                _LOG.warning(
                    "[usage_tracking] D1 did not confirm the write; pending=%s. "
                    "Check D1 permissions, schema, and quota; will retry later.",
                    len(batch),
                )
                state["retry_after"] = time.monotonic() + _RETRY_SECONDS
                return
        del state["pending"][:len(batch)]
        state["retry_after"] = 0.0
        state["config_warned"] = False
        _LOG.info("[usage_tracking] Saved %s event(s).", len(batch))
    except Exception as error:
        _LOG.warning(
            "[usage_tracking] Save unavailable (%s); pending=%s. "
            "Will retry on a later interaction.",
            type(error).__name__, len(batch),
        )
        state["retry_after"] = time.monotonic() + _RETRY_SECONDS


def record_visit():
    """One visit per Streamlit session; ordinary widget reruns add no visit."""
    try:
        state = _state()
        if not state["visit_queued"]:
            _enqueue(state, "visit")
            state["visit_queued"] = True
        _flush(state)
    except Exception as error:
        _LOG.warning("[usage_tracking] Visit tracking unavailable (%s).", type(error).__name__)


def record_predict_click():
    """Button callback: count a click even if the subsequent prediction fails."""
    try:
        state = _state()
        _enqueue(state, "predict_click")
        _flush(state)
    except Exception as error:
        _LOG.warning("[usage_tracking] Click tracking unavailable (%s).", type(error).__name__)
