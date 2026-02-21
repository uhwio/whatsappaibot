import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi

load_dotenv()

app = Flask(__name__)

# --- logging (NO message contents / NO phone numbers) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()  # ERROR by default
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.ERROR))
log = logging.getLogger("wa-bot")

# Silence Werkzeug request logs
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# config
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# Gemini robustness knobs
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BASE_SLEEP = float(os.getenv("GEMINI_RETRY_BASE_SLEEP", "0.8"))  # seconds
GEMINI_TIMEOUT_NOTE = os.getenv(
    "GEMINI_FAIL_MESSAGE",
    "brain error",
)

# HTTP session reuse
http = requests.Session()

# setup gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# setup mongo (TLS fix for Linux)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot

users = db.chats
processed = db.wa_processed_message_ids


def _safe_create_indexes() -> None:
    """
    Create only needed indexes.
    Never create _id index manually (Mongo already has it).
    """
    try:
        processed.create_index(
            [("expiresAt", ASCENDING)],
            expireAfterSeconds=0,
            name="expiresAt_ttl",
        )
    except OperationFailure:
        # If index create is not permitted, proceed anyway.
        pass


_safe_create_indexes()


def _mark_processed_once(message_id: Optional[str]) -> bool:
    """
    Returns True if new -> process it.
    False if duplicate webhook / retry of same message.
    """
    if not message_id:
        return True

    expires_at = datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)
    try:
        processed.insert_one({"_id": message_id, "expiresAt": expires_at})
        return True
    except DuplicateKeyError:
        return False
    except OperationFailure:
        # If DB insert fails transiently, don't block processing.
        return True


def _wa_send_text(to: str, body: str) -> None:
    url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages"
    try:
        http.post(
            url,
            headers={
                "Authorization": f"Bearer {WA_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": body},
            },
            timeout=12,
        )
    except Exception as e:
        # No sensitive info; ok to log the exception type only
        log.error("wa_send_text_failed: %s", type(e).__name__)


def _build_gemini_history(raw_hist) -> list:
    """
    Convert compact history [{r,t},...] to Gemini format, skipping malformed entries.
    """
    out = []
    if not isinstance(raw_hist, list):
        return out
    for m in raw_hist:
        if not isinstance(m, dict):
            continue
        r = m.get("r")
        t = m.get("t")
        if r in ("user", "model") and isinstance(t, str) and t.strip():
            out.append({"role": r, "parts": [t]})
    return out


def _call_gemini_with_retries(gemini_history: list, prompt: str) -> str:
    """
    Retries transient Gemini failures with exponential backoff.
    """
    last_err = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            chat = model.start_chat(history=gemini_history)
            resp = chat.send_message(prompt)
            text = (resp.text or "").strip()
            return text if text else "..."
        except Exception as e:
            last_err = e
            # backoff
            sleep_s = GEMINI_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            time.sleep(sleep_s)

    # After retries, raise last error to be handled by caller
    raise last_err  # type: ignore[misc]


def get_response(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        raw_hist = doc.get("history", [])

        gemini_history = _build_gemini_history(raw_hist)

        answer = _call_gemini_with_retries(gemini_history, prompt)

        # Unlimited history (no cap)
        users.update_one(
            {"_id": uid},
            {
                "$push": {
                    "history": {
                        "$each": [
                            {"r": "user", "t": prompt},
                            {"r": "model", "t": answer},
                        ]
                    }
                }
            },
            upsert=True,
        )

        return answer

    except Exception as e:
        # Log only exception type (no content / no uid)
        log.error("get_response_failed: %s", type(e).__name__)
        return GEMINI_TIMEOUT_NOTE


@app.route("/webhook", methods=["GET"])
def verify_token():
    args = request.args
    if args.get("hub.mode") == "subscribe" and args.get("hub.verify_token") == VERIFY:
        return args.get("hub.challenge"), 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def inbound():
    data = request.get_json(silent=True) or {}

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                val = change.get("value", {})

                # Ignore status webhooks
                if "statuses" in val:
                    continue

                for msg in (val.get("messages") or []):
                    mid = msg.get("id")
                    if not _mark_processed_once(mid):
                        continue

                    sender = msg.get("from")
                    if not sender:
                        continue

                    if msg.get("type") != "text":
                        continue

                    txt = (msg.get("text") or {}).get("body", "")
                    if not txt:
                        continue

                    low = txt.strip().lower()

                    if low == "/reset":
                        users.delete_one({"_id": sender})
                        _wa_send_text(sender, "memory wiped from db")
                        continue

                    if low == "ping":
                        _wa_send_text(sender, "pong")
                        continue

                    out = get_response(sender, txt)
                    _wa_send_text(sender, out)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        # Always 200 to prevent retries; log only exception type
        log.error("webhook_failed: %s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
