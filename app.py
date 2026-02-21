import os
import logging
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi

load_dotenv()

app = Flask(__name__)

# Silence Flask request logs
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# config
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# setup gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# setup mongo (TLS fix for Linux)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot

# chats: one doc per user (_id = sender)
users = db.chats

# dedupe collection: store processed WhatsApp message IDs with TTL
processed = db.wa_processed_message_ids


def _safe_create_indexes() -> None:
    """
    Create only the indexes we actually need.
    NEVER create an index on _id manually (Mongo already has it).
    Also: do not crash app startup if index already exists / permissions restricted.
    """
    try:
        # TTL index for dedupe (expiresAt in UTC)
        processed.create_index(
            [("expiresAt", ASCENDING)],
            expireAfterSeconds=0,
            name="expiresAt_ttl",
        )
    except OperationFailure:
        # If your Mongo user can't create indexes, the bot can still run;
        # dedupe will be weaker (duplicates possible on retries).
        pass


_safe_create_indexes()


def _mark_processed_once(message_id: str) -> bool:
    """
    Returns True if this message_id is new and should be processed.
    Returns False if already processed.
    """
    if not message_id:
        # if missing, allow processing
        return True

    expires_at = datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)
    try:
        processed.insert_one({"_id": message_id, "expiresAt": expires_at})
        return True
    except DuplicateKeyError:
        return False
    except OperationFailure:
        # If we can't insert (permissions/transient), don't block processing.
        return True


def _wa_send_text(to: str, body: str) -> None:
    url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages"
    try:
        requests.post(
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
    except Exception:
        # no console logging
        pass


def get_response(uid: str, prompt: str) -> str:
    """
    Storage-lean schema:
      history: [{ "r": "user|model", "t": "..." }, ...]
    Unlimited history (as you requested).
    """
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        raw_hist = doc.get("history", [])

        gemini_history = [
            {"role": m["r"], "parts": [m["t"]]}
            for m in raw_hist
            if isinstance(m, dict) and "r" in m and "t" in m
        ]

        chat = model.start_chat(history=gemini_history)
        resp = chat.send_message(prompt)
        answer = (resp.text or "").strip() or "..."

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

    except Exception:
        return "brain error"


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

                # Ignore status updates entirely (these are the extra webhooks)
                if "statuses" in val:
                    continue

                for msg in (val.get("messages") or []):
                    # Deduplicate: only process each WhatsApp message.id once
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

    except Exception:
        # Always return 200 so Meta doesn't retry-spam you
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
