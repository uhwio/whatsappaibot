import os
import logging
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
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

users = db.chats
processed = db.processed_message_ids

# indexes
processed.create_index([("expiresAt", ASCENDING)], expireAfterSeconds=0)
users.create_index([("_id", ASCENDING)], unique=True)


def _mark_processed_once(message_id: str) -> bool:
    if not message_id:
        return True

    expires_at = datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)
    try:
        processed.insert_one({"_id": message_id, "expiresAt": expires_at})
        return True
    except DuplicateKeyError:
        return False


def _wa_send_text(to: str, body: str) -> None:
    url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages"
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


def get_response(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        raw_hist = doc.get("history", [])

        gemini_history = [
            {"role": m["r"], "parts": [m["t"]]}
            for m in raw_hist
            if "r" in m and "t" in m
        ]

        chat = model.start_chat(history=gemini_history)
        resp = chat.send_message(prompt)
        answer = (resp.text or "").strip() or "..."

        # Unlimited history (no slice cap)
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
        entries = data.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                val = change.get("value", {})

                # Ignore status updates completely
                if "statuses" in val:
                    continue

                msgs = val.get("messages") or []
                for msg in msgs:
                    mid = msg.get("id")
                    if not _mark_processed_once(mid):
                        continue

                    sender = msg.get("from")
                    mtype = msg.get("type")

                    if not sender or mtype != "text":
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
        # Always return 200 to prevent Meta retry spam
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
