import os
import hmac
import hashlib
import time
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError
import certifi

load_dotenv()
app = Flask(__name__)

# config
WA_TOKEN  = os.getenv("WHATSAPP_TOKEN")
PHONE_ID  = os.getenv("PHONE_NUMBER_ID")
VERIFY    = os.getenv("VERIFY_TOKEN")
API_KEY   = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# secret used ONLY to hash user ids (do NOT reuse VERIFY_TOKEN)
UID_SALT = os.getenv("UID_SALT", "change-me-please")

# gemini
genai.configure(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful WhatsApp assistant. "
    "You may be given a short 'User memory summary' that must NOT include verbatim quotes. "
    "Do not reveal private data. Keep responses concise unless asked otherwise."
)

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

# mongo
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot

# stores minimal per-user memory
users = db.user_state

# stores message ids we've already processed (TTL)
seen = db.seen_messages

# ---- One-time indexes (safe to run every boot) ----
# Dedupe: unique _id + TTL (expires docs automatically)
# expireAt must be a Date; pymongo handles TTL index on a date field.
seen.create_index("expireAt", expireAfterSeconds=0)

# Optional: expire inactive users to save storage (e.g. 30 days)
# Store lastSeen as a Date and expire with TTL index
users.create_index("lastSeen", expireAfterSeconds=30 * 24 * 3600)


def uid_hash(sender: str) -> str:
    """Hash the WhatsApp sender id so you never store the phone number."""
    digest = hmac.new(
        UID_SALT.encode("utf-8"),
        sender.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return digest


def mark_seen_once(message_id: str, ttl_seconds: int = 24 * 3600) -> bool:
    """
    Returns True if this message_id is new and should be processed.
    Returns False if already processed.
    """
    # store minimal doc with expiry time
    # using unix -> milliseconds is fine but TTL index needs a datetime object
    from datetime import datetime, timedelta
    doc = {"_id": message_id, "expireAt": datetime.utcnow() + timedelta(seconds=ttl_seconds)}
    try:
        seen.insert_one(doc)
        return True
    except DuplicateKeyError:
        return False


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def update_memory_summary(old_summary: str, new_user_text: str) -> str:
    """
    Creates a SMALL memory summary without storing the raw message.
    We do NOT persist the raw text; we only persist the result summary.
    """
    prompt = (
        "Update the User memory summary based on the new message.\n"
        "Rules:\n"
        "- Do NOT quote verbatim.\n"
        "- Do NOT store phone numbers or identifiable details.\n"
        "- Keep it short (max ~600 chars).\n\n"
        f"Current summary:\n{old_summary or '(none)'}\n\n"
        f"New user message (do not quote it back):\n{new_user_text}\n\n"
        "Return ONLY the updated summary."
    )
    resp = model.generate_content(prompt)
    return clamp_text(resp.text, 600)


def generate_reply(summary: str, user_text: str) -> str:
    """
    Generates the assistant reply. We feed summary + current text,
    but we do not store the raw user_text.
    """
    prompt = (
        f"User memory summary:\n{summary or '(none)'}\n\n"
        f"User message:\n{user_text}\n\n"
        "Reply to the user."
    )
    resp = model.generate_content(prompt)
    return (resp.text or "").strip() or "ok"


def reply_wa(target: str, body: str) -> None:
    url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages"
    try:
        res = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {WA_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "messaging_product": "whatsapp",
                "to": target,
                "type": "text",
                "text": {"body": body}
            },
            timeout=10
        )
        # no message content logs, only status codes if needed
        if res.status_code != 200:
            print("wa api fail:", res.status_code)
    except Exception as e:
        print("req error:", str(e))


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
        entry = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        val = changes.get("value") or {}

        # Ignore WhatsApp status callbacks (sent/delivered/read)
        if "statuses" in val:
            return jsonify({"status": "ignored_status"}), 200

        msgs = val.get("messages")
        if not msgs:
            return jsonify({"status": "no_message"}), 200

        msg = msgs[0]
        m_type = msg.get("type")
        if m_type != "text":
            return jsonify({"status": "ignored_non_text"}), 200

        message_id = msg.get("id")
        sender = msg.get("from")  # needed to reply, not stored
        txt = (msg.get("text") or {}).get("body", "")

        # Dedupe: process each WA message id once
        if message_id and not mark_seen_once(message_id):
            return jsonify({"status": "duplicate_ignored"}), 200

        # Hashed user id (no phone stored)
        uid = uid_hash(sender)

        # Reset command: delete only hashed user state
        if txt.strip().lower() == "/reset":
            users.delete_one({"_id": uid})
            reply_wa(sender, "memory wiped")
            return jsonify({"status": "ok"}), 200

        # Load minimal summary (small storage)
        doc = users.find_one({"_id": uid}, {"summary": 1})
        old_summary = (doc or {}).get("summary", "")

        # Update summary WITHOUT storing txt
        new_summary = update_memory_summary(old_summary, txt)

        # Generate reply
        out = generate_reply(new_summary, txt)

        # Store only summary + lastSeen
        from datetime import datetime
        users.update_one(
            {"_id": uid},
            {"$set": {"summary": new_summary, "lastSeen": datetime.utcnow()}},
            upsert=True
        )

        reply_wa(sender, out)
        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print("webhook fatal:", str(e))
        return jsonify({"status": "err"}), 500


if __name__ == "__main__":
    # IMPORTANT: avoid double-processing in dev
    app.run(port=5000, debug=False, use_reloader=False)
