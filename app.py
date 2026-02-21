import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi

load_dotenv()

app = Flask(__name__)

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
log = logging.getLogger("wa-bot")

# ------------------ CONFIG ------------------
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CF_API_TOKEN = os.getenv("CF_API_TOKEN")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")

MONGO_URI = os.getenv("MONGO_URI")

DEDUP_TTL_HOURS = 48
CONTEXT_TURNS = 12
FAIL_MESSAGE = "brain error"

# ------------------ SETUP ------------------
http = requests.Session()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot

users = db.chats
processed = db.wa_processed_message_ids


def _safe_indexes():
    try:
        processed.create_index(
            [("expiresAt", ASCENDING)],
            expireAfterSeconds=0,
            name="expiresAt_ttl",
        )
    except OperationFailure:
        pass

_safe_indexes()


# ------------------ DEDUPE ------------------
def _mark_processed_once(mid: Optional[str]) -> bool:
    if not mid:
        return True
    try:
        processed.insert_one({
            "_id": mid,
            "expiresAt": datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)
        })
        return True
    except DuplicateKeyError:
        return False
    except OperationFailure:
        return True


# ------------------ WHATSAPP SEND ------------------
def _wa_send(payload: Dict[str, Any]) -> None:
    try:
        http.post(
            f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages",
            headers={
                "Authorization": f"Bearer {WA_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
    except Exception as e:
        log.error("wa_send_failed: %s", type(e).__name__)


def send_text(to: str, body: str):
    _wa_send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body}
    })


def send_mode_buttons(to: str):
    _wa_send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "What do you want to do?"},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": "IMG_MODE", "title": "🖼️ Generate image"}
                    },
                    {
                        "type": "reply",
                        "reply": {"id": "CHAT_MODE", "title": "💬 Chat with AI"}
                    }
                ]
            }
        }
    })


# ------------------ CLOUDFLARE IMAGE GEN ------------------
def cf_generate_image(prompt: str) -> Optional[bytes]:
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"

        r = http.post(
            url,
            headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
            json={"prompt": prompt},
            timeout=90,
        )

        r.raise_for_status()
        return r.content  # raw image bytes

    except Exception as e:
        log.error("cf_img_failed: %s", type(e).__name__)
        return None


def wa_upload_media(image_bytes: bytes) -> Optional[str]:
    try:
        url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/media"

        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg"),
        }

        data = {"messaging_product": "whatsapp"}

        r = http.post(
            url,
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            files=files,
            data=data,
            timeout=30,
        )

        r.raise_for_status()
        return r.json().get("id")

    except Exception as e:
        log.error("wa_upload_failed: %s", type(e).__name__)
        return None


def send_generated_image(to: str, prompt: str):
    image_bytes = cf_generate_image(prompt)
    if not image_bytes:
        send_text(to, "Image generation failed.")
        return

    media_id = wa_upload_media(image_bytes)
    if not media_id:
        send_text(to, "Image upload failed.")
        return

    _wa_send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {
            "id": media_id,
            "caption": "Generated image"
        }
    })


# ------------------ GEMINI CHAT ------------------
def get_chat_response(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        history = doc.get("history", [])

        tail = history[-CONTEXT_TURNS:]

        gemini_hist = [
            {"role": m["r"], "parts": [m["t"]]}
            for m in tail if isinstance(m, dict)
        ]

        chat = model.start_chat(history=gemini_hist)
        resp = chat.send_message(prompt)
        answer = (resp.text or "").strip() or "..."

        users.update_one(
            {"_id": uid},
            {"$push": {"history": {
                "$each": [
                    {"r": "user", "t": prompt},
                    {"r": "model", "t": answer},
                ]
            }}},
            upsert=True,
        )

        return answer

    except Exception:
        return FAIL_MESSAGE


# ------------------ IMAGE INTENT DETECTION ------------------
def looks_like_image_request(text: str) -> bool:
    t = text.lower()
    triggers = [
        "generate image",
        "create image",
        "draw",
        "make an image",
        "create a picture",
        "generate a picture"
    ]
    return any(x in t for x in triggers)


# ------------------ WEBHOOK ------------------
@app.route("/webhook", methods=["GET"])
def verify():
    a = request.args
    if a.get("hub.mode") == "subscribe" and a.get("hub.verify_token") == VERIFY:
        return a.get("hub.challenge"), 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def inbound():
    data = request.get_json(silent=True) or {}

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                val = change.get("value", {})

                if "statuses" in val:
                    continue

                for msg in val.get("messages", []):
                    mid = msg.get("id")
                    if not _mark_processed_once(mid):
                        continue

                    sender = msg.get("from")
                    if not sender:
                        continue

                    doc = users.find_one({"_id": sender}) or {}

                    # -------- BUTTON RESPONSE --------
                    if msg.get("type") == "interactive":
                        bid = msg["interactive"]["button_reply"]["id"]
                        mode = "img" if bid == "IMG_MODE" else "chat"
                        users.update_one(
                            {"_id": sender},
                            {"$set": {"mode": mode}},
                            upsert=True,
                        )
                        send_text(sender, f"Mode set to {mode}.")
                        continue

                    if msg.get("type") != "text":
                        continue

                    txt = msg["text"]["body"].strip()

                    # -------- FIRST MESSAGE --------
                    if "mode" not in doc:
                        send_mode_buttons(sender)
                        continue

                    # -------- COMMAND --------
                    if txt.lower().startswith("/imggen"):
                        prompt = txt[7:].strip()
                        if prompt:
                            send_generated_image(sender, prompt)
                        else:
                            send_text(sender, "Usage: /imggen your prompt here")
                        continue

                    # -------- AUTO DETECT IMAGE INTENT --------
                    if looks_like_image_request(txt):
                        send_generated_image(sender, txt)
                        continue

                    # -------- IMAGE MODE --------
                    if doc.get("mode") == "img":
                        send_generated_image(sender, txt)
                        continue

                    # -------- CHAT MODE --------
                    reply = get_chat_response(sender, txt)
                    send_text(sender, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error("webhook_failed: %s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
