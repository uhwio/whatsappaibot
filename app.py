import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi

load_dotenv()

app = Flask(__name__)

# ------------------ LOGGING (NO CONTENT / NO PHONE NOS) ------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.ERROR))
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

DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "12"))
FAIL_MESSAGE = os.getenv("FAIL_MESSAGE", "brain error")

# Workers AI model id
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/stabilityai/stable-diffusion-xl-base-1.0")

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
        processed.create_index([("expiresAt", ASCENDING)], expireAfterSeconds=0, name="expiresAt_ttl")
    except OperationFailure:
        pass


_safe_indexes()


# ------------------ DEDUPE ------------------
def _mark_processed_once(mid: Optional[str]) -> bool:
    if not mid:
        return True
    try:
        processed.insert_one(
            {"_id": mid, "expiresAt": datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)}
        )
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
            headers={"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
    except Exception as e:
        log.error("wa_send_failed: %s", type(e).__name__)


def send_text(to: str, body: str):
    _wa_send({"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body}})


def send_mode_buttons(to: str):
    _wa_send(
        {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": "What do you want to do?"},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": "IMG_MODE", "title": "🖼️ Generate image"}},
                        {"type": "reply", "reply": {"id": "CHAT_MODE", "title": "💬 Chat with AI"}},
                    ]
                },
            },
        }
    )


# ------------------ CLOUDFLARE IMAGE GEN (ROBUST) ------------------
def _short(s: str, n: int = 350) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def cf_generate_image(prompt: str) -> tuple[Optional[bytes], Optional[str]]:
    """
    Returns (image_bytes, error_text).
    - On success: (bytes, None)
    - On failure: (None, "Cloudflare AI error (code): ...")
    """
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return None, "Cloudflare not configured (missing CF_ACCOUNT_ID/CF_API_TOKEN)."

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_IMAGE_MODEL}"

    try:
        r = http.post(
            url,
            headers={
                "Authorization": f"Bearer {CF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"prompt": prompt},
            timeout=90,
        )

        # If not 2xx, capture body safely
        if r.status_code < 200 or r.status_code >= 300:
            ctype = (r.headers.get("content-type") or "").lower()
            body_snip = ""
            try:
                if "application/json" in ctype:
                    body_snip = _short(r.text)
                else:
                    # could be HTML error page
                    body_snip = _short(r.text)
            except Exception:
                body_snip = ""

            err = f"Cloudflare AI error ({r.status_code}). {body_snip}"
            log.error("cf_img_http_%s: %s", r.status_code, _short(body_snip, 250))
            return None, err

        # Success: ensure it actually looks like an image response
        ctype = (r.headers.get("content-type") or "").lower()
        if "image/" in ctype:
            return r.content, None

        # Sometimes providers return JSON even on success; handle unexpected
        if "application/json" in ctype:
            # likely a structured response you weren't expecting or an edge case
            txt = _short(r.text)
            log.error("cf_img_unexpected_json: %s", _short(txt, 250))
            return None, f"Cloudflare AI returned JSON instead of an image. {txt}"

        # Unknown content-type
        log.error("cf_img_unexpected_ctype: %s", ctype)
        return None, f"Cloudflare AI returned unexpected content-type: {ctype}"

    except requests.exceptions.Timeout:
        log.error("cf_img_timeout")
        return None, "Cloudflare AI timed out."
    except Exception as e:
        log.error("cf_img_failed: %s", type(e).__name__)
        return None, "Cloudflare AI request failed."


def wa_upload_media(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[Optional[str], Optional[str]]:
    """
    Returns (media_id, error_text)
    """
    try:
        url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/media"
        files = {"file": ("image.jpg", image_bytes, mime_type)}
        data = {"messaging_product": "whatsapp"}

        r = http.post(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, files=files, data=data, timeout=30)

        if r.status_code < 200 or r.status_code >= 300:
            err = f"WhatsApp media upload error ({r.status_code}). {_short(r.text)}"
            log.error("wa_media_http_%s", r.status_code)
            return None, err

        media_id = (r.json() or {}).get("id")
        if not media_id:
            return None, "WhatsApp media upload returned no media id."
        return media_id, None

    except Exception as e:
        log.error("wa_upload_failed: %s", type(e).__name__)
        return None, "WhatsApp media upload failed."


def send_generated_image(to: str, prompt: str):
    image_bytes, err = cf_generate_image(prompt)
    if err:
        send_text(to, err)
        return
    if not image_bytes:
        send_text(to, "Image generation failed.")
        return

    media_id, up_err = wa_upload_media(image_bytes)
    if up_err:
        send_text(to, up_err)
        return
    if not media_id:
        send_text(to, "Image upload failed.")
        return

    _wa_send(
        {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {"id": media_id, "caption": "Generated image"},
        }
    )


# ------------------ GEMINI CHAT ------------------
def get_chat_response(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        history = doc.get("history", [])

        tail = history[-CONTEXT_TURNS:] if isinstance(history, list) else []
        gemini_hist = [
            {"role": m.get("r"), "parts": [m.get("t")]}
            for m in tail
            if isinstance(m, dict) and m.get("r") in ("user", "model") and isinstance(m.get("t"), str)
        ]

        chat = model.start_chat(history=gemini_hist)
        resp = chat.send_message(prompt)
        answer = (resp.text or "").strip() or "..."

        users.update_one(
            {"_id": uid},
            {"$push": {"history": {"$each": [{"r": "user", "t": prompt}, {"r": "model", "t": answer}]}}},
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
        "make an image",
        "generate a picture",
        "create a picture",
        "draw",
        "make a picture",
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

                # ignore status updates
                if "statuses" in val:
                    continue

                for msg in val.get("messages", []):
                    mid = msg.get("id")
                    if not _mark_processed_once(mid):
                        continue

                    sender = msg.get("from")
                    if not sender:
                        continue

                    doc = users.find_one({"_id": sender}, {"mode": 1}) or {}

                    # button reply
                    if msg.get("type") == "interactive":
                        try:
                            bid = msg["interactive"]["button_reply"]["id"]
                            mode = "img" if bid == "IMG_MODE" else "chat"
                            users.update_one({"_id": sender}, {"$set": {"mode": mode}}, upsert=True)
                            send_text(sender, f"Mode set to {mode}.")
                        except Exception:
                            send_text(sender, "Could not read button reply.")
                        continue

                    if msg.get("type") != "text":
                        continue

                    txt = (msg.get("text") or {}).get("body", "").strip()
                    if not txt:
                        continue

                    # first message: show buttons once
                    if "mode" not in doc:
                        send_mode_buttons(sender)
                        continue

                    low = txt.lower()

                    # commands
                    if low.startswith("/imggen"):
                        prompt = txt[7:].strip()
                        if not prompt:
                            send_text(sender, "Usage: /imggen your prompt here")
                        else:
                            send_generated_image(sender, prompt)
                        continue

                    if low.startswith("/mode"):
                        arg = low[5:].strip()
                        if arg in ("chat", "img"):
                            users.update_one({"_id": sender}, {"$set": {"mode": arg}}, upsert=True)
                            send_text(sender, f"Mode set to {arg}.")
                        else:
                            send_text(sender, "Usage: /mode chat  OR  /mode img")
                        continue

                    # auto-detect image requests even in chat mode
                    if looks_like_image_request(txt):
                        send_generated_image(sender, txt)
                        continue

                    # mode behavior
                    if doc.get("mode") == "img":
                        send_generated_image(sender, txt)
                        continue

                    reply = get_chat_response(sender, txt)
                    send_text(sender, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error("webhook_failed: %s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
