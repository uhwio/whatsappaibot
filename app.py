import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi

load_dotenv()

app = Flask(__name__)

# ------------------ LOGGING (GUARANTEED) ------------------
# Set LOG_LEVEL=INFO (or DEBUG) in .env to see more
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "bot.log")

handlers: List[logging.Handler] = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(LOG_FILE))
except Exception:
    # If file can't be created (permissions), still log to stdout
    pass

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=handlers,
    force=True,  # <-- important: overrides any prior logging config
)

# silence request spam from werkzeug but still allow errors from our app
logging.getLogger("werkzeug").setLevel(logging.WARNING)
log = logging.getLogger("wa-bot")

log.info("BOOT: starting bot (logging active).")

# ------------------ CONFIG ------------------
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CF_API_TOKEN = os.getenv("CF_API_TOKEN")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")

MONGO_URI = os.getenv("MONGO_URI")

# WhatsApp webhook dedupe
DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# Gemini robustness
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
GEMINI_RETRY_BASE_SLEEP = float(os.getenv("GEMINI_RETRY_BASE_SLEEP", "0.9"))
GEMINI_FAIL_MESSAGE = os.getenv("GEMINI_FAIL_MESSAGE", "I’m having trouble right now. Try again in a moment.")

# Context management (keeps Mongo unlimited but Gemini context small)
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "12"))
SUMMARIZE_MIN_NEW = int(os.getenv("SUMMARIZE_MIN_NEW", "20"))
SUMMARIZE_MAX_CHUNK = int(os.getenv("SUMMARIZE_MAX_CHUNK", "60"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "3500"))

# Workers AI model id (image)
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
        log.info("BOOT: ensured TTL index for dedupe.")
    except OperationFailure as e:
        log.warning("BOOT: cannot create TTL index (permissions?). %s", type(e).__name__)


_safe_indexes()


# ------------------ UTILS ------------------
def _short(s: str, n: int = 350) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


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
        r = http.post(
            f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        if r.status_code < 200 or r.status_code >= 300:
            log.error("wa_send_http_%s: %s", r.status_code, _short(r.text))
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


# ------------------ CLOUDFLARE IMAGE GEN ------------------
def cf_generate_image(prompt: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return None, "Cloudflare not configured (missing CF_ACCOUNT_ID/CF_API_TOKEN)."

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_IMAGE_MODEL}"

    try:
        r = http.post(
            url,
            headers={"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"},
            json={"prompt": prompt},
            timeout=90,
        )

        if r.status_code < 200 or r.status_code >= 300:
            log.error("cf_img_http_%s: %s", r.status_code, _short(r.text))
            return None, f"Cloudflare AI error ({r.status_code}). {_short(r.text)}"

        ctype = (r.headers.get("content-type") or "").lower()
        if "image/" in ctype:
            return r.content, None

        log.error("cf_img_unexpected_ctype: %s body=%s", ctype, _short(r.text))
        return None, f"Cloudflare AI returned unexpected response. {ctype}"

    except requests.exceptions.Timeout:
        log.error("cf_img_timeout")
        return None, "Cloudflare AI timed out."
    except Exception as e:
        log.error("cf_img_failed: %s", type(e).__name__)
        return None, "Cloudflare AI request failed."


def wa_upload_media(image_bytes: bytes, mime_type: str = "image/jpeg") -> Tuple[Optional[str], Optional[str]]:
    try:
        url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/media"
        files = {"file": ("image.jpg", image_bytes, mime_type)}
        data = {"messaging_product": "whatsapp"}

        r = http.post(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, files=files, data=data, timeout=30)

        if r.status_code < 200 or r.status_code >= 300:
            log.error("wa_media_http_%s: %s", r.status_code, _short(r.text))
            return None, f"WhatsApp media upload error ({r.status_code}). {_short(r.text)}"

        media_id = (r.json() or {}).get("id")
        if not media_id:
            log.error("wa_media_no_id: %s", _short(r.text))
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

    media_id, up_err = wa_upload_media(image_bytes)
    if up_err:
        send_text(to, up_err)
        return

    _wa_send(
        {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {"id": media_id, "caption": "Generated image"},
        }
    )


# ------------------ GEMINI ------------------
def _call_gemini_with_retries(history: List[Dict[str, Any]], prompt: str) -> Tuple[Optional[str], Optional[str]]:
    last_msg = ""
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            chat = model.start_chat(history=history)
            resp = chat.send_message(prompt)
            text = (resp.text or "").strip()
            return (text if text else "..."), None
        except Exception as e:
            last_msg = str(e).lower()
            is_rate = ("429" in last_msg) or ("rate" in last_msg) or ("quota" in last_msg) or ("resource_exhausted" in last_msg)
            is_busy = ("503" in last_msg) or ("unavailable" in last_msg) or ("timeout" in last_msg)

            log.warning("gemini_attempt_%s_failed: %s", attempt, type(e).__name__)

            sleep_s = GEMINI_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            if is_rate or is_busy:
                sleep_s += 0.6
            time.sleep(sleep_s)

    if ("429" in last_msg) or ("rate" in last_msg) or ("quota" in last_msg) or ("resource_exhausted" in last_msg):
        return None, "RATE_LIMIT"
    return None, "OTHER"


def _format_msgs_for_summary(msgs: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    lines = []
    total = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        r = m.get("r")
        t = m.get("t")
        if r not in ("user", "model") or not isinstance(t, str):
            continue
        prefix = "USER" if r == "user" else "ASSISTANT"
        chunk = f"{prefix}: {t.strip()}\n"
        if total + len(chunk) > max_chars:
            break
        lines.append(chunk)
        total += len(chunk)
    return "".join(lines)


def _summarize_fold(existing_summary: str, new_msgs: List[Dict[str, Any]]) -> Optional[str]:
    transcript = _format_msgs_for_summary(new_msgs)

    prompt = (
        "Update the rolling conversation summary using NEW TRANSCRIPT.\n"
        "Rules: concise, info-dense, preserve decisions/preferences; no identifiers; no long quotes.\n"
        "Output ONLY updated summary.\n\n"
        f"EXISTING SUMMARY:\n{(existing_summary or '').strip()}\n\n"
        f"NEW TRANSCRIPT:\n{transcript}\n"
    )

    text, err = _call_gemini_with_retries(history=[], prompt=prompt)
    if not text:
        log.warning("summary_failed: %s", err or "UNKNOWN")
        return None

    text = text.strip()
    if len(text) > SUMMARY_MAX_CHARS:
        text = text[:SUMMARY_MAX_CHARS].rstrip() + "…"
    return text


def _maybe_update_summary(uid: str, doc: Dict[str, Any]) -> Tuple[str, int]:
    history = doc.get("history", [])
    if not isinstance(history, list):
        history = []

    summary = doc.get("summary", "")
    if not isinstance(summary, str):
        summary = ""

    summarized_upto = doc.get("summarized_upto", 0)
    if not isinstance(summarized_upto, int) or summarized_upto < 0:
        summarized_upto = 0

    total = len(history)
    if total <= CONTEXT_TURNS:
        return summary, summarized_upto

    tail_start = max(0, total - CONTEXT_TURNS)
    eligible_end = tail_start

    if summarized_upto >= eligible_end:
        return summary, summarized_upto

    new_count = eligible_end - summarized_upto
    if new_count < SUMMARIZE_MIN_NEW:
        return summary, summarized_upto

    chunk_end = min(eligible_end, summarized_upto + SUMMARIZE_MAX_CHUNK)
    chunk = history[summarized_upto:chunk_end]

    updated = _summarize_fold(summary, chunk)
    if not updated:
        return summary, summarized_upto

    try:
        users.update_one({"_id": uid}, {"$set": {"summary": updated, "summarized_upto": chunk_end}}, upsert=True)
        log.info("summary_updated: chunk_end=%s", chunk_end)
        return updated, chunk_end
    except Exception as e:
        log.error("summary_update_db_failed: %s", type(e).__name__)
        return summary, summarized_upto


def _build_gemini_history(summary: str, tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if summary and summary.strip():
        out.append({"role": "user", "parts": [f"Conversation summary (context only):\n{summary.strip()}"]})
        out.append({"role": "model", "parts": ["Understood."]})

    for m in tail:
        if not isinstance(m, dict):
            continue
        r = m.get("r")
        t = m.get("t")
        if r in ("user", "model") and isinstance(t, str) and t.strip():
            out.append({"role": r, "parts": [t]})
    return out


def get_chat_response(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1, "summary": 1, "summarized_upto": 1}) or {}
        summary, _ = _maybe_update_summary(uid, doc)

        history = doc.get("history", [])
        if not isinstance(history, list):
            history = []

        tail = history[-CONTEXT_TURNS:] if CONTEXT_TURNS > 0 else []
        gemini_hist = _build_gemini_history(summary, tail)

        text, err = _call_gemini_with_retries(gemini_hist, prompt)
        if not text:
            if err == "RATE_LIMIT":
                return "I’m getting rate-limited right now. Try again in a minute."
            return GEMINI_FAIL_MESSAGE

        users.update_one(
            {"_id": uid},
            {"$push": {"history": {"$each": [{"r": "user", "t": prompt}, {"r": "model", "t": text}]}}},
            upsert=True,
        )
        return text

    except Exception as e:
        log.error("get_chat_response_failed: %s", type(e).__name__)
        return GEMINI_FAIL_MESSAGE


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
        "turn this into an image",
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

                    doc = users.find_one({"_id": sender}, {"mode": 1}) or {}

                    if msg.get("type") == "interactive":
                        try:
                            bid = msg["interactive"]["button_reply"]["id"]
                            mode = "img" if bid == "IMG_MODE" else "chat"
                            users.update_one({"_id": sender}, {"$set": {"mode": mode}}, upsert=True)
                            send_text(sender, f"Mode set to {mode}.")
                            log.info("mode_set: %s", mode)
                        except Exception as e:
                            log.error("button_parse_failed: %s", type(e).__name__)
                            send_text(sender, "Could not read button reply.")
                        continue

                    if msg.get("type") != "text":
                        continue

                    txt = (msg.get("text") or {}).get("body", "").strip()
                    if not txt:
                        continue

                    if "mode" not in doc:
                        send_mode_buttons(sender)
                        log.info("sent_mode_buttons")
                        continue

                    low = txt.lower()

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
                            log.info("mode_set: %s", arg)
                        else:
                            send_text(sender, "Usage: /mode chat  OR  /mode img")
                        continue

                    if low == "/reset":
                        users.delete_one({"_id": sender})
                        send_text(sender, "memory wiped from db")
                        log.info("memory_reset")
                        continue

                    if looks_like_image_request(txt):
                        send_generated_image(sender, txt)
                        continue

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
