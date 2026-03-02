import os
import logging
import time
import base64
import urllib.parse
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

# ------------------ LOGGING ------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "bot.log")

handlers: List[logging.Handler] = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(LOG_FILE))
except Exception:
    pass

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=handlers,
    force=True,
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
log = logging.getLogger("wa-bot")
log.info("BOOT: starting bot (logging active).")

# ------------------ CONFIG ------------------
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Pollinations
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1024"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "1024"))

POLLINATIONS_VIDEO_MODEL = os.getenv("POLLINATIONS_VIDEO_MODEL", "grok-video")
POLLINATIONS_VIDEO_DURATION = int(os.getenv("POLLINATIONS_VIDEO_DURATION", "6"))  # 1-10 typical
POLLINATIONS_VIDEO_ASPECT_RATIO = os.getenv("POLLINATIONS_VIDEO_ASPECT_RATIO", "16:9")  # "16:9" or "9:16"
POLLINATIONS_VIDEO_AUDIO = os.getenv("POLLINATIONS_VIDEO_AUDIO", "false").lower() in ("1", "true", "yes", "on")

# dedupe
DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# Gemini (bounded context)
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))

# Gemini retries (circuit breaker handles ResourceExhausted)
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BASE_SLEEP = float(os.getenv("GEMINI_RETRY_BASE_SLEEP", "1.0"))

RATE_LIMIT_MESSAGE = os.getenv(
    "RATE_LIMIT_MESSAGE",
    "I’m rate-limited right now. Try again in {seconds}s.",
)
GENERIC_FAIL_MESSAGE = os.getenv(
    "GEMINI_FAIL_MESSAGE",
    "I’m having trouble right now. Try again in a moment.",
)

# Circuit breaker (global)
CB_BASE_COOLDOWN = int(os.getenv("CB_BASE_COOLDOWN", "60"))
CB_MAX_COOLDOWN = int(os.getenv("CB_MAX_COOLDOWN", "600"))
CB_BACKOFF_FACTOR = float(os.getenv("CB_BACKOFF_FACTOR", "2.0"))

# Per-user cooldown
USER_COOLDOWN_SECONDS = float(os.getenv("USER_COOLDOWN_SECONDS", "2.0"))

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
        log.warning("BOOT: cannot create TTL index. %s", type(e).__name__)


_safe_indexes()

# ------------------ CIRCUIT BREAKER STATE ------------------
_cb_until = 0.0
_cb_cooldown = float(CB_BASE_COOLDOWN)


def _cb_is_open() -> bool:
    return time.time() < _cb_until


def _cb_remaining() -> int:
    rem = int(_cb_until - time.time())
    return rem if rem > 0 else 0


def _cb_trip():
    global _cb_until, _cb_cooldown
    now = time.time()
    _cb_until = now + _cb_cooldown
    log.warning("circuit_breaker_open: %ss", int(_cb_cooldown))
    _cb_cooldown = min(float(CB_MAX_COOLDOWN), _cb_cooldown * CB_BACKOFF_FACTOR)


def _cb_reset():
    global _cb_cooldown
    _cb_cooldown = max(float(CB_BASE_COOLDOWN), _cb_cooldown / CB_BACKOFF_FACTOR)


# ------------------ UTILS ------------------
def _short(s: str, n: int = 350) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def _looks_like_jpeg(b: bytes) -> bool:
    return len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8


def _looks_like_png(b: bytes) -> bool:
    return len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n"


def _looks_like_mp4(b: bytes) -> bool:
    # MP4 often contains 'ftyp' at byte 4
    return len(b) >= 12 and b[4:8] == b"ftyp"


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
    # 3 buttons: image / video / chat
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
                        {"type": "reply", "reply": {"id": "IMG_MODE", "title": "🖼️ Image"}},
                        {"type": "reply", "reply": {"id": "VID_MODE", "title": "🎬 Video"}},
                        {"type": "reply", "reply": {"id": "CHAT_MODE", "title": "💬 Chat"}},
                    ]
                },
            },
        }
    )


def wa_upload_media(media_bytes: bytes, filename: str, mime_type: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Upload media to WhatsApp to get a media_id.
    """
    try:
        url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/media"
        files = {"file": (filename, media_bytes, mime_type)}
        data = {"messaging_product": "whatsapp"}

        r = http.post(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, files=files, data=data, timeout=60)

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


def send_image_by_media_id(to: str, media_id: str, caption: str = "Generated image"):
    _wa_send(
        {"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id, "caption": caption}}
    )


def send_video_by_media_id(to: str, media_id: str, caption: str = "Generated video"):
    _wa_send(
        {"messaging_product": "whatsapp", "to": to, "type": "video", "video": {"id": media_id, "caption": caption}}
    )


# ------------------ POLLINATIONS MEDIA GEN (image OR video) ------------------
def pollinations_generate_media(
    prompt: str,
    model_name: str,
    width: int,
    height: int,
    *,
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    audio: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Returns (bytes, mime_type, error_text).

    Uses Pollinations unified endpoint:
      GET https://gen.pollinations.ai/image/{prompt}?model=...&width=...&height=...&duration=...&aspectRatio=...&audio=...
    Can return:
      - image/jpeg or image/png
      - video/mp4
      - or application/json with base64 in result.image (sometimes)
    """
    if not POLLINATIONS_API_KEY:
        return None, None, "Pollinations not configured (missing POLLINATIONS_API_KEY)."

    safe_prompt = urllib.parse.quote(prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{safe_prompt}"

    params: Dict[str, Any] = {
        "key": POLLINATIONS_API_KEY,
        "model": model_name,
        "width": int(width),
        "height": int(height),
    }
    if seed is not None:
        params["seed"] = int(seed)

    # video params
    if duration is not None:
        params["duration"] = int(duration)
    if aspect_ratio is not None:
        params["aspectRatio"] = aspect_ratio
    if audio is not None:
        params["audio"] = "true" if audio else "false"

    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}

    try:
        r = http.get(url, params=params, headers=headers, timeout=180)

        if r.status_code < 200 or r.status_code >= 300:
            log.error("poll_media_http_%s: %s", r.status_code, _short(r.text))
            return None, None, f"Pollinations error ({r.status_code}). {_short(r.text)}"

        ctype = (r.headers.get("content-type") or "").lower()

        # binary image/video
        if "image/" in ctype or "video/" in ctype:
            return r.content, ctype.split(";")[0], None

        # JSON base64 (defensive)
        if "application/json" in ctype:
            try:
                data = r.json()
                b64 = (data.get("result") or {}).get("image")
                if isinstance(b64, str) and b64:
                    blob = base64.b64decode(b64)
                    # best-effort type sniff
                    if _looks_like_mp4(blob):
                        return blob, "video/mp4", None
                    if _looks_like_png(blob):
                        return blob, "image/png", None
                    if _looks_like_jpeg(blob):
                        return blob, "image/jpeg", None
                    return blob, "application/octet-stream", None
                return None, None, f"Pollinations returned JSON but no result.image field. {_short(r.text)}"
            except Exception:
                return None, None, f"Pollinations JSON parse failed. {_short(r.text)}"

        # Unknown but maybe bytes are still media
        blob = r.content
        if _looks_like_mp4(blob):
            return blob, "video/mp4", None
        if _looks_like_png(blob):
            return blob, "image/png", None
        if _looks_like_jpeg(blob):
            return blob, "image/jpeg", None

        log.error("poll_media_unexpected_ctype: %s body=%s", ctype, _short(r.text))
        return None, None, f"Pollinations returned unexpected content-type: {ctype}"

    except requests.exceptions.Timeout:
        log.error("poll_media_timeout")
        return None, None, "Pollinations timed out."
    except Exception as e:
        log.error("poll_media_failed: %s", type(e).__name__)
        return None, None, "Pollinations request failed."


def send_generated_image(to: str, prompt: str):
    blob, mime, err = pollinations_generate_media(
        prompt=prompt,
        model_name=POLLINATIONS_IMAGE_MODEL,
        width=POLLINATIONS_WIDTH,
        height=POLLINATIONS_HEIGHT,
    )
    if err:
        send_text(to, err)
        return
    if not blob or not mime:
        send_text(to, "Image generation failed.")
        return

    filename = "image.png" if mime == "image/png" else "image.jpg"
    media_id, up_err = wa_upload_media(blob, filename=filename, mime_type=mime)
    if up_err:
        send_text(to, up_err)
        return
    send_image_by_media_id(to, media_id)


def send_generated_video(to: str, prompt: str):
    blob, mime, err = pollinations_generate_media(
        prompt=prompt,
        model_name=POLLINATIONS_VIDEO_MODEL,  # grok-video by default
        width=POLLINATIONS_WIDTH,
        height=POLLINATIONS_HEIGHT,
        duration=POLLINATIONS_VIDEO_DURATION,
        aspect_ratio=POLLINATIONS_VIDEO_ASPECT_RATIO,
        audio=POLLINATIONS_VIDEO_AUDIO,
    )
    if err:
        send_text(to, err)
        return
    if not blob or not mime or mime != "video/mp4":
        # If Pollinations returned something else, show the mime for debugging
        send_text(to, f"Video generation failed (got {mime or 'no mime'}).")
        return

    media_id, up_err = wa_upload_media(blob, filename="video.mp4", mime_type="video/mp4")
    if up_err:
        send_text(to, up_err)
        return
    send_video_by_media_id(to, media_id)


# ------------------ GEMINI ------------------
def _looks_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "resourceexhausted" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate" in msg
        or "429" in msg
    )


def _call_gemini(history: List[Dict[str, Any]], prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if _cb_is_open():
        return None, "RATE_LIMIT"

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            chat = model.start_chat(history=history)
            resp = chat.send_message(prompt)
            text = (resp.text or "").strip() or "..."
            _cb_reset()
            return text, None
        except Exception as e:
            if _looks_rate_limited(e):
                log.warning("gemini_rate_limited: %s", type(e).__name__)
                _cb_trip()
                return None, "RATE_LIMIT"

            log.warning("gemini_attempt_%s_failed: %s", attempt, type(e).__name__)
            time.sleep(GEMINI_RETRY_BASE_SLEEP * (2 ** (attempt - 1)))

    return None, "OTHER"


def _build_gemini_history(tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        history = doc.get("history", [])
        if not isinstance(history, list):
            history = []

        tail = history[-CONTEXT_TURNS:] if CONTEXT_TURNS > 0 else []
        gemini_hist = _build_gemini_history(tail)

        text, err = _call_gemini(gemini_hist, prompt)
        if not text:
            if err == "RATE_LIMIT":
                return RATE_LIMIT_MESSAGE.format(seconds=_cb_remaining() or 30)
            return GENERIC_FAIL_MESSAGE

        users.update_one(
            {"_id": uid},
            {"$push": {"history": {"$each": [{"r": "user", "t": prompt}, {"r": "model", "t": text}]}}},
            upsert=True,
        )
        return text

    except Exception as e:
        log.error("get_chat_response_failed: %s", type(e).__name__)
        return GENERIC_FAIL_MESSAGE


# ------------------ INTENT DETECTION ------------------
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


def looks_like_video_request(text: str) -> bool:
    t = text.lower()
    triggers = [
        "generate video",
        "create video",
        "make a video",
        "animate this",
        "turn this into a video",
        "video of",
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

                    # per-user cooldown
                    now = time.time()
                    doc = users.find_one({"_id": sender}, {"mode": 1, "last_user_at": 1}) or {}
                    last_user_at = doc.get("last_user_at", 0)
                    if isinstance(last_user_at, (int, float)) and (now - last_user_at) < USER_COOLDOWN_SECONDS:
                        log.info("user_cooldown_skip")
                        continue
                    users.update_one({"_id": sender}, {"$set": {"last_user_at": now}}, upsert=True)

                    # button reply
                    if msg.get("type") == "interactive":
                        try:
                            bid = msg["interactive"]["button_reply"]["id"]
                            if bid == "IMG_MODE":
                                mode = "img"
                            elif bid == "VID_MODE":
                                mode = "vid"
                            else:
                                mode = "chat"
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

                    # first message => show buttons once
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

                    if low.startswith("/vidgen"):
                        prompt = txt[7:].strip()
                        if not prompt:
                            send_text(sender, "Usage: /vidgen your prompt here")
                        else:
                            send_generated_video(sender, prompt)
                        continue

                    if low.startswith("/mode"):
                        arg = low[5:].strip()
                        if arg in ("chat", "img", "vid"):
                            users.update_one({"_id": sender}, {"$set": {"mode": arg}}, upsert=True)
                            send_text(sender, f"Mode set to {arg}.")
                        else:
                            send_text(sender, "Usage: /mode chat  OR  /mode img  OR  /mode vid")
                        continue

                    if low == "/reset":
                        users.delete_one({"_id": sender})
                        send_text(sender, "memory wiped from db")
                        continue

                    # auto intent detection (works even in chat mode)
                    if looks_like_video_request(txt):
                        send_generated_video(sender, txt)
                        continue

                    if looks_like_image_request(txt):
                        send_generated_image(sender, txt)
                        continue

                    # mode behavior
                    if doc.get("mode") == "vid":
                        send_generated_video(sender, txt)
                        continue

                    if doc.get("mode") == "img":
                        send_generated_image(sender, txt)
                        continue

                    # chat mode
                    reply = get_chat_response(sender, txt)
                    send_text(sender, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error("webhook_failed: %s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
