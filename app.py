import os
import logging
import time
import base64
import urllib.parse
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
from flask import Flask, request, jsonify, Response
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
POLLINATIONS_VIDEO_DURATION = int(os.getenv("POLLINATIONS_VIDEO_DURATION", "6"))
POLLINATIONS_VIDEO_ASPECT_RATIO = os.getenv("POLLINATIONS_VIDEO_ASPECT_RATIO", "16:9")
POLLINATIONS_VIDEO_AUDIO = os.getenv("POLLINATIONS_VIDEO_AUDIO", "false").lower() in ("1", "true", "yes", "on")

# Public URL for reference image proxy
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
REF_IMAGE_TTL_SECONDS = int(os.getenv("REF_IMAGE_TTL_SECONDS", "3600"))
REF_DIR = os.getenv("REF_DIR", "/tmp/wa_ref_images")

# dedupe
DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# Gemini
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BASE_SLEEP = float(os.getenv("GEMINI_RETRY_BASE_SLEEP", "1.0"))

RATE_LIMIT_MESSAGE = os.getenv("RATE_LIMIT_MESSAGE", "I’m rate-limited right now. Try again in {seconds}s.")
GENERIC_FAIL_MESSAGE = os.getenv("GEMINI_FAIL_MESSAGE", "I’m having trouble right now. Try again in a moment.")

# Circuit breaker (global)
CB_BASE_COOLDOWN = int(os.getenv("CB_BASE_COOLDOWN", "60"))
CB_MAX_COOLDOWN = int(os.getenv("CB_MAX_COOLDOWN", "600"))
CB_BACKOFF_FACTOR = float(os.getenv("CB_BACKOFF_FACTOR", "2.0"))

# Per-user cooldown
USER_COOLDOWN_SECONDS = float(os.getenv("USER_COOLDOWN_SECONDS", "2.0"))

# If True, image generation also uses last reference photo (image-to-image style)
USE_REF_FOR_IMG = os.getenv("USE_REF_FOR_IMG", "false").lower() in ("1", "true", "yes", "on")

# ------------------ SETUP ------------------
os.makedirs(REF_DIR, exist_ok=True)

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


def _looks_like_mp4(b: bytes) -> bool:
    return len(b) >= 12 and b[4:8] == b"ftyp"


def _guess_mime(b: bytes) -> str:
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8:
        return "image/jpeg"
    if _looks_like_mp4(b):
        return "video/mp4"
    return "application/octet-stream"


def _cleanup_old_refs():
    # best-effort cleanup
    now = time.time()
    try:
        for fn in os.listdir(REF_DIR):
            p = os.path.join(REF_DIR, fn)
            try:
                st = os.stat(p)
                if now - st.st_mtime > REF_IMAGE_TTL_SECONDS:
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


# ------------------ DEDUPE ------------------
def _mark_processed_once(mid: Optional[str]) -> bool:
    if not mid:
        return True
    try:
        processed.insert_one({"_id": mid, "expiresAt": datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)})
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
                        {"type": "reply", "reply": {"id": "IMG_MODE", "title": "🖼️ Image"}},
                        {"type": "reply", "reply": {"id": "VID_MODE", "title": "🎬 Video"}},
                        {"type": "reply", "reply": {"id": "CHAT_MODE", "title": "💬 Chat"}},
                    ]
                },
            },
        }
    )


def wa_upload_media(media_bytes: bytes, filename: str, mime_type: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/media"
        files = {"file": (filename, media_bytes, mime_type)}
        data = {"messaging_product": "whatsapp"}

        r = http.post(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, files=files, data=data, timeout=90)

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
    _wa_send({"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id, "caption": caption}})


def send_video_by_media_id(to: str, media_id: str, caption: str = "Generated video"):
    _wa_send({"messaging_product": "whatsapp", "to": to, "type": "video", "video": {"id": media_id, "caption": caption}})


# ------------------ WHATSAPP MEDIA FETCH (for reference images) ------------------
def wa_download_incoming_media(media_id: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    1) GET /{media_id} -> {url, mime_type}
    2) GET url with Authorization header -> bytes
    """
    try:
        meta = http.get(
            f"https://graph.facebook.com/v21.0/{media_id}",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            timeout=20,
        )
        if meta.status_code < 200 or meta.status_code >= 300:
            return None, None, f"WhatsApp media meta error ({meta.status_code}). {_short(meta.text)}"

        j = meta.json() or {}
        url = j.get("url")
        mime = j.get("mime_type") or "application/octet-stream"
        if not url:
            return None, None, "WhatsApp media meta returned no url."

        blob = http.get(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=60)
        if blob.status_code < 200 or blob.status_code >= 300:
            return None, None, f"WhatsApp media download error ({blob.status_code}). {_short(blob.text)}"

        return blob.content, mime, None
    except Exception as e:
        return None, None, f"WhatsApp media download failed: {type(e).__name__}"


def store_reference_image(uid: str, image_bytes: bytes) -> str:
    """
    Save to disk and store token in Mongo (tiny storage usage).
    """
    _cleanup_old_refs()
    token = secrets.token_urlsafe(20)
    path = os.path.join(REF_DIR, token)
    with open(path, "wb") as f:
        f.write(image_bytes)

    users.update_one(
        {"_id": uid},
        {"$set": {"ref_token": token, "ref_set_at": time.time()}},
        upsert=True,
    )
    return token


def get_reference_image_url(uid: str) -> Optional[str]:
    if not PUBLIC_BASE_URL:
        return None
    doc = users.find_one({"_id": uid}, {"ref_token": 1, "ref_set_at": 1}) or {}
    token = doc.get("ref_token")
    ts = doc.get("ref_set_at")
    if not token or not isinstance(token, str):
        return None
    if isinstance(ts, (int, float)) and (time.time() - ts) > REF_IMAGE_TTL_SECONDS:
        return None
    return f"{PUBLIC_BASE_URL}/ref/{token}"


@app.route("/ref/<token>", methods=["GET"])
def serve_ref(token: str):
    """
    Public endpoint for Pollinations to fetch the reference image.
    """
    _cleanup_old_refs()
    path = os.path.join(REF_DIR, token)
    if not os.path.isfile(path):
        return "Not found", 404
    try:
        with open(path, "rb") as f:
            blob = f.read()
        mime = _guess_mime(blob)
        return Response(blob, status=200, mimetype=mime)
    except Exception:
        return "Error", 500


# ------------------ POLLINATIONS MEDIA GEN ------------------
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
    image_url: Optional[str] = None,  # reference image URL
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    GET https://gen.pollinations.ai/image/{prompt}?model=...&width=...&height=...&duration=...&aspectRatio=...&audio=...&image=...
    Typically returns binary image/* or video/mp4. Some paths may return JSON with base64 in result.image.
    """
    if not POLLINATIONS_API_KEY:
        return None, None, "Pollinations not configured (missing POLLINATIONS_API_KEY)."

    safe_prompt = urllib.parse.quote(prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{safe_prompt}"

    params: Dict[str, Any] = {"model": model_name, "width": int(width), "height": int(height), "key": POLLINATIONS_API_KEY}
    if seed is not None:
        params["seed"] = int(seed)

    if duration is not None:
        params["duration"] = int(duration)
    if aspect_ratio is not None:
        params["aspectRatio"] = aspect_ratio
    if audio is not None:
        params["audio"] = "true" if audio else "false"

    if image_url:
        params["image"] = image_url  # reference image

    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}

    try:
        r = http.get(url, params=params, headers=headers, timeout=240)
        if r.status_code < 200 or r.status_code >= 300:
            return None, None, f"Pollinations error ({r.status_code}). {_short(r.text)}"

        ctype = (r.headers.get("content-type") or "").lower()

        if "image/" in ctype or "video/" in ctype:
            return r.content, ctype.split(";")[0], None

        if "application/json" in ctype:
            try:
                data = r.json()
                b64 = (data.get("result") or {}).get("image")
                if isinstance(b64, str) and b64:
                    blob = base64.b64decode(b64)
                    return blob, _guess_mime(blob), None
                return None, None, f"Pollinations returned JSON but no result.image. {_short(r.text)}"
            except Exception:
                return None, None, f"Pollinations JSON parse failed. {_short(r.text)}"

        blob = r.content
        return blob, _guess_mime(blob), None

    except requests.exceptions.Timeout:
        return None, None, "Pollinations timed out."
    except Exception as e:
        return None, None, f"Pollinations request failed: {type(e).__name__}"


def send_generated_image(to: str, prompt: str, *, ref_url: Optional[str] = None):
    send_text(to, "Generating image…")
    blob, mime, err = pollinations_generate_media(
        prompt=prompt,
        model_name=POLLINATIONS_IMAGE_MODEL,
        width=POLLINATIONS_WIDTH,
        height=POLLINATIONS_HEIGHT,
        image_url=ref_url if USE_REF_FOR_IMG else None,
    )
    if err:
        send_text(to, err)
        return
    if not blob or not mime or not mime.startswith("image/"):
        send_text(to, f"Image generation failed (got {mime or 'no mime'}).")
        return

    filename = "image.png" if mime == "image/png" else "image.jpg"
    media_id, up_err = wa_upload_media(blob, filename=filename, mime_type=mime)
    if up_err:
        send_text(to, up_err)
        return
    send_image_by_media_id(to, media_id)


def send_generated_video(to: str, prompt: str, *, ref_url: Optional[str] = None):
    send_text(to, "Generating video… (this can take a bit)")
    blob, mime, err = pollinations_generate_media(
        prompt=prompt,
        model_name=POLLINATIONS_VIDEO_MODEL,
        width=POLLINATIONS_WIDTH,
        height=POLLINATIONS_HEIGHT,
        duration=POLLINATIONS_VIDEO_DURATION,
        aspect_ratio=POLLINATIONS_VIDEO_ASPECT_RATIO,
        audio=POLLINATIONS_VIDEO_AUDIO,
        image_url=ref_url,  # reference image
    )
    if err:
        send_text(to, err)
        return
    if not blob or not mime or mime != "video/mp4":
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
    return "resourceexhausted" in msg or "resource_exhausted" in msg or "quota" in msg or "rate" in msg or "429" in msg


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
    return any(x in t for x in ["generate image", "create image", "make an image", "generate a picture", "create a picture", "draw"])


def looks_like_video_request(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in ["generate video", "create video", "make a video", "animate this", "turn this into a video", "video of"])


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
                            mode = "img" if bid == "IMG_MODE" else ("vid" if bid == "VID_MODE" else "chat")
                            users.update_one({"_id": sender}, {"$set": {"mode": mode}}, upsert=True)
                            send_text(sender, f"Mode set to {mode}.")
                        except Exception:
                            send_text(sender, "Could not read button reply.")
                        continue

                    # incoming image = set reference
                    if msg.get("type") == "image":
                        media_id = (msg.get("image") or {}).get("id")
                        if not media_id:
                            send_text(sender, "I couldn’t read that image.")
                            continue
                        img_bytes, img_mime, err = wa_download_incoming_media(media_id)
                        if err or not img_bytes:
                            send_text(sender, err or "Could not download the image.")
                            continue
                        token = store_reference_image(sender, img_bytes)
                        if PUBLIC_BASE_URL:
                            send_text(sender, "Reference photo saved ✅ Now send /vidgen <prompt> (or just type your prompt in Video mode).")
                        else:
                            send_text(
                                sender,
                                "Reference photo received ✅ BUT your server has no PUBLIC_BASE_URL set, so I can’t pass it to Pollinations. Set PUBLIC_BASE_URL first.",
                            )
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

                    ref_url = get_reference_image_url(sender)

                    low = txt.lower()

                    if low.startswith("/imggen"):
                        prompt = txt[7:].strip()
                        if not prompt:
                            send_text(sender, "Usage: /imggen your prompt here")
                        else:
                            send_generated_image(sender, prompt, ref_url=ref_url)
                        continue

                    if low.startswith("/vidgen"):
                        prompt = txt[7:].strip()
                        if not prompt:
                            send_text(sender, "Usage: /vidgen your prompt here")
                        else:
                            send_generated_video(sender, prompt, ref_url=ref_url)
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

                    # auto detection
                    if looks_like_video_request(txt):
                        send_generated_video(sender, txt, ref_url=ref_url)
                        continue

                    if looks_like_image_request(txt):
                        send_generated_image(sender, txt, ref_url=ref_url)
                        continue

                    # mode behavior
                    mode = doc.get("mode")
                    if mode == "vid":
                        send_generated_video(sender, txt, ref_url=ref_url)
                        continue

                    if mode == "img":
                        send_generated_image(sender, txt, ref_url=ref_url)
                        continue

                    # chat
                    reply = get_chat_response(sender, txt)
                    send_text(sender, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error("webhook_failed: %s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)
