import os
import time
import json
import hmac
import base64
import hashlib
import secrets
import logging
import threading
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import certifi


# =========================
# Load env + app
# =========================
load_dotenv()
app = Flask(__name__)


# =========================
# Logging
# =========================
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
log.info("BOOT: started")


# =========================
# Config
# =========================
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")

MONGO_URI = os.getenv("MONGO_URI")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Kling AI
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")
KLING_BASE_URL = os.getenv("KLING_BASE_URL", "https://api-singapore.klingai.com").rstrip("/")

# These paths are isolated on purpose because Kling may change naming/versioning.
# If your account docs show different paths, only change these values.
KLING_CREATE_PATH = os.getenv("KLING_CREATE_PATH", "/v1/videos/image2video")
KLING_QUERY_PATH_TEMPLATE = os.getenv("KLING_QUERY_PATH_TEMPLATE", "/v1/videos/image2video/{task_id}")

KLING_VIDEO_MODEL = os.getenv("KLING_VIDEO_MODEL", "kling-v3-omni")
KLING_RESOLUTION = os.getenv("KLING_RESOLUTION", "1080p")
KLING_ENABLE_AUDIO = os.getenv("KLING_ENABLE_AUDIO", "false").lower() in ("1", "true", "yes", "on")
KLING_TRANSITION_DURATION = int(os.getenv("KLING_TRANSITION_DURATION", "3"))
KLING_ASPECT_RATIO = os.getenv("KLING_ASPECT_RATIO", "16:9")
KLING_MODE = os.getenv("KLING_MODE", "std")

# Photo collection behavior
PHOTO_TRANSITION_MIN_COUNT = int(os.getenv("PHOTO_TRANSITION_MIN_COUNT", "2"))
PHOTO_TRANSITION_MAX_COUNT = int(os.getenv("PHOTO_TRANSITION_MAX_COUNT", "5"))
PHOTO_TRANSITION_MAX_GAP_SECONDS = int(os.getenv("PHOTO_TRANSITION_MAX_GAP_SECONDS", "1800"))

# Public base URL for serving reference images (ngrok / public domain)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
REF_DIR = os.getenv("REF_DIR", "/tmp/wa_ref_images")
REF_IMAGE_TTL_SECONDS = int(os.getenv("REF_IMAGE_TTL_SECONDS", "3600"))

# WhatsApp webhook dedupe
DEDUP_TTL_HOURS = int(os.getenv("DEDUP_TTL_HOURS", "48"))

# Cooldowns + resilience
USER_COOLDOWN_SECONDS = float(os.getenv("USER_COOLDOWN_SECONDS", "1.5"))

# Gemini circuit breaker
CB_BASE_COOLDOWN = int(os.getenv("CB_BASE_COOLDOWN", "60"))
CB_MAX_COOLDOWN = int(os.getenv("CB_MAX_COOLDOWN", "600"))
CB_BACKOFF_FACTOR = float(os.getenv("CB_BACKOFF_FACTOR", "2.0"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BASE_SLEEP = float(os.getenv("GEMINI_RETRY_BASE_SLEEP", "1.0"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))
RATE_LIMIT_MESSAGE = os.getenv("RATE_LIMIT_MESSAGE", "I'm rate-limited right now 😅 Try again in {seconds}s.")
GENERIC_FAIL_MESSAGE = os.getenv("GEMINI_FAIL_MESSAGE", "I hit a problem just now. Try again in a moment.")

WELCOME_TEXT = (
    "Hello! 👋\n\n"
    "I can:\n"
    "🖼 Create images from text\n"
    "🎬 Create a transition video from 2 to 5 photos\n"
    "💬 Chat and answer questions\n\n"
    "How to use:\n"
    "• Send 2 to 5 photos, then send /makevideo\n"
    "• Send text and I’ll chat with you\n"
    "• Ask for an image and I’ll generate one\n\n"
    "Commands:\n"
    "• /menu      → show help\n"
    "• /limpar    → clear chat memory\n"
    "• /photos    → see how many photos are queued\n"
    "• /resetphotos → clear queued photos\n"
    "• /makevideo → generate the transition video"
)

# Image generation via Pollinations is kept for text-to-image only
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1024"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "1024"))


# =========================
# Setup
# =========================
os.makedirs(REF_DIR, exist_ok=True)
http = requests.Session()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gemini_model = None

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot
users = db.chats
processed = db.wa_processed_message_ids


def ensure_indexes():
    try:
        processed.create_index([("expiresAt", ASCENDING)], expireAfterSeconds=0, name="expiresAt_ttl")
        users.create_index([("transition_photos.ts", ASCENDING)], name="transition_photos_ts")
        log.info("BOOT: indexes ok")
    except OperationFailure as e:
        log.warning("BOOT: index creation failed: %s", type(e).__name__)


ensure_indexes()


# =========================
# Helpers
# =========================
def _short(s: str, n: int = 280) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def _looks_rate_limited(exc: Exception) -> bool:
    m = str(exc).lower()
    return ("resourceexhausted" in m) or ("resource_exhausted" in m) or ("quota" in m) or ("rate" in m) or ("429" in m)


def _guess_mime(blob: bytes) -> str:
    if len(blob) >= 8 and blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(blob) >= 2 and blob[0] == 0xFF and blob[1] == 0xD8:
        return "image/jpeg"
    if len(blob) >= 12 and blob[4:8] == b"ftyp":
        return "video/mp4"
    return "application/octet-stream"


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


# =========================
# Dedupe
# =========================
def mark_processed_once(mid: Optional[str]) -> bool:
    if not mid:
        return True
    try:
        processed.insert_one({"_id": mid, "expiresAt": datetime.now(timezone.utc) + timedelta(hours=DEDUP_TTL_HOURS)})
        return True
    except DuplicateKeyError:
        return False
    except OperationFailure:
        return True


# =========================
# WhatsApp send
# =========================
def wa_send(payload: Dict[str, Any]) -> None:
    try:
        r = http.post(
            f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=25,
        )
        if r.status_code < 200 or r.status_code >= 300:
            log.error("wa_send_http_%s: %s", r.status_code, _short(r.text))
    except Exception as e:
        log.error("wa_send_failed: %s", type(e).__name__)


def send_text(to: str, body: str):
    wa_send({"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body}})


def wa_upload_media(media_bytes: bytes, filename: str, mime_type: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = http.post(
            f"https://graph.facebook.com/v21.0/{PHONE_ID}/media",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            files={"file": (filename, media_bytes, mime_type)},
            data={"messaging_product": "whatsapp"},
            timeout=120,
        )
        if r.status_code < 200 or r.status_code >= 300:
            log.error("wa_media_http_%s: %s", r.status_code, _short(r.text))
            return None, f"Error uploading media to WhatsApp ({r.status_code}). {_short(r.text)}"
        media_id = (r.json() or {}).get("id")
        if not media_id:
            return None, "WhatsApp did not return a media id."
        return media_id, None
    except Exception as e:
        log.error("wa_upload_failed: %s", type(e).__name__)
        return None, "Failed to upload media to WhatsApp."


def send_image_by_id(to: str, media_id: str):
    wa_send({"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id, "caption": "🖼 Here is your image"}})


def send_video_by_id(to: str, media_id: str):
    wa_send({"messaging_product": "whatsapp", "to": to, "type": "video", "video": {"id": media_id, "caption": "🎬 Here is your video"}})


# =========================
# WhatsApp incoming media download
# =========================
def wa_download_incoming_media(media_id: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    try:
        meta = http.get(
            f"https://graph.facebook.com/v21.0/{media_id}",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            timeout=20,
        )
        if meta.status_code < 200 or meta.status_code >= 300:
            return None, None, f"Error getting media info ({meta.status_code}). {_short(meta.text)}"

        j = meta.json() or {}
        url = j.get("url")
        mime = j.get("mime_type") or "application/octet-stream"
        if not url:
            return None, None, "WhatsApp did not return the media URL."

        blob = http.get(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=60)
        if blob.status_code < 200 or blob.status_code >= 300:
            return None, None, f"Error downloading media ({blob.status_code}). {_short(blob.text)}"

        return blob.content, mime, None
    except Exception as e:
        return None, None, f"Failed to download media: {type(e).__name__}"


# =========================
# Reference image hosting
# =========================
def cleanup_old_refs():
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


def store_reference_blob(uid: str, image_bytes: bytes) -> str:
    cleanup_old_refs()
    token = secrets.token_urlsafe(20)
    path = os.path.join(REF_DIR, token)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return token


def store_transition_photo(uid: str, image_bytes: bytes) -> str:
    token = store_reference_blob(uid, image_bytes)
    now = time.time()

    doc = users.find_one({"_id": uid}, {"transition_photos": 1}) or {}
    photos = doc.get("transition_photos", [])
    if not isinstance(photos, list):
        photos = []

    fresh: List[Dict[str, Any]] = []
    for item in photos:
        if isinstance(item, dict) and isinstance(item.get("ts"), (int, float)):
            if now - item["ts"] <= PHOTO_TRANSITION_MAX_GAP_SECONDS:
                fresh.append(item)

    fresh.append({"token": token, "ts": now})
    fresh = fresh[-PHOTO_TRANSITION_MAX_COUNT:]

    users.update_one(
        {"_id": uid},
        {"$set": {"transition_photos": fresh}},
        upsert=True,
    )
    return token


def get_transition_photos(uid: str) -> List[Dict[str, Any]]:
    doc = users.find_one({"_id": uid}, {"transition_photos": 1}) or {}
    photos = doc.get("transition_photos", [])
    if not isinstance(photos, list):
        return []

    now = time.time()
    out = []
    for item in photos:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        ts = item.get("ts")
        if isinstance(token, str) and isinstance(ts, (int, float)):
            if now - ts <= PHOTO_TRANSITION_MAX_GAP_SECONDS:
                out.append(item)
    return out


def get_transition_photo_urls(uid: str) -> List[str]:
    if not PUBLIC_BASE_URL:
        return []

    out = []
    now = time.time()
    for item in get_transition_photos(uid):
        token = item.get("token")
        ts = item.get("ts")
        if isinstance(token, str) and isinstance(ts, (int, float)):
            if now - ts <= REF_IMAGE_TTL_SECONDS:
                out.append(f"{PUBLIC_BASE_URL}/ref/{token}")
    return out


def clear_transition_photos(uid: str):
    users.update_one({"_id": uid}, {"$unset": {"transition_photos": ""}}, upsert=True)


@app.route("/ref/<token>", methods=["GET"])
def serve_ref(token: str):
    cleanup_old_refs()
    path = os.path.join(REF_DIR, token)
    if not os.path.isfile(path):
        return "Not found", 404
    try:
        with open(path, "rb") as f:
            blob = f.read()
        return Response(blob, status=200, mimetype=_guess_mime(blob))
    except Exception:
        return "Error", 500


# =========================
# Pollinations image generation
# =========================
def pollinations_generate_image(prompt: str, width: int, height: int, timeout_seconds: int = 180) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if not POLLINATIONS_API_KEY:
        return None, None, "Pollinations not configured."

    try:
        from urllib.parse import quote
        safe_prompt = quote(prompt, safe="")
        url = f"https://gen.pollinations.ai/image/{safe_prompt}"
        params = {
            "key": POLLINATIONS_API_KEY,
            "model": POLLINATIONS_IMAGE_MODEL,
            "width": width,
            "height": height,
        }
        headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}
        r = http.get(url, params=params, headers=headers, timeout=timeout_seconds)

        if r.status_code < 200 or r.status_code >= 300:
            return None, None, f"Pollinations error ({r.status_code}). {_short(r.text)}"

        ctype = (r.headers.get("content-type") or "").lower()
        if "image/" in ctype:
            return r.content, ctype.split(";")[0], None

        if "application/json" in ctype:
            try:
                data = r.json()
                b64 = (data.get("result") or {}).get("image")
                if isinstance(b64, str) and b64:
                    blob = base64.b64decode(b64)
                    return blob, _guess_mime(blob), None
                return None, None, f"Pollinations returned JSON without image. {_short(r.text)}"
            except Exception:
                return None, None, f"Pollinations returned unreadable JSON. {_short(r.text)}"

        blob = r.content
        return blob, _guess_mime(blob), None
    except requests.exceptions.Timeout:
        return None, None, "Pollinations timed out."
    except Exception as e:
        return None, None, f"Pollinations request failed: {type(e).__name__}"


def generate_and_send_image(to: str, prompt: str):
    send_text(to, "Creating your image... 🖼")
    blob, mime, err = pollinations_generate_image(prompt, POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT)
    if err:
        send_text(to, err)
        return
    if not blob or not mime or not mime.startswith("image/"):
        send_text(to, f"Couldn't create the image (got {mime or 'unknown'}).")
        return

    filename = "image.png" if mime == "image/png" else "image.jpg"
    media_id, up_err = wa_upload_media(blob, filename=filename, mime_type=mime)
    if up_err:
        send_text(to, up_err)
        return
    send_image_by_id(to, media_id)


# =========================
# Kling AI
# =========================
def kling_is_configured() -> bool:
    return bool(KLING_ACCESS_KEY and KLING_SECRET_KEY and KLING_BASE_URL and PUBLIC_BASE_URL)


def kling_estimated_credits(photo_count: int, seconds_per_transition: int) -> int:
    transitions = max(0, photo_count - 1)
    # VIDEO 3.0 No Native Audio 1080p = 8 credits/sec
    return transitions * seconds_per_transition * 8


def kling_build_auth_headers(method: str, path: str, body: Optional[dict] = None) -> Dict[str, str]:
    """
    IMPORTANT:
    This helper isolates Kling auth because public browsing access did not expose
    the full exact signing recipe. Replace only this function if your Kling account
    docs use a different signing/token method.
    """
    timestamp = str(int(time.time()))
    body_str = json.dumps(body or {}, separators=(",", ":"), ensure_ascii=False)

    signing_string = f"{method}\n{path}\n{timestamp}\n{body_str}"
    signature = hmac.new(
        KLING_SECRET_KEY.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    token = f"{KLING_ACCESS_KEY}:{timestamp}:{signature}"

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def kling_extract_task_id(data: dict) -> Optional[str]:
    candidates = [
        data.get("task_id"),
        data.get("id"),
        (data.get("data") or {}).get("task_id") if isinstance(data.get("data"), dict) else None,
        (data.get("data") or {}).get("id") if isinstance(data.get("data"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return None


def kling_extract_status(data: dict) -> str:
    candidates = [
        data.get("status"),
        data.get("task_status"),
        (data.get("data") or {}).get("status") if isinstance(data.get("data"), dict) else None,
        (data.get("data") or {}).get("task_status") if isinstance(data.get("data"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, str):
            return c.lower().strip()
    return ""


def kling_extract_video_url(data: dict) -> Optional[str]:
    candidates = []

    if isinstance(data.get("video_url"), str):
        candidates.append(data.get("video_url"))

    d = data.get("data")
    if isinstance(d, dict):
        if isinstance(d.get("video_url"), str):
            candidates.append(d.get("video_url"))

        result = d.get("result")
        if isinstance(result, dict) and isinstance(result.get("video_url"), str):
            candidates.append(result.get("video_url"))

        works = d.get("works")
        if isinstance(works, list):
            for w in works:
                if isinstance(w, dict):
                    resource = w.get("resource")
                    if isinstance(resource, dict) and isinstance(resource.get("resource"), str):
                        candidates.append(resource.get("resource"))

    result_top = data.get("result")
    if isinstance(result_top, dict) and isinstance(result_top.get("video_url"), str):
        candidates.append(result_top.get("video_url"))

    for c in candidates:
        if c and isinstance(c, str):
            return c
    return None


def kling_create_transition_task(start_image_url: str, end_image_url: str) -> Tuple[Optional[str], Optional[str]]:
    if not kling_is_configured():
        return None, "Kling is not configured correctly. Check KLING_* and PUBLIC_BASE_URL."

    path = KLING_CREATE_PATH

    body = {
        "model_name": KLING_VIDEO_MODEL,
        "prompt": "smooth cinematic transition between the two photos, elegant motion, natural morphing, stable composition, high quality",
        "duration": KLING_TRANSITION_DURATION,
        "mode": KLING_MODE,
        "resolution": KLING_RESOLUTION,
        "aspect_ratio": KLING_ASPECT_RATIO,
        "sound": False,
        # Common pattern for start/end frame style requests:
        "image_url": [start_image_url, end_image_url],
    }

    try:
        r = http.post(
            f"{KLING_BASE_URL}{path}",
            headers=kling_build_auth_headers("POST", path, body),
            json=body,
            timeout=60,
        )
        if r.status_code < 200 or r.status_code >= 300:
            return None, f"Kling create task failed ({r.status_code}): {_short(r.text)}"

        data = r.json() or {}
        task_id = kling_extract_task_id(data)
        if not task_id:
            return None, f"Kling did not return a task id. Raw: {_short(json.dumps(data, ensure_ascii=False))}"
        return task_id, None

    except Exception as e:
        return None, f"Failed to create Kling task: {type(e).__name__}"


def kling_poll_video_result(task_id: str, timeout_seconds: int = 900) -> Tuple[Optional[bytes], Optional[str]]:
    path = KLING_QUERY_PATH_TEMPLATE.format(task_id=task_id)
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            r = http.get(
                f"{KLING_BASE_URL}{path}",
                headers=kling_build_auth_headers("GET", path, None),
                timeout=30,
            )
            if r.status_code < 200 or r.status_code >= 300:
                return None, f"Kling query failed ({r.status_code}): {_short(r.text)}"

            data = r.json() or {}
            status = kling_extract_status(data)

            if status in ("succeed", "success", "completed", "done"):
                video_url = kling_extract_video_url(data)
                if not video_url:
                    return None, "Kling completed but did not return a video URL."

                vr = http.get(video_url, timeout=180)
                if vr.status_code < 200 or vr.status_code >= 300:
                    return None, f"Failed to download Kling video ({vr.status_code})."
                return vr.content, None

            if status in ("failed", "error", "canceled", "cancelled"):
                return None, f"Kling generation failed: {_short(json.dumps(data, ensure_ascii=False))}"

            time.sleep(5)

        except Exception as e:
            return None, f"Failed while polling Kling task: {type(e).__name__}"

    return None, "Kling took too long to finish."


def concat_mp4_clips(clips: List[bytes]) -> Tuple[Optional[bytes], Optional[str]]:
    if not clips:
        return None, "No clips to concatenate."

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, blob in enumerate(clips, start=1):
            p = os.path.join(tmp, f"clip_{i}.mp4")
            with open(p, "wb") as f:
                f.write(blob)
            paths.append(p)

        list_path = os.path.join(tmp, "inputs.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"file '{p}'\n")

        out_path = os.path.join(tmp, "final.mp4")

        # First try stream copy
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        proc = subprocess.run(cmd_copy, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                return f.read(), None

        # Fallback: re-encode
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            out_path,
        ]
        proc2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
        if proc2.returncode != 0:
            return None, f"ffmpeg failed: {_short(proc2.stderr or proc.stderr)}"

        with open(out_path, "rb") as f:
            return f.read(), None


def generate_and_send_kling_transition_video(to: str, photo_urls: List[str]):
    photo_count = len(photo_urls)

    if photo_count < PHOTO_TRANSITION_MIN_COUNT:
        send_text(to, f"Please send at least {PHOTO_TRANSITION_MIN_COUNT} photos.")
        return

    if photo_count > PHOTO_TRANSITION_MAX_COUNT:
        photo_urls = photo_urls[:PHOTO_TRANSITION_MAX_COUNT]
        photo_count = len(photo_urls)

    est = kling_estimated_credits(photo_count, KLING_TRANSITION_DURATION)
    send_text(
        to,
        f"Creating your {photo_count}-photo transition video in 1080p, no audio 🎬\n"
        f"Estimated Kling cost: ~{est} credits."
    )

    clips: List[bytes] = []

    for i in range(len(photo_urls) - 1):
        start_url = photo_urls[i]
        end_url = photo_urls[i + 1]

        task_id, err = kling_create_transition_task(start_url, end_url)
        if err:
            send_text(to, err)
            return

        clip_blob, err = kling_poll_video_result(task_id, timeout_seconds=900)
        if err:
            send_text(to, err)
            return

        clips.append(clip_blob)

    final_blob, err = concat_mp4_clips(clips)
    if err:
        send_text(to, err)
        return

    media_id, up_err = wa_upload_media(final_blob, filename="transitions.mp4", mime_type="video/mp4")
    if up_err:
        send_text(to, up_err)
        return

    send_video_by_id(to, media_id)


# =========================
# Gemini chat
# =========================
_cb_until = 0.0
_cb_cooldown = float(CB_BASE_COOLDOWN)


def cb_is_open() -> bool:
    return time.time() < _cb_until


def cb_remaining() -> int:
    rem = int(_cb_until - time.time())
    return rem if rem > 0 else 0


def cb_trip():
    global _cb_until, _cb_cooldown
    now = time.time()
    _cb_until = now + _cb_cooldown
    log.warning("gemini_circuit_open:%ss", int(_cb_cooldown))
    _cb_cooldown = min(float(CB_MAX_COOLDOWN), _cb_cooldown * CB_BACKOFF_FACTOR)


def cb_reset():
    global _cb_cooldown
    _cb_cooldown = max(float(CB_BASE_COOLDOWN), _cb_cooldown / CB_BACKOFF_FACTOR)


def build_gemini_history(tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in tail:
        if not isinstance(m, dict):
            continue
        r = m.get("r")
        t = m.get("t")
        if r in ("user", "model") and isinstance(t, str) and t.strip():
            out.append({"role": r, "parts": [t]})
    return out


def call_gemini(history: List[Dict[str, Any]], prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if not gemini_model:
        return None, "OTHER"
    if cb_is_open():
        return None, "RATE_LIMIT"

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            chat = gemini_model.start_chat(history=history)
            resp = chat.send_message(prompt)
            text = (resp.text or "").strip() or "..."
            cb_reset()
            return text, None
        except Exception as e:
            if _looks_rate_limited(e):
                cb_trip()
                return None, "RATE_LIMIT"
            time.sleep(GEMINI_RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
    return None, "OTHER"


def chat_reply(uid: str, prompt: str) -> str:
    try:
        doc = users.find_one({"_id": uid}, {"history": 1}) or {}
        history = doc.get("history", [])
        if not isinstance(history, list):
            history = []
        tail = history[-CONTEXT_TURNS:] if CONTEXT_TURNS > 0 else []
        gem_hist = build_gemini_history(tail)

        text, err = call_gemini(gem_hist, prompt)
        if not text:
            if err == "RATE_LIMIT":
                return RATE_LIMIT_MESSAGE.format(seconds=cb_remaining() or 30)
            return GENERIC_FAIL_MESSAGE

        users.update_one(
            {"_id": uid},
            {"$push": {"history": {"$each": [{"r": "user", "t": prompt}, {"r": "model", "t": text}]}}},
            upsert=True,
        )
        return text
    except Exception as e:
        log.error("chat_reply_failed:%s", type(e).__name__)
        return GENERIC_FAIL_MESSAGE


# =========================
# Intent detection
# =========================
def seems_like_image_request(text: str) -> bool:
    t = text.lower().strip()
    triggers = [
        "generate an image", "create an image", "make an image",
        "draw", "illustration", "image of", "photo of", "art of"
    ]
    return any(x in t for x in triggers)


# =========================
# Webhook endpoints
# =========================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
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

                if "statuses" in val:
                    continue

                for msg in val.get("messages", []):
                    mid = msg.get("id")
                    if not mark_processed_once(mid):
                        continue

                    sender = msg.get("from")
                    if not sender:
                        continue

                    # cooldown
                    doc = users.find_one({"_id": sender}, {"last_user_at": 1}) or {}
                    now = time.time()
                    last_user_at = doc.get("last_user_at", 0)
                    if isinstance(last_user_at, (int, float)) and (now - last_user_at) < USER_COOLDOWN_SECONDS:
                        continue
                    users.update_one({"_id": sender}, {"$set": {"last_user_at": now}}, upsert=True)

                    mtype = msg.get("type")

                    # ---------- IMAGE = queue for transition video
                    if mtype == "image":
                        media_id = (msg.get("image") or {}).get("id")
                        if not media_id:
                            send_text(sender, "I couldn't read that photo. Please send it again.")
                            continue

                        img_bytes, img_mime, err = wa_download_incoming_media(media_id)
                        if err or not img_bytes:
                            send_text(sender, err or "Failed to download the photo.")
                            continue

                        store_transition_photo(sender, img_bytes)
                        photo_count = len(get_transition_photos(sender))

                        if photo_count >= PHOTO_TRANSITION_MAX_COUNT:
                            send_text(
                                sender,
                                f"Photo {photo_count}/{PHOTO_TRANSITION_MAX_COUNT} received ✅\n"
                                f"Send /makevideo when you're ready."
                            )
                        else:
                            send_text(
                                sender,
                                f"Photo {photo_count}/{PHOTO_TRANSITION_MAX_COUNT} received 📸\n"
                                f"Send more photos, or send /makevideo once you have at least {PHOTO_TRANSITION_MIN_COUNT}."
                            )
                        continue

                    # ---------- NON-TEXT
                    if mtype != "text":
                        continue

                    txt = (msg.get("text") or {}).get("body", "").strip()
                    if not txt:
                        continue

                    low = txt.lower().strip()

                    # Commands
                    if low in ("/menu", "menu", "/start", "start", "/help", "help"):
                        send_text(sender, WELCOME_TEXT)
                        continue

                    if low == "/limpar":
                        users.update_one(
                            {"_id": sender},
                            {"$unset": {"history": "", "transition_photos": ""}},
                            upsert=True,
                        )
                        send_text(sender, "Memory cleared 🧹")
                        continue

                    if low == "/photos":
                        count = len(get_transition_photos(sender))
                        est = kling_estimated_credits(count, KLING_TRANSITION_DURATION) if count >= 2 else 0
                        send_text(
                            sender,
                            f"You currently have {count} queued photo(s).\n"
                            f"Min: {PHOTO_TRANSITION_MIN_COUNT}, max: {PHOTO_TRANSITION_MAX_COUNT}.\n"
                            f"Estimated cost right now: ~{est} credits."
                        )
                        continue

                    if low == "/resetphotos":
                        clear_transition_photos(sender)
                        send_text(sender, "Queued photos cleared.")
                        continue

                    if low == "/makevideo":
                        photo_urls = get_transition_photo_urls(sender)
                        count = len(photo_urls)

                        if not PUBLIC_BASE_URL:
                            send_text(sender, "PUBLIC_BASE_URL is missing, so I can't expose the photos to Kling.")
                            continue

                        if count < PHOTO_TRANSITION_MIN_COUNT:
                            send_text(sender, f"Please send at least {PHOTO_TRANSITION_MIN_COUNT} photos first.")
                            continue

                        if count > PHOTO_TRANSITION_MAX_COUNT:
                            photo_urls = photo_urls[:PHOTO_TRANSITION_MAX_COUNT]
                            count = len(photo_urls)

                        # Clear queue before async generation so duplicates don't pile up
                        clear_transition_photos(sender)

                        threading.Thread(
                            target=generate_and_send_kling_transition_video,
                            args=(sender, photo_urls),
                            daemon=True
                        ).start()
                        continue

                    # onboarding
                    if users.count_documents({"_id": sender}, limit=1) == 0:
                        send_text(sender, WELCOME_TEXT)

                    # image generation
                    if seems_like_image_request(txt):
                        prompt = txt
                        generate_and_send_image(sender, prompt)
                        continue

                    # chat
                    reply = chat_reply(sender, txt)
                    send_text(sender, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error("webhook_failed:%s", type(e).__name__)
        return jsonify({"status": "ok"}), 200


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
