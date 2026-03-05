import os
import time
import base64
import secrets
import urllib.parse
import logging
import threading
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
# Logging (no messages, no numbers)
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
log.info("BOOT: iniciado (logs ativos)")

# =========================
# Config
# =========================
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")

MONGO_URI = os.getenv("MONGO_URI")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Pollinations
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")
POLLINATIONS_VIDEO_MODEL = os.getenv("POLLINATIONS_VIDEO_MODEL", "grok-video")
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "512"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "512"))
POLLINATIONS_VIDEO_DURATION = int(os.getenv("POLLINATIONS_VIDEO_DURATION", "3"))
POLLINATIONS_VIDEO_ASPECT_RATIO = os.getenv("POLLINATIONS_VIDEO_ASPECT_RATIO", "16:9")
POLLINATIONS_VIDEO_AUDIO = os.getenv("POLLINATIONS_VIDEO_AUDIO", "false").lower() in ("1", "true", "yes", "on")

# Public base URL for serving reference images (ngrok URL)
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
RATE_LIMIT_MESSAGE = os.getenv("RATE_LIMIT_MESSAGE", "Estou limitado agora 😅 Tente novamente em {seconds}s.")
GENERIC_FAIL_MESSAGE = os.getenv("GEMINI_FAIL_MESSAGE", "Tive um problema agora. Tente de novo em instantes.")

# Fixed prompt for image->video
ANIME_VIDEO_PROMPT = os.getenv(
    "ANIME_VIDEO_PROMPT",
    "cinematic anime animation, studio-quality, smooth motion, dramatic lighting, vibrant colors, "
    "depth of field, film grain, dynamic camera movement, richly detailed background, "
    "soft shadows, high fidelity, anime movie look"
)

WELCOME_TEXT = (
    "Olá! 👋\n\n"
    "Eu posso:\n"
    "🖼 Criar imagens a partir de texto\n"
    "🎬 Transformar uma foto em vídeo estilo anime\n"
    "💬 Conversar e responder perguntas\n\n"
    "Como usar:\n"
    "• Envie uma *foto* → eu crio um *vídeo anime* automaticamente\n"
    "• Escreva um texto → eu converso (ou gero uma imagem se você pedir)\n\n"
    "Comandos:\n"
    "• /menu  → ver ajuda\n"
    "• /limpar → apagar memória da conversa"
)

# =========================
# Setup http, Gemini, Mongo
# =========================
os.makedirs(REF_DIR, exist_ok=True)
http = requests.Session()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot
users = db.chats
processed = db.wa_processed_message_ids

def ensure_indexes():
    try:
        processed.create_index([("expiresAt", ASCENDING)], expireAfterSeconds=0, name="expiresAt_ttl")
        log.info("BOOT: index TTL dedupe ok")
    except OperationFailure as e:
        log.warning("BOOT: não conseguiu criar index TTL: %s", type(e).__name__)

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
            timeout=90,
        )
        if r.status_code < 200 or r.status_code >= 300:
            log.error("wa_media_http_%s: %s", r.status_code, _short(r.text))
            return None, f"Erro ao enviar mídia para WhatsApp ({r.status_code}). {_short(r.text)}"
        media_id = (r.json() or {}).get("id")
        if not media_id:
            return None, "WhatsApp não retornou o id da mídia."
        return media_id, None
    except Exception as e:
        log.error("wa_upload_failed: %s", type(e).__name__)
        return None, "Falha ao enviar mídia para WhatsApp."

def send_image_by_id(to: str, media_id: str):
    wa_send({"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id, "caption": "🖼 Aqui está sua imagem"}})

def send_video_by_id(to: str, media_id: str):
    wa_send({"messaging_product": "whatsapp", "to": to, "type": "video", "video": {"id": media_id, "caption": "🎬 Aqui está seu vídeo"}})

# =========================
# WhatsApp incoming media download (for ref image)
# =========================
def wa_download_incoming_media(media_id: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    try:
        meta = http.get(
            f"https://graph.facebook.com/v21.0/{media_id}",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            timeout=20,
        )
        if meta.status_code < 200 or meta.status_code >= 300:
            return None, None, f"Erro ao pegar info da mídia ({meta.status_code}). {_short(meta.text)}"

        j = meta.json() or {}
        url = j.get("url")
        mime = j.get("mime_type") or "application/octet-stream"
        if not url:
            return None, None, "WhatsApp não retornou a URL da mídia."

        blob = http.get(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=60)
        if blob.status_code < 200 or blob.status_code >= 300:
            return None, None, f"Erro ao baixar mídia ({blob.status_code}). {_short(blob.text)}"

        return blob.content, mime, None
    except Exception as e:
        return None, None, f"Falha ao baixar mídia: {type(e).__name__}"

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

def store_reference_image(uid: str, image_bytes: bytes) -> str:
    cleanup_old_refs()
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

def get_reference_url(uid: str) -> Optional[str]:
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
    cleanup_old_refs()
    path = os.path.join(REF_DIR, token)
    if not os.path.isfile(path):
        return "Não encontrado", 404
    try:
        with open(path, "rb") as f:
            blob = f.read()
        return Response(blob, status=200, mimetype=_guess_mime(blob))
    except Exception:
        return "Erro", 500

# =========================
# Pollinations (image/video) with retry + configurable timeouts
# =========================
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
    image_url: Optional[str] = None,
    timeout_seconds: int = 240,
    retries: int = 1,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if not POLLINATIONS_API_KEY:
        return None, None, "Pollinations não configurado (faltando POLLINATIONS_API_KEY)."

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
    if duration is not None:
        params["duration"] = int(duration)
    if aspect_ratio is not None:
        params["aspectRatio"] = aspect_ratio
    if audio is not None:
        params["audio"] = "true" if audio else "false"
    if image_url:
        params["image"] = image_url

    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"}

    for attempt in range(retries + 1):
        try:
            r = http.get(url, params=params, headers=headers, timeout=timeout_seconds)

            if r.status_code < 200 or r.status_code >= 300:
                return None, None, f"Erro Pollinations ({r.status_code}). {_short(r.text)}"

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
                    return None, None, f"Pollinations retornou JSON sem imagem. {_short(r.text)}"
                except Exception:
                    return None, None, f"Pollinations retornou JSON mas falhou ao ler. {_short(r.text)}"

            blob = r.content
            return blob, _guess_mime(blob), None

        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None, None, "Pollinations demorou demais e expirou (timeout)."

        except Exception as e:
            return None, None, f"Falha na requisição Pollinations: {type(e).__name__}"

    return None, None, "Falha desconhecida."

def generate_and_send_image(to: str, prompt: str):
    send_text(to, "Criando sua imagem... 🖼\nIsso pode levar alguns segundos.")
    blob, mime, err = pollinations_generate_media(
        prompt=prompt,
        model_name=POLLINATIONS_IMAGE_MODEL,
        width=POLLINATIONS_WIDTH,
        height=POLLINATIONS_HEIGHT,
        timeout_seconds=180,
        retries=0,
    )
    if err:
        send_text(to, err)
        return
    if not blob or not mime or not mime.startswith("image/"):
        send_text(to, f"Não consegui criar a imagem (recebi {mime or 'sem tipo'}).")
        return

    filename = "imagem.png" if mime == "image/png" else "imagem.jpg"
    media_id, up_err = wa_upload_media(blob, filename=filename, mime_type=mime)
    if up_err:
        send_text(to, up_err)
        return
    send_image_by_id(to, media_id)

def generate_and_send_video_from_photo(to: str, ref_url: Optional[str]):

    for attempt in range(3):

        blob, mime, err = pollinations_generate_media(
            prompt=ANIME_VIDEO_PROMPT,
            model_name=POLLINATIONS_VIDEO_MODEL,
            width=POLLINATIONS_WIDTH,
            height=POLLINATIONS_HEIGHT,
            duration=POLLINATIONS_VIDEO_DURATION,
            aspect_ratio=POLLINATIONS_VIDEO_ASPECT_RATIO,
            audio=POLLINATIONS_VIDEO_AUDIO,
            image_url=ref_url,
            timeout_seconds=600,
            retries=1,
        )

        if not err:
            break

        # se erro 520 tenta novamente
        if "520" in err and attempt < 2:
            time.sleep(6)
            continue

        send_text(
            to,
            "O servidor que cria os vídeos está ocupado agora 😕\n\n"
            "Tente novamente enviando a foto daqui a alguns minutos."
        )
        return

    if not blob or mime != "video/mp4":
        send_text(to, "Não consegui criar o vídeo agora.")
        return

    media_id, up_err = wa_upload_media(blob, filename="video.mp4", mime_type="video/mp4")

    if up_err:
        send_text(to, "O vídeo foi criado mas falhou ao enviar para o WhatsApp.")
        return

    send_video_by_id(to, media_id)

# =========================
# Gemini chat (circuit breaker)
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
    log.warning("gemini_circuit_aberto:%ss", int(_cb_cooldown))
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
def parece_pedido_de_imagem(texto: str) -> bool:
    t = texto.lower().strip()
    gatilhos = [
        "gera uma imagem", "gerar uma imagem", "crie uma imagem", "criar uma imagem",
        "faz uma imagem", "fazer uma imagem", "desenha", "desenhe", "desenhar",
        "uma imagem de", "imagem de", "foto de", "crie um desenho", "desenho de",
        "criar arte", "arte de", "ilustração", "ilustracao"
    ]
    return any(g in t for g in gatilhos)

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

                    # ---------- FOTO = VÍDEO AUTOMÁTICO (RUNS IN BACKGROUND)
                    if mtype == "image":
                        media_id = (msg.get("image") or {}).get("id")
                        if not media_id:
                            send_text(sender, "Não consegui ler essa foto. Tente enviar de novo.")
                            continue

                        img_bytes, img_mime, err = wa_download_incoming_media(media_id)
                        if err or not img_bytes:
                            send_text(sender, err or "Falhei ao baixar a foto. Tente novamente.")
                            continue

                        store_reference_image(sender, img_bytes)
                        ref_url = get_reference_url(sender)

                        send_text(sender, "Criando seu vídeo... 🎬\nIsso pode levar cerca de 1 minuto.")

                        # IMPORTANT: do not block webhook thread
                        threading.Thread(
                            target=generate_and_send_video_from_photo,
                            args=(sender, ref_url),
                            daemon=True
                        ).start()

                        continue

                    # ---------- TEXTO
                    if mtype != "text":
                        continue

                    txt = (msg.get("text") or {}).get("body", "").strip()
                    if not txt:
                        continue

                    low = txt.lower().strip()

                    # comandos
                    if low in ("/menu", "menu", "/start", "start", "/help", "help"):
                        send_text(sender, WELCOME_TEXT)
                        continue

                    if low == "/limpar":
                        users.delete_one({"_id": sender})
                        send_text(sender, "Memória apagada 🧹\nPodemos começar do zero.")
                        continue

                    # onboarding (send help once if new)
                    if users.count_documents({"_id": sender}, limit=1) == 0:
                        send_text(sender, WELCOME_TEXT)

                    # imagem
                    if parece_pedido_de_imagem(txt):
                        prompt = txt
                        for g in [
                            "gera uma imagem de", "gerar uma imagem de", "crie uma imagem de", "criar uma imagem de",
                            "faz uma imagem de", "fazer uma imagem de", "uma imagem de", "imagem de"
                        ]:
                            prompt = prompt.lower().replace(g, "").strip()
                        prompt = prompt.strip() or txt
                        generate_and_send_image(sender, prompt)
                        continue

                    # conversa
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
