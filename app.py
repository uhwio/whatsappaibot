import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
import certifi  # <--- NEW IMPORT

load_dotenv()

app = Flask(__name__)

# config
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# setup gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# setup mongo WITH SSL FIX
# tlsCAFile=certifi.where() fixes the handshake error on linux
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.whatsapp_bot
users = db.chats

def get_response(uid, prompt):
    try:
        user_doc = users.find_one({"_id": uid})
        
        # deserialize history
        # (convert list of dicts back to gemini format if needed, 
        # but usually gemini can take the raw text if we structure it right.
        # simpler approach: just feed list of dicts)
        
        raw_history = user_doc['history'] if user_doc else []
        
        # reconstruct history for gemini
        gemini_history = []
        for msg in raw_history:
            gemini_history.append({
                "role": msg["role"],
                "parts": msg["parts"]
            })

        chat = model.start_chat(history=gemini_history)
        resp = chat.send_message(prompt)
        
        # serialize new history for mongo
        new_history = []
        for msg in chat.history:
            # handle parts being a list of objects or strings
            text_parts = []
            for part in msg.parts:
                text_parts.append(part.text)
                
            new_history.append({
                "role": msg.role,
                "parts": text_parts
            })
            
        users.update_one(
            {"_id": uid}, 
            {"$set": {"history": new_history}}, 
            upsert=True
        )
        return resp.text
        
    except Exception as e:
        print(f"gemini/mongo crash: {e}")
        return "brain error"

def reply_wa(target, body):
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
            }
        )
        if res.status_code != 200:
            print(f"wa api fail: {res.text}")
    except Exception as e:
        print(f"req error: {e}")

@app.route("/webhook", methods=["GET"])
def verify_token():
    args = request.args
    if args.get("hub.mode") == "subscribe" and args.get("hub.verify_token") == VERIFY:
        return args.get("hub.challenge"), 200
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def inbound():
    data = request.get_json()

    try:
        entry = data['entry'][0]
        changes = entry['changes'][0]
        val = changes['value']
        
        if "statuses" in val:
            return jsonify({"status": "ignored"}), 200

        if "messages" in val:
            msg = val["messages"][0]
            sender = msg["from"]
            m_type = msg["type"]
            
            if m_type == "text":
                txt = msg["text"]["body"]
                print(f"rx {sender}: {txt}")

                if txt.lower() == "/reset":
                    users.delete_one({"_id": sender})
                    reply_wa(sender, "memory wiped from db")
                elif txt.lower() == "ping":
                    reply_wa(sender, "pong")
                else:
                    out = get_response(sender, txt)
                    reply_wa(sender, out)

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"webhook fatal: {e}")
        return jsonify({"status": "err"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
