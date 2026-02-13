import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# load .env
load_dotenv()

app = Flask(__name__)

# .env cfg
WA_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY = os.getenv("VERIFY_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")

# setup gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# just a dict for now need redis later
memory = {}

def get_response(uid, prompt):
    try:
        # grab history
        hist = memory.get(uid, [])
        
        chat = model.start_chat(history=hist)
        resp = chat.send_message(prompt)
        
        # save back to dict
        memory[uid] = chat.history
        return resp.text
        
    except Exception as e:
        print(f"gemini crash: {e}")
        return "brain error, try again"

def reply_wa(target, body):
    url = f"https://graph.facebook.com/v21.0/{PHONE_ID}/messages"
    
    # sending the post request
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
        # just logging non-200s
        if res.status_code != 200:
            print(f"wa api fail: {res.text}")
            
    except Exception as e:
        print(f"req error: {e}")

@app.route("/webhook", methods=["GET"])
def verify_token():
    # meta handshake
    args = request.args
    if args.get("hub.mode") == "subscribe" and args.get("hub.verify_token") == VERIFY:
        return args.get("hub.challenge"), 200
    return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def inbound():
    data = request.get_json()

    try:
        # extracting the deep nested json
        entry = data['entry'][0]
        changes = entry['changes'][0]
        val = changes['value']
        
        # ignore status updates (delivered, read, etc)
        if "statuses" in val:
            return jsonify({"status": "ignored"}), 200

        if "messages" in val:
            msg = val["messages"][0]
            sender = msg["from"]
            m_type = msg["type"]
            
            if m_type == "text":
                txt = msg["text"]["body"]
                print(f"rx {sender}: {txt}")

                # manual override commands
                if txt.lower() == "/reset":
                    if sender in memory: del memory[sender]
                    reply_wa(sender, "context wiped")
                
                elif txt.lower() == "ping":
                    reply_wa(sender, "pong")
                    
                else:
                    # hit the ai
                    out = get_response(sender, txt)
                    reply_wa(sender, out)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        # catch all for now
        print(f"webhook fatal: {e}")
        return jsonify({"status": "err"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
