import streamlit as st
import joblib
import json
import os

# Load model (FIXED)
model = joblib.load("cyberbullying_model.pkl")

st.set_page_config(page_title="AI Chat Moderation", layout="centered")

st.title("💬 AI Moderated Chat Platform")

# File to store chat
CHAT_FILE = "chat.json"

# Load chat safely (FIXED)
def load_chat():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

# Save chat
def save_chat(chat):
    with open(CHAT_FILE, "w") as f:
        json.dump(chat, f)

# Load existing chat
chat_data = load_chat()

# Inputs
name = st.text_input("Enter your name")
message = st.text_input("Enter your message")

# Toxic emojis + keywords
toxic_emojis = ["😡","🤬","😠","💀","👿","🖕"]
toxic_words = ["stupid", "idiot", "hate", "ugly", "dumb"]

if st.button("Send"):

    if name and message:

        is_toxic = False
        confidence = 0.0

        msg_lower = message.lower()

        # Emoji check
        if any(e in message for e in toxic_emojis):
            is_toxic = True
            confidence = 1.0

        # Keyword check
        elif any(word in msg_lower for word in toxic_words):
            is_toxic = True
            confidence = 0.9

        else:
            pred = model.predict([message])[0]
            prob = model.predict_proba([message])[0][1]

            if pred == 1:
                is_toxic = True
                confidence = prob

        # Add to chat
        chat_data.append({
            "user": name,
            "message": message,
            "toxic": is_toxic,
            "confidence": confidence
        })

        save_chat(chat_data)

# Display chat
st.subheader("📡 Live Chat Room")

for chat in chat_data:
    user = chat["user"]
    msg = chat["message"]
    toxic = chat["toxic"]
    conf = chat["confidence"]

    if toxic:
        st.error(f"🔴 {user}: {msg} ({conf*100:.1f}%)")
    else:
        st.success(f"🟢 {user}: {msg}")

# Analytics
total = len(chat_data)
toxic_count = sum(1 for c in chat_data if c["toxic"])

if total > 0:
    st.write(f"📊 Toxic Messages: {toxic_count}/{total}")
