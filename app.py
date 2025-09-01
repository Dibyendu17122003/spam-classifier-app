import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

# -------------------- Load Model --------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("CountVectorizer.pkl")

# -------------------- Session State --------------------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=5)

# -------------------- Page Config --------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="wide")

# -------------------- Ultra-Modern CSS --------------------
st.markdown("""
<style>
/* App Background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e, #222244);
    font-family: 'Segoe UI', sans-serif;
    color: #fff;
}

/* Title Animation */
h1 {
    font-size: 2.8rem !important;
    text-align: center;
    background: linear-gradient(90deg, #ff4dff, #00e6e6, #7d5fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #ff4dff; }
    to { text-shadow: 0 0 30px #00e6e6, 0 0 50px #7d5fff; }
}

/* Subtitle */
h2, h3 {
    text-shadow: 0 0 15px rgba(0,255,255,0.3);
    color: #bb86fc !important;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(45deg, #7d5fff, #00e6e6);
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: bold;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: all 0.3s ease-in-out;
}
button[kind="primary"]:hover {
    transform: scale(1.08);
    box-shadow: 0 0 20px rgba(0,255,255,0.6);
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    padding: 12px !important;
    transition: 0.3s;
}
textarea:focus {
    outline: none !important;
    border: 1px solid #00e6e6 !important;
    box-shadow: 0 0 15px #00e6e6;
}

/* History Cards */
.custom-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 14px 20px;
    margin-bottom: 14px;
    backdrop-filter: blur(12px);
    animation: fadeIn 0.6s ease;
    transition: transform 0.25s ease, background 0.3s ease;
}
.custom-card:hover {
    transform: translateY(-4px) scale(1.02);
    background: rgba(255,255,255,0.15);
    box-shadow: 0 0 20px rgba(0,255,255,0.25);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px);}
    to { opacity: 1; transform: translateY(0);}
}

/* Labels */
.msg-label { color: #87CEFA; font-size: 15px; }
.spam-label { color: #ff4c4c; font-weight: bold; font-size: 15px; text-shadow: 0 0 10px #ff1a1a; }
.ham-label { color: #00ff99; font-weight: bold; font-size: 15px; text-shadow: 0 0 10px #00ff99; }
.confidence-label { color: #bb86fc; font-size: 14px; text-shadow: 0 0 5px #bb86fc; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(20,20,40,0.9);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 15px rgba(187,134,252,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.title("📩 Spam / Ham Classifier")
st.caption("✨ Ultra-modern AI-powered SMS & Email Spam Detection ✨")

# -------------------- Input --------------------
msg = st.text_area("✍️ Enter a message to classify:", height=120)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("⚖️ Classify"):
        if msg.strip():
            X = vectorizer.transform([msg])
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

            label = "Spam" if prediction == 1 else "Ham"
            confidence = np.max(prob) * 100

            # Save to history
            st.session_state.history.appendleft({
                "Message": msg,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%"
            })

            if label == "Spam":
                st.error(f"🔥 **Spam detected!** {confidence:.2f}% confident")
            else:
                st.success(f"🌱 **Ham detected!** {confidence:.2f}% confident")
        else:
            st.warning("⚠️ Please enter a message!")

# -------------------- History --------------------
st.markdown("## 📜 Classified Messages")

if st.session_state.history:
    for i, entry in enumerate(st.session_state.history):
        st.markdown(f"""
            <div class="custom-card">
                <p style="margin:0; color:#ccc; font-size:13px;">#{i+1}</p>
                <p class="msg-label"><b>📩 Message:</b> {entry['Message']}</p>
                <p class="{ 'spam-label' if entry['Prediction']=='Spam' else 'ham-label'}">
                    <b>🔍 Prediction:</b> {entry['Prediction']}
                </p>
                <p class="confidence-label"><b>📊 Confidence:</b> {entry['Confidence']}</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("🚫 No history yet. Try classifying a message!")

# -------------------- Detailed Analysis --------------------
st.markdown("## 🔬 Detailed Analysis of Last Message")
if st.session_state.history:
    last_entry = st.session_state.history[0]
    X_last = vectorizer.transform([last_entry["Message"]])
    probs = model.predict_proba(X_last)[0]

    colA, colB = st.columns(2)

    with colA:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Ham", "Spam"],
            values=probs,
            hole=0.5,
            marker=dict(colors=["#00ff99", "#ff4c4c"]),
            textinfo="label+percent",
            pull=[0.05, 0]
        )])
        fig_pie.update_layout(title_text="Spam vs Ham Probability", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        fig_bar = px.bar(
            x=["Ham", "Spam"],
            y=probs,
            text=[f"{p:.2f}" for p in probs],
            color=["Ham", "Spam"],
            color_discrete_map={"Ham": "#00ff99", "Spam": "#ff4c4c"}
        )
        fig_bar.update_traces(textposition="outside", marker=dict(line=dict(width=1, color="black")))
        fig_bar.update_layout(
            yaxis=dict(range=[0, 1]),
            title_text="Confidence Distribution",
            title_x=0.5
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- Description --------------------
st.markdown("## 📚 Description")

st.write("""
### 💀 Spam is Dangerous  
- 📩 Spam emails and messages can flood inboxes, reducing productivity.  
- 🦠 Many contain **malware, ransomware, or viruses** disguised as harmless files.  
- 🎭 Often used for **phishing attacks**, tricking users into sharing passwords or banking details.  
- 💸 Can lead to **financial loss, identity theft, and data breaches**.  

### 🔖 Why Classify Spam?  
- ✅ Protects users from **fraudulent activities and scams**.  
- 📊 Improves **email and chat system efficiency** by filtering junk.  
- 👨‍💻 Builds **trust in digital communication** by keeping it clean.  
- ⚡ Enables organizations to **save time and resources** wasted on spam.  

### 💡⭐ Why Our Web App?  
- 🤖 **ML-Powered Detection**: Uses advanced machine learning & NLP models for accuracy.  
- 🕒 **Real-Time Classification**: Instantly separates spam from genuine messages.  
- 💻 **User-Friendly Interface**: Clean, modern, and responsive for all devices.  
- 🔍 **Interactive Dashboard**: Provides insights, charts, and spam trends analysis.  
- 🌐 **Secure & Scalable**: Designed for personal use, enterprises, and educational purposes.  
""")

# -------------------- About --------------------
with st.sidebar:
    st.markdown("## 📌 Project Details")
    st.markdown("""
    ### 📋 **Project**  
    SMS / Email Spam Classifier  

    ### 🤖 **Model**  
    Trained on spam & ham dataset using NLP preprocessing + ML  

    ### 🎯 **Goal**  
    Detect spam messages with accuracy and speed  

    ### 👨‍💻 **Designed & Trained by**  
    Dibyendu Karmahapatra  
    """)
