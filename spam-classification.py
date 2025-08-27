import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Initialize stemmer
ps = PorterStemmer()
# ---------- Text Preprocessing ----------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)
# ---------- Load Model & Vectorizer ----------
tfidf = pickle.load(open('CountVectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
# ---------- Page Config ----------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="wide")
# ---------- Modern CSS ----------
st.markdown("""
    <style>
    /* Background gradient animation */
    body {
        background: linear-gradient(270deg, #c3e0ff, #d5c6ff, #b2cfff, #e0ccff);
        background-size: 800% 800%;
        animation: gradientFlow 18s ease infinite;
    }
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Main card */
    .main-card {
        background: rgba(255, 255, 255, 0.25);
        padding: 40px;
        border-radius: 20px;
        backdrop-filter: blur(15px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
        text-align: center;
        animation: slideUp 1.2s ease;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Title */
    .title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #6a5acd, #00bfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 10px rgba(106,90,205,0.3);
        transition: transform 0.3s ease;
    }
    .title:hover {
        transform: scale(1.05);
        text-shadow: 0px 0px 20px rgba(106,90,205,0.6);
    }
    /* Subtitle */
    .subtitle {
        font-size: 18px;
        color: #333;
        margin-bottom: 25px;
    }
    /* Input */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #9b8cff;
        padding: 14px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #6a5acd;
        box-shadow: 0px 0px 12px rgba(106,90,205,0.5);
    }
    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #89c4f4, #a89df7);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 22px;
        border: none;
        transition: all 0.3s ease;
        font-size: 17px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #72b3f6, #917ef6);
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
    }
    .stButton>button::after {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.3);
        transition: 0.5s;
    }
    .stButton>button:hover::after {
        left: 100%;
    }
    /* Results */
    .result-spam {
        font-size: 26px;
        font-weight: bold;
        color: #ff4c4c;
        text-shadow: 0 0 15px rgba(255,76,76,0.7);
        animation: pulse 1.5s infinite;
    }
    .result-notspam {
        font-size: 26px;
        font-weight: bold;
        color: #3cb371;
        text-shadow: 0 0 15px rgba(60,179,113,0.6);
        animation: fadeIn 2s;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Sidebar custom styling */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(12px);
        border-right: 2px solid rgba(106,90,205,0.3);
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #5a3dd1;
    }
    .sidebar-text {
        font-size: 16px;
        color: #333;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)
# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>📌 Project Details</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='sidebar-text'>
        🚀 <b>Project:</b> SMS/Email Spam Classifier <br><br>
        🧠 <b>Model:</b> Trained on spam & ham dataset using NLP preprocessing + ML <br><br>
        🎯 <b>Goal:</b> Detect spam messages with accuracy and speed <br><br>
        👨‍💻 <b>Designed & Trained by:</b> <span style="color:#6a5acd;font-weight:bold;">Dibyendu</span>
        </p>
        """, 
        unsafe_allow_html=True
    )
# ---------- Main Layout ----------
st.markdown("<h1 class='title'>📩 Ultra Modern Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Zero Spam. Maximum Inbox Clarity. Always ✨</p>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    input_sms = st.text_input("✍️ Enter the Email or SMS to check", "")
    if st.button('🔮 Predict Message'):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter some text first!")
        else:
            # Preprocess
            transformed_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]
            # Display result
            if result == 1:
                st.markdown("<p class='result-spam'>🚨 Spam Message Detected!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='result-notspam'>✅ This message is Safe</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)