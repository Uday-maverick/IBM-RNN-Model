# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb  # type: ignore
from tensorflow.keras.preprocessing import sequence  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st

# Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Cool UI Styling
st.set_page_config(page_title="🎬 IMDB Sentiment Classifier", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
    }
    h1, h2, p {
        color: white;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #2d2a3a;
        color: #ffffff;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 25px;
        height: 3em;
        width: 50%;
        font-size: 18px;
        margin: 10px auto;
        display: block;
        box-shadow: 0 4px 10px rgba(255, 111, 97, 0.4);
    }
    .result {
        background-color: #ffffff10;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🎥 IMDB Movie Review Sentiment</h1>", unsafe_allow_html=True)
st.markdown("<p>✨ Enter a movie review and we'll tell you if it’s <b style='color:#00fa9a;'>Positive</b> or <b style='color:#ff6f61;'>Negative</b> ✨</p>", unsafe_allow_html=True)

# User Input
user_input = st.text_area('📝 Type your movie review here:', height=150)

# Predict and Display
if st.button('🚀 Classify Sentiment'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        sentiment = "🌟 Positive" if score > 0.5 else "💔 Negative"

        st.markdown(f"""
            <div class="result">
                <h2>Sentiment: {sentiment}</h2>
                <p>Confidence Score: <b>{score:.4f}</b></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some review text first.")
else:
    st.markdown("<p style='text-align:center;'>👇 Your prediction result will appear here after clicking the button.</p>", unsafe_allow_html=True)


