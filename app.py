import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Modern Dark Gradient Theme
st.markdown(
    """
    <style>
    /* Apply background to full Streamlit app */
    .stApp {
        background: linear-gradient(135deg, #6a11cb, #2575fc); /* Purple to blue */
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #d0d0d0;
        margin-bottom: 30px;
    }
    textarea {
        background-color: rgba(20, 20, 40, 0.9) !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 15px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff4b2b, #ff416c) !important; /* pink gradient */
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        transition: 0.3s;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .card {
        background: rgba(0,0,0,0.6);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    }
    .positive {
        color: #4caf50;
        font-size: 24px;
        font-weight: bold;
    }
    .negative {
        color: #ff1744;
        font-size: 24px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 13px;
        color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="main-title">💬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify text into Positive or Negative sentiment</div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("Write your text here:")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        # Only Positive / Negative
        if prediction == 1:
            st.markdown('<div class="card"><p class="positive">😊 Positive Sentiment</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><p class="negative">☹️ Negative Sentiment</p></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text.")

# Footer
st.markdown('<div class="footer">✨ Made with ❤️ by Vrinda Parmar</div>', unsafe_allow_html=True)
