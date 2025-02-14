from transformers import pipeline  # Doit fonctionner après l'installation
import streamlit as st
from transformers import pipeline
pip install transformers 
transformers>=4.30.0
MODEL = "jy46604790/Fake-News-Bert-Detect"
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# ✅ Fake news prediction function
def predict_fake_news(text):
    result = clf(text)
    return result
    #return "😊 Positive" if prediction[0] == 1 else "😞 Negative"

# ✅ Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="💬", layout="centered")

st.title("💡 Fake News Detection")
st.markdown("### Enter an article below and check if it's fake now! 🚀")

# User input
user_input = st.text_area("📝 Enter your article here:", height=150, placeholder="Type or paste an article...")

# Predict sentiment
if st.button("🔍 Detect Fake News"):
    if user_input.strip():
        result = predict_fake_news(user_input)
        if result != "Error":
            st.success(f"*Result :* {result}")
    else:
        st.warning("⚠️ Please enter an article before analyzing.")

# ✅ Extra UI enhancements
st.markdown("---")
st.markdown("### ✨ Why use this app?")
st.markdown("- 🔥 *Instant Sentiment Analysis* for your tweets!")
st.markdown("- 🎨 *Beautiful & Minimal UI* for easy interaction.")
st.markdown("- 🚀 *Fast & Efficient* model built with TF-IDF & Logistic Regression.")
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit and Machine Learning! ✨")
