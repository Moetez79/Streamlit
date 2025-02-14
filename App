import streamlit as st
import pickle
import re
import os

# ✅ Define the correct model path
model_path = "../models/sentiment_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

# ✅ Load the saved vectorizer and model only once (better performance)
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    st.error("❌ Model or vectorizer file not found. Ensure both .pkl files are in the `models/` folder.")

# ✅ Clean function
def clean_tweet(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    return text

# ✅ Sentiment prediction function
def predict_sentiment(text):
    if not hasattr(vectorizer, "idf_"):  # Ensure vectorizer is fitted
        st.error("❌ The TF-IDF vectorizer is not fitted. Make sure it was trained and saved properly.")
        return "Error"

    cleaned_text = clean_tweet(text)
    transformed_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(transformed_text)
    return "😊 Positive" if prediction[0] == 1 else "😞 Negative"

# ✅ Streamlit UI
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💡 Tweet Sentiment Analyzer")
st.markdown("### Paste a tweet below and analyze its sentiment in real-time! 🚀")

# User input
user_input = st.text_area("📝 Enter your tweet here:", height=150, placeholder="Type or paste a tweet...")

# Predict sentiment
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        result = predict_sentiment(user_input)
        if result != "Error":
            st.success(f"**Predicted Sentiment:** {result}")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")

# ✅ Extra UI enhancements
st.markdown("---")
st.markdown("### ✨ Why use this app?")
st.markdown("- 🔥 **Instant Sentiment Analysis** for your tweets!")
st.markdown("- 🎨 **Beautiful & Minimal UI** for easy interaction.")
st.markdown("- 🚀 **Fast & Efficient** model built with TF-IDF & Logistic Regression.")
st.markdown("---")
st.markdown("💡 *Built with ❤️ using Streamlit and Machine Learning!* ✨")
