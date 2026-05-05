# app.py
import streamlit as st
import pickle
import re
import string
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# -----------------------------
# NLTK Downloads
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Load Pickled Models
# -----------------------------
@st.cache_data
def load_models():
    with open("mnb_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("mtfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return nb_model, vectorizer

nb_model, vectorizer = load_models()

# -----------------------------
# Load Metrics JSON
# -----------------------------
@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        metrics = json.load(f)
    return metrics

metrics = load_metrics()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="wide")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict whether it is Positive or Negative.")

review_text = st.text_area("Enter your review:", placeholder="e.g., The movie was fantastic with amazing performances!")

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter some text!")
    else:
        # Preprocess & predict
        clean_review = preprocess_text(review_text)
        vect_review = vectorizer.transform([clean_review])
        prediction = nb_model.predict(vect_review)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        color = "#2ecc71" if prediction == 1 else "#ff4b4b"

        # Probability for progress bars
        if hasattr(nb_model, "predict_proba"):
            prob = nb_model.predict_proba(vect_review)[0]
            pos_prob = prob[1]
            neg_prob = prob[0]
        else:
            pos_prob = 0.8 if prediction == 1 else 0.2
            neg_prob = 1 - pos_prob

        # Display in columns
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**Predicted Sentiment:** <span style='color:{color}; font-size:20px'>{sentiment}</span>", unsafe_allow_html=True)
        with col2:
            st.write("Prediction Probabilities:")
            st.progress(int(pos_prob * 100))
            st.write(f"Positive: {pos_prob:.2%}")
            st.progress(int(neg_prob * 100))
            st.write(f"Negative: {neg_prob:.2%}")

# -----------------------------
# Model Performance Section
# -----------------------------
st.markdown("---")
st.subheader("Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

# -----------------------------
# Confusion Matrix as Table
# -----------------------------
st.write("Confusion Matrix:")
cm_df = pd.DataFrame(
    metrics['confusion_matrix'],
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)
st.dataframe(cm_df.style.background_gradient(cmap="Reds"))
