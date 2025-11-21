# src/dashboards/streamlit_app.py

import streamlit as st
import joblib
from pathlib import Path
from utils.config import MODELS_DIR
from utils.topic_evaluation import topic_distribution, top_docs
from utils.logging_utils import get_logger
import pandas as pd

logger = get_logger("dashboard")


# ============================
# Load models
# ============================

@st.cache_resource
def load_sentiment_model():
    model = joblib.load(MODELS_DIR / "sentiment" / "logreg_model.joblib")
    tfidf = joblib.load(MODELS_DIR / "sentiment" / "tfidf_vectorizer.joblib")
    return model, tfidf


@st.cache_resource
def load_topic_model():
    from bertopic import BERTopic
    return BERTopic.load(MODELS_DIR / "topic" / "bertopic_model")


# ============================
# Sidebar
# ============================

st.sidebar.title("Customer Voice Intelligence")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Sentiment Analysis", "Topic Exploration"]
)

# ============================
# Sentiment Page
# ============================

if page == "Sentiment Analysis":
    st.title("Sentiment Prediction Demo")

    model, tfidf = load_sentiment_model()

    text = st.text_area("Enter a review:", height=200)

    if st.button("Predict"):
        vec = tfidf.transform([text])
        pred = int(model.predict(vec)[0])
        prob = model.predict_proba(vec)[0][pred]

        label = "Positive" if pred == 1 else "Negative"

        st.markdown(f"### Prediction: {label}")
        st.write(f"Confidence: {prob:.4f}")

# ============================
# Topic Modeling Page
# ============================

elif page == "Topic Exploration":
    st.title("Topic Modeling Dashboard")

    topic_model = load_topic_model()

    uploaded = st.file_uploader("Upload topic assignments CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("### Topic distribution")
        dist = topic_distribution(df)

        st.bar_chart(dist)

        topic_id = st.number_input("Select topic", 0, int(df.Topic.max()))

        st.write("### Top words")
        st.json(topic_model.get_topic(topic_id))

        st.write("### Sample documents")
        st.table(top_docs(df, topic_id, n=5))

