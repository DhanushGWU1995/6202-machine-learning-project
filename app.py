# import streamlit as st
# from transformers import pipeline

# MODEL_NAME = "DhanushGWU1995/ecris-category-model"

# st.title("ECRIS - Customer Risk Intelligence System")

# @st.cache_resource
# def load_models():
#     classifier = pipeline("text-classification", model=MODEL_NAME)
#     sentiment = pipeline("sentiment-analysis")
#     return classifier, sentiment

# classifier, sentiment = load_models()

# text = st.text_area("Enter customer complaint")

# if st.button("Analyze"):
#     if text:
#         category = classifier(text)[0]
#         senti = sentiment(text)[0]

#         st.subheader("Results")
#         st.write("Category:", category["label"])
#         st.write("Confidence:", round(category["score"], 2))
#         st.write("Sentiment:", senti["label"])



# import streamlit as st
# from transformers import pipeline
# from urgency import detect_urgency
# from risk_engine import compute_priority
# from response_generator import generate_response

# MODEL_NAME = "DhanushGWU1995/ecris-category-model"
# # Load models
# category_classifier = pipeline("text-classification", model=MODEL_NAME)
# sentiment_model = pipeline("sentiment-analysis")

# st.title("Enterprise Customer Risk Intelligence System (ECRIS)")

# text = st.text_area("Enter customer complaint")

# if st.button("Analyze"):
#     category = category_classifier(text)[0]["label"]
#     sentiment = sentiment_model(text)[0]["label"]
#     urgency = detect_urgency(text)
#     priority = compute_priority(category, sentiment, urgency, len(text))
#     response = generate_response(priority)

#     st.write("### Results")
#     st.write("Category:", category)
#     st.write("Sentiment:", sentiment)
#     st.write("Urgency:", "High" if urgency else "Normal")
#     st.write("Priority Score:", priority)
#     st.write("Suggested Response:")
#     st.write(response)


import streamlit as st
from transformers import pipeline

# ==============================
# Configuration
# ==============================
MODEL_NAME = "DhanushGWU1995/ecris-category-model"

st.set_page_config(page_title="ECRIS", layout="wide")
st.title("Enterprise Customer Risk Intelligence System (ECRIS)")
st.write("AI-powered real-time complaint risk analysis and prioritization")

# ==============================
# Load Models (Cached)
# ==============================
@st.cache_resource
def load_models():
    category_model = pipeline("text-classification", model=MODEL_NAME)
    sentiment_model = pipeline("sentiment-analysis")
    return category_model, sentiment_model

category_model, sentiment_model = load_models()

# ==============================
# Enterprise Risk Engine
# ==============================

HIGH_RISK_CATEGORIES = [
    "Debt collection",
    "Credit reporting",
    "Credit card"
]

MEDIUM_RISK_CATEGORIES = [
    "Mortgage",
    "Bank account or service"
]

URGENT_KEYWORDS = [
    "urgent",
    "immediately",
    "asap",
    "fraud",
    "scam",
    "lawsuit",
    "legal",
    "regulator",
    "complaint authority",
    "refund",
    "unauthorized"
]


def detect_urgency(text):
    text_lower = text.lower()
    for word in URGENT_KEYWORDS:
        if word in text_lower:
            return 1
    return 0


def category_score(category):
    if category in HIGH_RISK_CATEGORIES:
        return 3
    elif category in MEDIUM_RISK_CATEGORIES:
        return 2
    else:
        return 1


def sentiment_score(sentiment):
    if sentiment == "NEGATIVE":
        return 3
    elif sentiment == "NEUTRAL":
        return 2
    else:
        return 1


def length_score(text_length):
    if text_length > 400:
        return 2
    elif text_length > 150:
        return 1
    else:
        return 0


def compute_priority(category, sentiment, urgency, text_length):
    score = (
        category_score(category) * 2 +
        sentiment_score(sentiment) * 1.5 +
        urgency * 3 +
        length_score(text_length)
    )
    return round(min(score, 10), 2)


def explain_priority(category, sentiment, urgency, text_length):
    reasons = []

    if category in HIGH_RISK_CATEGORIES:
        reasons.append("High-risk product category")

    if sentiment == "NEGATIVE":
        reasons.append("Negative customer sentiment")

    if urgency:
        reasons.append("Urgent or escalation language detected")

    if text_length > 150:
        reasons.append("Detailed complaint (higher seriousness)")

    if not reasons:
        reasons.append("Standard service request")

    return reasons


def generate_response(priority):
    if priority >= 8:
        return ("We sincerely apologize for the inconvenience. "
                "Your case has been escalated to our priority support team "
                "and will be addressed immediately.")
    elif priority >= 5:
        return ("Thank you for bringing this issue to our attention. "
                "Our support team is reviewing your case and will respond shortly.")
    else:
        return ("Thank you for contacting us. "
                "We have received your request and our team will assist you soon.")


# ==============================
# UI
# ==============================
text = st.text_area("Enter Customer Complaint", height=200)

if st.button("Analyze Complaint"):
    if text.strip() == "":
        st.warning("Please enter complaint text")
    else:
        # Model predictions
        category_result = category_model(text)[0]
        sentiment_result = sentiment_model(text)[0]

        category = category_result["label"]
        sentiment = sentiment_result["label"]
        urgency = detect_urgency(text)
        priority = compute_priority(category, sentiment, urgency, len(text))
        reasons = explain_priority(category, sentiment, urgency, len(text))
        response = generate_response(priority)

        # ==============================
        # Display Results
        # ==============================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("AI Analysis")
            st.write("**Category:**", category)
            st.write("**Confidence:**", round(category_result["score"], 2))
            st.write("**Sentiment:**", sentiment)
            st.write("**Urgency:**", "High" if urgency else "Normal")

        with col2:
            st.subheader("Risk Assessment")
            st.metric("Priority Score (0-10)", priority)

            if priority >= 8:
                st.error("High Risk - Immediate Attention Required")
            elif priority >= 5:
                st.warning("Medium Risk - Priority Handling")
            else:
                st.success("Low Risk - Standard Queue")

        st.subheader("Why this score?")
        for r in reasons:
            st.write("- ", r)

        st.subheader("Suggested Response")
        st.write(response)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("ECRIS | Enterprise AI Complaint Intelligence | Hugging Face Deployment")
