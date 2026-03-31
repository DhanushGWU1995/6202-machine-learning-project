"""
ECRIS - Enterprise Customer Risk Intelligence System
Streamlit App for Complaint Analysis using LSTM

This app predicts:
1. Customer Tone (Negative/Neutral/Positive)
2. Urgency Rating (Low/Medium/High/Critical)
"""

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="ECRIS - Complaint Analyzer",
    page_icon="🚨",
    layout="wide"
)

# Model repository
MODEL_REPO = "DhanushGWU1995/ecris-category-model"


def _load_json_if_exists(path: str | Path) -> dict:
    p = Path(path)
    if p.exists() and p.is_file():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _merge_dicts(base: dict, override: dict) -> dict:
    """Shallow merge of dicts (override wins)."""
    merged = dict(base or {})
    merged.update(override or {})
    return merged

# Load models and config
@st.cache_resource
def load_models():
    """Load LSTM model/tokenizer and config.

    Priority for config metrics:
    1) Local Space artifacts under ./reports (uploaded by hf_push_space.py)
    2) HF model repo (MODEL_REPO)
    3) Hard-coded defaults
    """
    
    try:
        st.info(f"📥 Downloading models from Hugging Face: {MODEL_REPO}")
        
        # Download model files from HF Hub
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="lstm_complaint_classifier.h5",
            repo_type="model"
        )
        
        tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="lstm_tokenizer.pkl",
            repo_type="model"
        )
        
        # Prefer local config for Spaces (since we now upload reports/)
        local_config = _load_json_if_exists("reports/lstm_config.json")
        local_metrics_summary = _load_json_if_exists("reports/metrics_summary.json")

        # Try to download config from HF model repo (optional fallback)
        config_path = None
        try:
            config_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="lstm_config.json",
                repo_type="model",
            )
        except Exception:
            config_path = None
        
        st.success("✓ Models downloaded successfully!")
        
        # Load LSTM model
        model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Default config (ensures app doesn't crash if keys are missing)
        config: dict = {
            "max_len": 150,
            "urgency_labels": ["Low", "Medium", "High", "Critical"],
            "tone_labels": ["Negative", "Neutral", "Positive"],
            "test_metrics": {
                "urgency_accuracy": 0.75,
                "tone_accuracy": 0.72,
            },
        }

        # Merge in HF config then local config (local wins)
        hf_config = {}
        if config_path and Path(config_path).exists():
            hf_config = _load_json_if_exists(config_path)
        config = _merge_dicts(config, hf_config)
        config = _merge_dicts(config, local_config)

        # If metrics_summary.json contains relevant metrics, let it override display metrics.
        # (metrics_summary.json currently tracks classical model results; we only use it
        # when it includes keys we recognize.)
        if isinstance(local_metrics_summary, dict) and local_metrics_summary.get("lstm_test_metrics"):
            config["test_metrics"] = _merge_dicts(config.get("test_metrics", {}), local_metrics_summary["lstm_test_metrics"])
        
        return model, tokenizer, config
        
    except Exception as e:
        st.error(f"Error loading models from Hugging Face: {e}")
        st.error(f"Repository: {MODEL_REPO}")
        st.info("💡 Make sure the repository is public or you have access to it.")
        return None, None, None

lstm_model, tokenizer, config = load_models()

# UI Header
st.title("🚨 ECRIS: Enterprise Customer Risk Intelligence System")
st.markdown("---")
st.subheader("AI-Powered Complaint Analysis using Bidirectional LSTM")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **ECRIS** uses advanced deep learning (Bidirectional LSTM) to analyze customer complaints and predict:
    
    - **🔥 Urgency Rating**: How quickly the complaint needs attention
    - **😊 Customer Tone**: The emotional state of the customer
    
    This helps financial institutions prioritize complaints and route them to appropriate teams.
    """)
    
    if config:
        st.markdown("---")
        st.header("📊 Model Performance")
        test_metrics = config.get("test_metrics", {}) or {}
        urg_acc = float(test_metrics.get("urgency_accuracy", 0.0))
        tone_acc = float(test_metrics.get("tone_accuracy", 0.0))
        st.metric("Urgency Accuracy", f"{urg_acc:.2%}")
        st.metric("Tone Accuracy", f"{tone_acc:.2%}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Customer Complaint")
    if "complaint_text" not in st.session_state:
        st.session_state["complaint_text"] = ""

    complaint = st.text_area(
        "Type or paste the complaint text below:",
        key="complaint_text",
        height=200,
        placeholder="Example: I found unauthorized charges on my credit card. This is fraud and I need immediate action!"
    )

    action_col1, action_col2 = st.columns([3, 1])
    with action_col1:
        analyze_button = st.button("🔍 Analyze Complaint", type="primary", use_container_width=True)
    with action_col2:
        if st.button("🧽 Clear", use_container_width=True):
            st.session_state["complaint_text"] = ""
            st.rerun()

with col2:
    st.markdown("### 🎯 Try These Examples")
    
    examples = {
        "Critical - Fraud": "I found unauthorized charges on my credit card totaling $5,000. This is fraud! I'm contacting my lawyer immediately.",
        "High - Unresponsive": "I've been trying to resolve this issue for three weeks with no response. This is completely unacceptable!",
        "Medium - Billing Issue": "There's an error on my mortgage statement. I need someone to look into this and fix it.",
        "Low - General Question": "I have a question about my account statement. Can someone explain the fees?"
    }
    
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["complaint_text"] = text
            st.rerun()

# Analysis
if analyze_button and complaint.strip():
    if lstm_model is None:
        st.error("⚠️ Models not loaded. Please ensure model files are in the correct directory.")
    else:
        with st.spinner("🧠 Analyzing complaint..."):
            # Preprocess
            seq = tokenizer.texts_to_sequences([complaint])
            padded = pad_sequences(seq, maxlen=config['max_len'], padding='post', truncating='post')
            
            # Predict
            urgency_probs, tone_probs = lstm_model.predict(padded, verbose=0)
            
            urgency_labels = config['urgency_labels']
            tone_labels = config['tone_labels']
            
            urgency_idx = np.argmax(urgency_probs[0])
            tone_idx = np.argmax(tone_probs[0])
            
            urgency_confidence = urgency_probs[0][urgency_idx]
            tone_confidence = tone_probs[0][tone_idx]
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main predictions
        col1, col2 = st.columns(2)
        
        with col1:
            urgency_color = {
                "Low": "🟢",
                "Medium": "🟡",
                "High": "🟠",
                "Critical": "🔴"
            }
            st.markdown(f"### {urgency_color[urgency_labels[urgency_idx]]} Urgency Rating")
            st.markdown(f"# **{urgency_labels[urgency_idx]}**")
            st.markdown(f"Confidence: **{urgency_confidence*100:.1f}%**")
        
        with col2:
            tone_color = {
                "Negative": "😠",
                "Neutral": "😐",
                "Positive": "😊"
            }
            st.markdown(f"### {tone_color[tone_labels[tone_idx]]} Customer Tone")
            st.markdown(f"# **{tone_labels[tone_idx]}**")
            st.markdown(f"Confidence: **{tone_confidence*100:.1f}%**")
        
        # Detailed probabilities
        st.markdown("---")
        st.markdown("### 📈 Detailed Probability Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Urgency Probabilities")
            for i, label in enumerate(urgency_labels):
                prob = urgency_probs[0][i]
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
        
        with col2:
            st.markdown("#### Tone Probabilities")
            for i, label in enumerate(tone_labels):
                prob = tone_probs[0][i]
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommended Actions")
        
        if urgency_labels[urgency_idx] == "Critical":
            st.error("🚨 **IMMEDIATE ACTION REQUIRED**")
            st.markdown("""
            - Escalate to senior management immediately
            - Assign to fraud/legal team if applicable
            - Respond within 24 hours
            - Document all actions taken
            """)
        elif urgency_labels[urgency_idx] == "High":
            st.warning("⚠️ **HIGH PRIORITY**")
            st.markdown("""
            - Assign to experienced agent
            - Respond within 48 hours
            - Monitor for escalation
            """)
        elif urgency_labels[urgency_idx] == "Medium":
            st.info("📋 **STANDARD PRIORITY**")
            st.markdown("""
            - Route to appropriate department
            - Respond within 5 business days
            - Follow standard resolution process
            """)
        else:
            st.success("✅ **LOW PRIORITY**")
            st.markdown("""
            - Add to standard queue
            - Respond within 10 business days
            - Can be handled by any available agent
            """)
        
        # Tone-specific recommendations
        if tone_labels[tone_idx] == "Negative":
            st.markdown("**Customer Sentiment Alert:**")
            st.markdown("- Use empathetic language in response")
            st.markdown("- Consider offering compensation if appropriate")
            st.markdown("- Monitor for potential social media escalation")

elif analyze_button:
    st.warning("⚠️ Please enter a complaint to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>ECRIS</strong> - Enterprise Customer Risk Intelligence System</p>
    <p>Powered by Bidirectional LSTM | Developed for DATS 6202</p>
</div>
""", unsafe_allow_html=True)
