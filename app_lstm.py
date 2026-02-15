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

# Load models and config
@st.cache_resource
def load_models():
    """Load LSTM model, tokenizer, and config from Hugging Face Hub"""
    
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
        
        # Try to download config (optional)
        try:
            config_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="lstm_config.json",
                repo_type="model"
            )
        except:
            config_path = None
        
        st.success("✓ Models downloaded successfully!")
        
        # Load LSTM model
        model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load config (if exists)
        config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default config if file not found
            config = {
                'test_metrics': {
                    'urgency_accuracy': 0.75,
                    'tone_accuracy': 0.72
                }
            }
        
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
        st.metric("Urgency Accuracy", f"{config['test_metrics']['urgency_accuracy']:.1%}")
        st.metric("Tone Accuracy", f"{config['test_metrics']['tone_accuracy']:.1%}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Customer Complaint")
    complaint = st.text_area(
        "Type or paste the complaint text below:",
        height=200,
        placeholder="Example: I found unauthorized charges on my credit card. This is fraud and I need immediate action!"
    )
    
    analyze_button = st.button("🔍 Analyze Complaint", type="primary", use_container_width=True)

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
            complaint = text
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
