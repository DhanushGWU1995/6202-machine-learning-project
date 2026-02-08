# 6202-machine-learning-project
The Enterprise Customer Risk Intelligence System (ECRIS) is a real-time Natural Language Processing (NLP) application designed to help organizations automatically analyze and prioritize customer complaints.

# Dataset

https://www.consumerfinance.gov/data-research/consumer-complaints/

Project Title

Enterprise Customer Risk Intelligence System (ECRIS)

Overview

ECRIS is a real-time NLP system that analyzes customer complaints and predicts their business risk level using Hugging Face Transformers.

Features

Complaint category classification (DistilBERT)

Sentiment analysis

Urgency detection

Escalation risk scoring

Automated response generation (T5)

Real-time Streamlit dashboard

Dataset

Consumer Financial Protection Bureau (CFPB)
Loaded via Hugging Face:
consumer_finance_complaints

ML Pipeline (Strict)

Load data

Train-test split

Remove uncommon/short text

Remove identifiers

Handle missing data

Tokenization

Feature-target split

No scaling (Transformer-based)

Class imbalance handling

Feature engineering (text length, urgency)

Model training

Deployment

Hugging Face Space (Streamlit)

Results

Accuracy: ~85–90%

Weighted F1: ~0.87