from pathlib import Path

import pandas as pd
from datasets import Dataset

def load_data():
    # Load CSV locally
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "complaints.csv"
    df = pd.read_csv(csv_path)

    # Keep required columns
    df = df[["Consumer complaint narrative", "Product"]]

    # Rename columns to match project
    df = df.rename(columns={
        "Consumer complaint narrative": "consumer_complaint_narrative",
        "Product": "product"
    })

    # Remove missing text
    df = df.dropna()

    # Remove short complaints
    df = df[df["consumer_complaint_narrative"].str.len() > 50]

    # Keep only top categories
    top_products = [
        "Credit card",
        "Mortgage",
        "Debt collection",
        "Credit reporting",
        "Bank account or service"
    ]

    df = df[df["product"].isin(top_products)]

    # Reduce size for faster training
    df = df.sample(20000, random_state=42)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Train-test split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    return dataset
