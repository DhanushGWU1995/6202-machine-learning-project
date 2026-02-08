import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from data_processing import load_data

dataset = load_data()
train_ds = dataset["train"]
test_ds = dataset["test"]

labels = train_ds.unique("product")
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def encode(example):
    example["label"] = label2id[example["product"]]
    return example

train_ds = train_ds.map(encode)
test_ds = test_ds.map(encode)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["consumer_complaint_narrative"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# training_args = TrainingArguments(
#     output_dir="./models/category_model",
#     evaluation_strategy="epoch",
#     num_train_epochs=2,
#     per_device_train_batch_size=8,
# )

training_args = TrainingArguments(
    output_dir="./models/category_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_steps=500,
    do_train=True,
    do_eval=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=test_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.save_model("models/category_model")
tokenizer.save_pretrained("models/category_model")

metrics = trainer.evaluate()
print("Final Evaluation Metrics:")
print(metrics)

