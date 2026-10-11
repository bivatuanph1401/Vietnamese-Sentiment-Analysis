import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

MODEL_DIR = "models/phobert/final"
TEST_FILE = "data/clean/test.csv"

MAX_LENGTH = 128
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Loading PhoBERT...")


# =========================
# Load model + tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


# =========================
# Load test data
# =========================

df = pd.read_csv(TEST_FILE)

texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

label_names = ["negative", "neutral", "positive"]

# Nếu label trong CSV là chữ
if isinstance(labels[0], str):
    label_to_id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    labels = [label_to_id[x] for x in labels]


# =========================
# Dataset
# =========================

class SentimentDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


dataset = SentimentDataset(texts, labels)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Prediction
# =========================

all_predictions = []
all_labels = []

print("\n===== TESTING =====")

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch_labels.cpu().numpy()
        )

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Processed: {batch_idx + 1}/{len(loader)} batches"
            )


# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

print("\n==============================")
print("       PHOBERT TEST RESULT")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {f1:.4f}")


# =========================
# Classification report
# =========================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=label_names,
        digits=4,
        zero_division=0
    )
)


# =========================
# Confusion Matrix
# =========================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("             Predicted")
print("             Neg  Neu  Pos")

print(
    f"Actual Neg   {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}"
)

print(
    f"Actual Neu   {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}"
)

print(
    f"Actual Pos   {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}"
)


# =========================
# Save results
# =========================

os.makedirs("results", exist_ok=True)

result_df = pd.DataFrame({
    "text": texts,
    "actual": all_labels,
    "predicted": all_predictions
})

result_df.to_csv(
    "results/phobert_test_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("results/phobert_test_predictions.csv")

print("\n===== DONE =====")