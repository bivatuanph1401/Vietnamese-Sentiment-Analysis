
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =========================
# 1. Configuration
# =========================
MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dùng thư mục riêng, không ghi đè model cũ
CHECKPOINT_DIR = "models/phobert/checkpoint_augmented"
FINAL_DIR = "models/phobert/final_augmented"

TRAIN_PATH = "data/clean/train_augmented.csv"
VALID_PATH = "data/clean/valid.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

print("=" * 55)
print("PHOBERT TRAINING - AUGMENTED NEUTRAL DATA")
print("=" * 55)
print("Device:", DEVICE)
print("Train file:", TRAIN_PATH)
print("Validation file:", VALID_PATH)


# =========================
# 2. Labels
# =========================
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# =========================
# 3. Dataset
# =========================
class SentimentDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer

        unknown_labels = set(self.labels) - set(label2id.keys())

        if unknown_labels:
            raise ValueError(
                f"Nhãn không hợp lệ trong dữ liệu: {unknown_labels}"
            )

        self.labels = [
            label2id[label] for label in self.labels
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }


# =========================
# 4. Load data
# =========================
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

for name, dataframe in [
    ("Train", train_df),
    ("Validation", valid_df)
]:
    dataframe.dropna(subset=["text", "label"], inplace=True)
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = (
        dataframe["label"].astype(str).str.strip().str.lower()
    )
    dataframe.drop(
        dataframe[dataframe["text"] == ""].index,
        inplace=True
    )

    invalid = set(dataframe["label"]) - set(label2id.keys())
    if invalid:
        raise ValueError(f"{name} có nhãn không hợp lệ: {invalid}")

print("\n===== DATASET =====")
print("Train:", len(train_df))
print("Validation:", len(valid_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(valid_df["label"].value_counts())


# =========================
# 5. Tokenizer and DataLoader
# =========================
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SentimentDataset(train_df, tokenizer)
valid_dataset = SentimentDataset(valid_df, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# 6. Load or resume augmented checkpoint
# =========================
checkpoint_file = os.path.join(
    CHECKPOINT_DIR,
    "training_state.pt"
)

start_epoch = 0

if os.path.exists(checkpoint_file):
    print("\n===== RESUMING AUGMENTED TRAINING =====")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    start_epoch = checkpoint["epoch"] + 1

else:
    print("\n===== STARTING FROM BASE PHOBERT =====")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

model.to(DEVICE)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

if os.path.exists(checkpoint_file):
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Resuming from epoch {start_epoch + 1}")
else:
    print("Previous checkpoint will not be used.")

print("Model loaded successfully.")


# =========================
# 7. Training
# =========================
for epoch in range(start_epoch, EPOCHS):

    print("\n" + "=" * 55)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 55)

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0 or step == len(train_loader):
            print(
                f"Step {step}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"\nAverage Training Loss: {avg_loss:.4f}")


    # =========================
    # 8. Validation
    # =========================
    print("\nValidating...")

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in valid_loader:

            labels = batch["labels"].to(DEVICE)

            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n===== VALIDATION RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=["negative", "neutral", "positive"],
            digits=4,
            zero_division=0
        )
    )


    # =========================
    # 9. Save checkpoint
    # =========================
    print("Saving checkpoint...")

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "accuracy": accuracy,
            "macro_f1": macro_f1
        },
        checkpoint_file
    )

    print("Checkpoint saved successfully.")


# =========================
# 10. Save final model
# =========================
print("\n===== TRAINING COMPLETE =====")

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("Final model saved to:", FINAL_DIR)
