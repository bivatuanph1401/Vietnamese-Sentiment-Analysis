import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Đọc dữ liệu
# =========================

train = pd.read_csv("data/clean/train.csv")
valid = pd.read_csv("data/clean/valid.csv")

X_train_text = train["text"]
y_train = train["label"]

X_valid_text = valid["text"]
y_valid = valid["label"]


# =========================
# 2. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)


# =========================
# 3. Train Linear SVM
# =========================

model = LinearSVC(C=1.0)
model.fit(X_train, y_train)


# =========================
# 4. Decision scores
# =========================

scores = model.decision_function(X_valid)

classes = model.classes_

print("Classes:", classes)
print("Score shape:", scores.shape)


# =========================
# 5. Baseline
# =========================

baseline_pred = model.predict(X_valid)

baseline_accuracy = accuracy_score(
    y_valid,
    baseline_pred
)

print("\n===== BASELINE =====")
print("Accuracy:", baseline_accuracy)

print("\nClassification Report:")
print(classification_report(y_valid, baseline_pred))


# =========================
# 6. Tune Neutral threshold
# =========================

best_accuracy = 0
best_threshold = None
best_pred = None

neutral_index = list(classes).index("neutral")

# Thử nhiều threshold
thresholds = np.arange(-0.5, 0.51, 0.01)

for threshold in thresholds:

    pred = classes[np.argmax(scores, axis=1)].copy()

    neutral_mask = scores[:, neutral_index] > threshold

    pred[neutral_mask] = "neutral"

    accuracy = accuracy_score(y_valid, pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_pred = pred


# =========================
# 7. Kết quả tốt nhất
# =========================

print("\n===== BEST THRESHOLD =====")
print("Best threshold:", best_threshold)
print("Best validation Accuracy:", best_accuracy)

print("\nClassification Report:")
print(classification_report(y_valid, best_pred))