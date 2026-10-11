import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================
# 1. Đọc train và test
# =========================

train = pd.read_csv("data/clean/train.csv")
test = pd.read_csv("data/clean/test.csv")

X_train_text = train["text"]
y_train = train["label"]

X_test_text = test["text"]
y_test = test["label"]


# =========================
# 2. TF-IDF cấu hình cuối
# =========================

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# =========================
# 3. Linear SVM
# =========================

model = LinearSVC(C=0.75)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =========================
# 4. Accuracy
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("===== FINAL TEST RESULT =====")
print(f"Accuracy: {accuracy:.6f}")
print(f"Accuracy (%): {accuracy * 100:.4f}%")


# =========================
# 5. Classification Report
# =========================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# =========================
# 6. Confusion Matrix
# =========================

labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("===== CONFUSION MATRIX =====")
print("             Predicted")
print("             Neg  Neu  Pos")

for label, row in zip(labels, cm):
    print(
        f"Actual {label[:3].capitalize():<3}     "
        f"{row[0]:>4} {row[1]:>4} {row[2]:>4}"
    )