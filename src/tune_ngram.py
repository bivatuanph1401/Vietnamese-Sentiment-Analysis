import pandas as pd

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
# 2. Thử các ngram_range
# =========================

ngram_values = [
    (1, 1),
    (1, 2),
    (1, 3)
]

results = []

print("===== TUNE NGRAM_RANGE =====")

for ngram_range in ngram_values:

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_valid = vectorizer.transform(X_valid_text)

    model = LinearSVC(C=0.75)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((ngram_range, accuracy))

    print(
        f"ngram_range = {ngram_range} | "
        f"Features = {X_train.shape[1]} | "
        f"Accuracy = {accuracy:.6f}"
    )


# =========================
# 3. Chọn cấu hình tốt nhất
# =========================

best_ngram, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST NGRAM =====")
print("Best ngram_range:", best_ngram)
print("Best validation Accuracy:", best_accuracy)


# =========================
# 4. Classification Report
# =========================

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=best_ngram,
    sublinear_tf=True,
    min_df=1
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)

best_model = LinearSVC(C=0.75)

best_model.fit(X_train, y_train)

best_pred = best_model.predict(X_valid)

print("\nClassification Report:")
print(classification_report(y_valid, best_pred))