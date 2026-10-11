import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)


# =========================
# 3. Tune Logistic Regression
# =========================

C_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

results = []

print("===== TUNE LOGISTIC REGRESSION =====")

for C in C_values:

    model = LogisticRegression(
        C=C,
        max_iter=2000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((C, accuracy))

    print(
        f"C = {C:<4} | "
        f"Accuracy = {accuracy:.6f}"
    )


# =========================
# 4. Chọn C tốt nhất
# =========================

best_C, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST LOGISTIC REGRESSION =====")
print("Best C:", best_C)
print("Best validation Accuracy:", best_accuracy)


# =========================
# 5. Classification Report
# =========================

best_model = LogisticRegression(
    C=best_C,
    max_iter=2000
)

best_model.fit(X_train, y_train)

best_pred = best_model.predict(X_valid)

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        best_pred,
        digits=4
    )
)