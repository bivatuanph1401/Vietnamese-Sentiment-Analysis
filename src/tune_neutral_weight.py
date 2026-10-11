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
# 3. Thử nhiều Neutral weight
# =========================

weights = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

results = []

print("===== TUNE NEUTRAL CLASS WEIGHT =====")

for neutral_weight in weights:

    model = LinearSVC(
        C=0.75,
        class_weight={
            "negative": 1.0,
            "neutral": neutral_weight,
            "positive": 1.0
        }
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((neutral_weight, accuracy))

    print(
        f"Neutral weight = {neutral_weight:<4} | "
        f"Accuracy = {accuracy:.6f}"
    )


# =========================
# 4. Chọn weight tốt nhất
# =========================

best_weight, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST NEUTRAL WEIGHT =====")
print("Best Neutral weight:", best_weight)
print("Best validation Accuracy:", best_accuracy)


# =========================
# 5. Classification Report
# =========================

best_model = LinearSVC(
    C=0.75,
    class_weight={
        "negative": 1.0,
        "neutral": best_weight,
        "positive": 1.0
    }
)

best_model.fit(X_train, y_train)

best_pred = best_model.predict(X_valid)

print("\nClassification Report:")
print(classification_report(y_valid, best_pred))