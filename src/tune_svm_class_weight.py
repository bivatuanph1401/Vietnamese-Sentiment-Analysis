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
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)


# =========================
# 3. So sánh class_weight
# =========================

weights = [
    None,
    "balanced"
]

results = []

print("===== TUNE SVM CLASS WEIGHT =====")

for weight in weights:

    model = LinearSVC(
        C=0.75,
        class_weight=weight
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((weight, accuracy))

    print(
        f"class_weight = {str(weight):<10} | "
        f"Accuracy = {accuracy:.6f}"
    )


# =========================
# 4. Chọn tốt nhất
# =========================

best_weight, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST CLASS WEIGHT =====")
print("Best class_weight:", best_weight)
print("Best validation Accuracy:", best_accuracy)


# =========================
# 5. Classification Report
# =========================

best_model = LinearSVC(
    C=0.75,
    class_weight=best_weight
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