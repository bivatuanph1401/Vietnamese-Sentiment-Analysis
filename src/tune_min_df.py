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
# 2. Thử nhiều min_df
# =========================

min_df_values = [1, 2, 3, 5]

results = []

print("===== TUNE min_df =====")

for min_df in min_df_values:

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=min_df
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_valid = vectorizer.transform(X_valid_text)

    model = LinearSVC(C=0.75)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((min_df, accuracy))

    print(
        f"min_df = {min_df:<2} | "
        f"Features = {X_train.shape[1]:<6} | "
        f"Accuracy = {accuracy:.6f}"
    )


# =========================
# 3. Chọn min_df tốt nhất
# =========================

best_min_df, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST min_df =====")
print("Best min_df:", best_min_df)
print("Best validation Accuracy:", best_accuracy)


# =========================
# 4. Classification Report
# =========================

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=best_min_df
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)

best_model = LinearSVC(C=0.75)
best_model.fit(X_train, y_train)

best_pred = best_model.predict(X_valid)

print("\nClassification Report:")
print(classification_report(y_valid, best_pred))