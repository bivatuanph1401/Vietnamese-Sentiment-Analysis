import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv("data/clean/train.csv")
valid = pd.read_csv("data/clean/valid.csv")

X_train_text = train["text"]
y_train = train["label"]

X_valid_text = valid["text"]
y_valid = valid["label"]


# TF-IDF cố định
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)


# Thử nhiều C
c_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

results = []

print("===== TUNE C ON VALIDATION =====")

for c in c_values:

    model = LinearSVC(C=c)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((c, accuracy))

    print(f"C = {c:<4} | Accuracy = {accuracy:.6f}")


# Tìm C tốt nhất
best_c, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST C =====")
print("Best C:", best_c)
print("Best validation Accuracy:", best_accuracy)


# Train lại model tốt nhất để xem report
best_model = LinearSVC(C=best_c)
best_model.fit(X_train, y_train)

best_pred = best_model.predict(X_valid)

print("\nClassification Report:")
print(classification_report(y_valid, best_pred))