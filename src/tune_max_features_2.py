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


max_features_values = [30000, 40000, 50000]

results = []

print("===== TUNE max_features - ROUND 2 =====")

for max_features in max_features_values:

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_valid = vectorizer.transform(X_valid_text)

    model = LinearSVC(C=0.75)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    results.append((max_features, accuracy))

    print(
        f"max_features = {max_features:<5} | "
        f"Actual features = {X_train.shape[1]:<6} | "
        f"Accuracy = {accuracy:.6f}"
    )


best_max_features, best_accuracy = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST max_features =====")
print("Best max_features:", best_max_features)
print("Best validation Accuracy:", best_accuracy)


vectorizer = TfidfVectorizer(
    max_features=best_max_features,
    ngram_range=(1, 2),
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