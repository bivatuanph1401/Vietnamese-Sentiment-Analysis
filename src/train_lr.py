import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv("data/clean/train.csv")
valid = pd.read_csv("data/clean/valid.csv")
test = pd.read_csv("data/clean/test.csv")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train["text"])
X_valid = vectorizer.transform(valid["text"])
X_test = vectorizer.transform(test["text"])

y_train = train["label"]
y_valid = valid["label"]
y_test = test["label"]

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)

print("===== VALIDATION =====")
print("Accuracy:", accuracy_score(y_valid, valid_pred))
print("\nClassification Report:")
print(classification_report(y_valid, valid_pred))

test_pred = model.predict(X_test)

print("\n===== TEST =====")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))