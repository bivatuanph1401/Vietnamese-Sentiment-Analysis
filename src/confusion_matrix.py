import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

train = pd.read_csv("data/clean/train.csv")
test = pd.read_csv("data/clean/test.csv")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

y_train = train["label"]
y_test = test["label"]

model = LinearSVC(class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
)

print("===== CONFUSION MATRIX =====")
print("                    Predicted")
print("                  Neg   Neu   Pos")
print(f"Actual Negative   {cm[0][0]:4}  {cm[0][1]:4}  {cm[0][2]:4}")
print(f"Actual Neutral    {cm[1][0]:4}  {cm[1][1]:4}  {cm[1][2]:4}")
print(f"Actual Positive   {cm[2][0]:4}  {cm[2][1]:4}  {cm[2][2]:4}")