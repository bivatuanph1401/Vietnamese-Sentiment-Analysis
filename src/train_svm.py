import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Đọc dữ liệu
# =========================

train = pd.read_csv("data/clean/train.csv")
test = pd.read_csv("data/clean/test.csv")

X_train_text = train["text"]
y_train = train["label"]

X_test_text = test["text"]
y_test = test["label"]


# =========================
# 2. Word-level TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True
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
# 4. Đánh giá
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("===== WORD TF-IDF + LINEAR SVM =====")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))