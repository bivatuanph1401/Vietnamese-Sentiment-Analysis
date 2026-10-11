import pandas as pd
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Đọc dữ liệu train
train = pd.read_csv("data/clean/train.csv")

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train["text"])
y_train = train["label"]

# SVM Balanced
model = LinearSVC(class_weight="balanced")
model.fit(X_train, y_train)

# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)

# Lưu TF-IDF
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Lưu model
with open("models/svm_balanced.pkl", "wb") as f:
    pickle.dump(model, f)

print("===== SAVE MODEL =====")
print("Đã lưu:")
print("models/tfidf.pkl")
print("models/svm_balanced.pkl")