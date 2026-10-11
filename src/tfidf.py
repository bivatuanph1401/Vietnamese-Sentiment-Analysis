import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Đọc dữ liệu đã làm sạch
train = pd.read_csv("data/clean/train.csv")
valid = pd.read_csv("data/clean/valid.csv")
test = pd.read_csv("data/clean/test.csv")


# Tạo TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)


# Fit trên TRAIN
X_train = vectorizer.fit_transform(train["text"])

# Chỉ transform VALID và TEST
X_valid = vectorizer.transform(valid["text"])
X_test = vectorizer.transform(test["text"])


print("===== TF-IDF =====")
print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)

print("\nSố từ/vocabulary:", len(vectorizer.vocabulary_))