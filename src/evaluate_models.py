import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# =========================
# 1. Load dữ liệu
# =========================

train = pd.read_csv("data/clean/train.csv")
test = pd.read_csv("data/clean/test.csv")

X_train_text = train["text"]
y_train = train["label"]

X_test_text = test["text"]
y_test = test["label"]


# =========================
# 2. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# =========================
# 3. Khai báo các mô hình
# =========================

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Linear SVM": LinearSVC(),

    "Linear SVM Balanced": LinearSVC(
        class_weight="balanced"
    )
}


# =========================
# 4. Huấn luyện + đánh giá
# =========================

results = []

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1
    })


# =========================
# 5. Lưu kết quả
# =========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results_full.csv",
    index=False
)

print("\n==============================")
print("KẾT QUẢ")
print("==============================")

print(results_df.to_string(index=False))

print("\nĐã lưu: results_full.csv")