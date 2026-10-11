import pickle

# Load TF-IDF
with open("models/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
with open("models/svm_balanced.pkl", "rb") as f:
    model = pickle.load(f)

# Các câu thử nghiệm
texts = [
    "Giảng viên dạy rất hay và dễ hiểu",
    "Môn học này thật sự rất tệ",
    "Môn học cũng bình thường, không có gì đặc biệt"
]

# Chuyển câu thành TF-IDF
X = vectorizer.transform(texts)

# Dự đoán
predictions = model.predict(X)

print("===== PREDICTION =====")

for text, prediction in zip(texts, predictions):
    print(f"\nCâu: {text}")
    print(f"Dự đoán: {prediction}")