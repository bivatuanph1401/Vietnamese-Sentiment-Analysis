
import pandas as pd
from pathlib import Path

train_path = Path("data/clean/train.csv")
neutral_path = Path("data/neutral_500.csv")
output_path = Path("data/clean/train_augmented.csv")

train = pd.read_csv(train_path)
neutral = pd.read_csv(neutral_path)

# Chỉ giữ các câu có đủ nội dung và nhãn
neutral = neutral.dropna(subset=["text", "label"])
neutral["text"] = neutral["text"].astype(str).str.strip()
neutral["label"] = neutral["label"].astype(str).str.strip().str.lower()
neutral = neutral[
    (neutral["text"] != "") &
    (neutral["label"] == "neutral")
]

# Loại câu trùng với dữ liệu train ban đầu
existing_texts = set(train["text"].astype(str).str.strip())
neutral = neutral[
    ~neutral["text"].astype(str).str.strip().isin(existing_texts)
]

# Gộp nhưng không sửa dữ liệu train gốc
augmented = pd.concat([train, neutral], ignore_index=True)
augmented = augmented.drop_duplicates(subset=["text", "label"])
augmented.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Train gốc:", len(train))
print("Câu Neutral mới được thêm:", len(neutral))
print("Train sau khi gộp:", len(augmented))
print("\nPhân bố nhãn:")
print(augmented["label"].value_counts())
print("\nĐã lưu:", output_path)
