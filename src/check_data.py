import pandas as pd

train = pd.read_csv("data/UIT-VSFC_train.csv")
valid = pd.read_csv("data/UIT-VSFC_valid.csv")
test = pd.read_csv("data/UIT-VSFC_test.csv")


print("===== VALIDATION - DÒNG TRỐNG =====")

empty_rows = valid[
    valid["text"].isna() |
    valid["label"].isna()
]

print(empty_rows)