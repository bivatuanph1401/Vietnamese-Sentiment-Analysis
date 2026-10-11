import pandas as pd
import os


# Tạo thư mục lưu dữ liệu đã xử lý
os.makedirs("data/clean", exist_ok=True)


def clean_data(input_file, output_file):
    # Đọc dữ liệu
    df = pd.read_csv(input_file)

    print(f"\nĐang xử lý: {input_file}")
    print("Số dòng ban đầu:", len(df))

    # Xóa các dòng thiếu text hoặc label
    df = df.dropna(subset=["text", "label"])

    # Chuyển text về kiểu chuỗi
    df["text"] = df["text"].astype(str)

    # Xóa khoảng trắng đầu/cuối
    df["text"] = df["text"].str.strip()

    # Xóa các text rỗng sau khi strip
    df = df[df["text"] != ""]

    # Xóa dòng trùng hoàn toàn
    df = df.drop_duplicates()

    # Lưu dữ liệu sạch
    df.to_csv(output_file, index=False)

    print("Số dòng sau khi xử lý:", len(df))
    print("Đã lưu:", output_file)


# Xử lý từng dataset
clean_data(
    "data/UIT-VSFC_train.csv",
    "data/clean/train.csv"
)

clean_data(
    "data/UIT-VSFC_valid.csv",
    "data/clean/valid.csv"
)

clean_data(
    "data/UIT-VSFC_test.csv",
    "data/clean/test.csv"
)