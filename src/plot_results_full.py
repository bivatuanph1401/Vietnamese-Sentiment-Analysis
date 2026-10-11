import pandas as pd
import matplotlib.pyplot as plt

# Đọc kết quả
df = pd.read_csv("results_full.csv")

metrics = {
    "Accuracy": "Accuracy (%)",
    "Macro_Precision": "Macro Precision (%)",
    "Macro_Recall": "Macro Recall (%)",
    "Macro_F1": "Macro F1 (%)"
}

for column, ylabel in metrics.items():

    plt.figure(figsize=(9, 5))

    plt.bar(
        df["Model"],
        df[column] * 100
    )

    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(f"So sánh {ylabel.replace('(%)', '').strip()} giữa các mô hình")
    plt.ylim(0, 100)

    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = column.lower() + "_comparison.png"

    plt.savefig(filename, dpi=200)

    plt.show()

    print("Đã tạo:", filename)