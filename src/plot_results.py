import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# =========================
# Accuracy
# =========================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Model"],
    df["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("So sánh Accuracy giữa các mô hình")
plt.ylim(0, 100)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("accuracy_comparison.png", dpi=200)

plt.show()


# =========================
# Macro F1
# =========================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Model"],
    df["Macro_F1"] * 100
)

plt.ylabel("Macro F1 (%)")
plt.xlabel("Model")
plt.title("So sánh Macro F1 giữa các mô hình")
plt.ylim(0, 100)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("macro_f1_comparison.png", dpi=200)

plt.show()

print("Đã tạo:")
print("accuracy_comparison.png")
print("macro_f1_comparison.png")