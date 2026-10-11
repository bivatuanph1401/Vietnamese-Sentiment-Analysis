import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


MODEL_DIR = "models/phobert/final"
MAX_LENGTH = 128

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()


# ==================================================
# TEST DATA
# ==================================================

tests = {

    "Ngắn": [
        ("Tệ", 0),
        ("Tốt", 2),
        ("Ổn", 1),
        ("Tuyệt", 2),
        ("Chán", 0),
        ("Bình thường", 1),
        ("Rất tốt", 2),
        ("Quá tệ", 0),
        ("Không ổn", 0),
        ("Được", 1),
    ],

    "Trung bình": [
        ("Sản phẩm này rất tốt", 2),
        ("Chất lượng khá tệ", 0),
        ("Mình thấy cũng bình thường", 1),
        ("Sản phẩm dùng rất ổn", 2),
        ("Mình không thích sản phẩm này", 0),
        ("Chất lượng không có gì đặc biệt", 1),
        ("Mình khá hài lòng với sản phẩm", 2),
        ("Sản phẩm làm mình thất vọng", 0),
        ("Mình thấy sản phẩm khá bình thường", 1),
        ("Sản phẩm dùng được nhưng chưa tốt lắm", 1),
    ],

    "Dài": [
        (
            "Sản phẩm này có chất lượng rất tốt, thiết kế đẹp và sử dụng "
            "cũng rất tiện lợi nên mình cảm thấy khá hài lòng.",
            2
        ),
        (
            "Mình đã sử dụng sản phẩm được một thời gian và cảm thấy "
            "chất lượng khá tệ, không giống như những gì cửa hàng mô tả.",
            0
        ),
        (
            "Theo mình thì sản phẩm này ở mức bình thường, không quá tốt "
            "nhưng cũng không có vấn đề gì nghiêm trọng khi sử dụng.",
            1
        ),
        (
            "Mình rất thích sản phẩm này vì chất lượng tốt, giao hàng nhanh "
            "và cách đóng gói cũng rất cẩn thận.",
            2
        ),
        (
            "Sản phẩm khiến mình khá thất vọng vì chất lượng không tốt, "
            "sử dụng không thuận tiện và thời gian giao hàng cũng lâu.",
            0
        ),
        (
            "Mình nghĩ sản phẩm này khá ổn, chất lượng đúng với mô tả "
            "nhưng không có điểm gì đặc biệt khiến mình ấn tượng.",
            1
        ),
        (
            "Sau khi sử dụng vài ngày mình thấy sản phẩm rất tốt, hoạt động "
            "ổn định và đúng với những gì mình mong đợi ban đầu.",
            2
        ),
        (
            "Mình không hài lòng với sản phẩm vì chất lượng thấp, thiết kế "
            "không đẹp và trải nghiệm sử dụng cũng không được tốt.",
            0
        ),
        (
            "Sản phẩm nhìn chung khá bình thường, không có ưu điểm nổi bật "
            "nhưng cũng không có nhược điểm quá nghiêm trọng.",
            1
        ),
        (
            "Mình thấy sản phẩm đáp ứng được nhu cầu cơ bản, tuy nhiên "
            "chất lượng chỉ ở mức trung bình và chưa thật sự nổi bật.",
            1
        ),
    ]
}


# ==================================================
# PREDICTION
# ==================================================

for category, samples in tests.items():

    print("\n" + "=" * 60)
    print(category.upper())
    print("=" * 60)

    y_true = []
    y_pred = []

    for text, true_label in samples:

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoding = {
            key: value.to(device)
            for key, value in encoding.items()
        }

        with torch.no_grad():

            outputs = model(**encoding)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

            prediction = torch.argmax(
                probabilities
            ).item()

        y_true.append(true_label)
        y_pred.append(prediction)

        print(
            f"\nText: {text}"
        )

        print(
            f"Actual: {LABELS[true_label]}"
        )

        print(
            f"Predicted: {LABELS[prediction]}"
        )

        print(
            f"Confidence: "
            f"{probabilities[prediction].item() * 100:.2f}%"
        )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n------------------------------")
    print(
        f"{category} Accuracy: {accuracy * 100:.2f}%"
    )


print("\n===== TEST COMPLETE =====")