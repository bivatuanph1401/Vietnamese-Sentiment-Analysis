import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# CẤU HÌNH
# =========================

MODEL_DIR = "models/phobert/final"
MAX_LENGTH = 128

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# =========================
# GIAO DIỆN
# =========================

st.set_page_config(
    page_title="Phân tích cảm xúc tiếng Việt",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Phân tích cảm xúc bình luận tiếng Việt")

st.write(
    "Hệ thống sử dụng mô hình **PhoBERT** "
    "để phân loại cảm xúc của bình luận thành "
    "**Negative, Neutral hoặc Positive**."
)

st.divider()


# =========================
# NHẬP BÌNH LUẬN
# =========================

text = st.text_area(
    "Nhập bình luận:",
    placeholder="Ví dụ: Sản phẩm rất tốt, tôi rất hài lòng!",
    height=120
)


# =========================
# DỰ ĐOÁN
# =========================

if st.button("🔍 Phân tích cảm xúc", use_container_width=True):

    if text.strip() == "":
        st.warning("Vui lòng nhập một bình luận.")

    else:

        with st.spinner("Đang phân tích..."):

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

                outputs = model(
                    **encoding
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )[0]

                prediction = torch.argmax(
                    probabilities
                ).item()

                confidence = probabilities[
                    prediction
                ].item()


        label = LABELS[prediction]


        # =========================
        # HIỂN THỊ KẾT QUẢ
        # =========================

        st.subheader("Kết quả")

        if label == "Positive":
            st.success(
                f"😊 Positive — {confidence * 100:.2f}%"
            )

        elif label == "Negative":
            st.error(
                f"😠 Negative — {confidence * 100:.2f}%"
            )

        else:
            st.info(
                f"😐 Neutral — {confidence * 100:.2f}%"
            )


        st.write("### Xác suất dự đoán")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Negative",
                f"{probabilities[0].item() * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Neutral",
                f"{probabilities[1].item() * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Positive",
                f"{probabilities[2].item() * 100:.2f}%"
            )