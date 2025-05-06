import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ------------------------ XỬ LÝ ẢNH ------------------------ #
def Erosion(imgin):
    w = cv2. getStructuringElement(cv2.MORPH_RECT, (45,45))
    imgout = cv2.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgout = cv2.erode(imgin,w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    return cv2.subtract(imgin, temp)

def Contour(imgin):
    img_color = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)  # Màu xanh
    return img_color

# ------------------------ CẤU HÌNH GIAO DIỆN ------------------------ #
st.set_page_config(
    page_title="Xử lý ảnh - Chương 9",
    page_icon="🖼️",
    layout="wide"
)

# CSS UI mới
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f3f7fa;
        }

        h1 {
            color: #1b2a41;
            text-align: center;
            padding: 20px 0;
        }

        .block-container {
            padding-top: 2rem;
        }

        .stButton>button, .stFileUploader, .stSelectbox>div {
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .stSelectbox>div>div {
            font-weight: 600;
        }

        .img-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }

        .img-caption {
            font-weight: 600;
            text-align: center;
            margin-top: 0.5rem;
            color: #1b2a41;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ GIAO DIỆN CHÍNH ------------------------ #
st.title("🖼️ Ứng dụng Xử lý Ảnh - Chương 9")

uploaded_file = st.file_uploader("📁 Chọn hình ảnh (jpg, png, tif)...", type=["jpg", "jpeg", "png", "tif"])

technique = st.selectbox("🔧 Chọn kỹ thuật xử lý ảnh", ["Erosion", "Dilation", "Boundary", "Contour"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    frame = np.array(image)

    with col1:
        with st.container():
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(frame, use_column_width=True, channels="L")
            st.markdown('<div class="img-caption">📷 Hình ảnh gốc (đen trắng)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Xử lý ảnh
    if technique == "Erosion":
        processed_img = Erosion(frame)
    elif technique == "Dilation":
        processed_img = Dilation(frame)
    elif technique == "Boundary":
        processed_img = Boundary(frame)
    elif technique == "Contour":
        processed_img = Contour(frame)

    with col2:
        with st.container():
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(processed_img, use_column_width=True)
            st.markdown(f'<div class="img-caption">🛠️ Ảnh sau khi áp dụng: {technique}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
