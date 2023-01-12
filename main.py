import streamlit as st
import io
from PIL import Image
import numpy as np
from detection.object_detection import detect_object
import time


def main():
    st.title("Распознование объектов с помощью YOLOv4")
    img_array = load_image()
    result = st.button('Распознать изображение')
    if result:
        if isinstance(img_array, np.ndarray):
            image = detect_object(img_array)
            st.write('**Результаты распознавания:**')
            with st.spinner('Ожидайте загрузку'):
                time.sleep(2)
            st.image(image)
            st.success('Успешное распознование', icon="✅")


def load_image():
    uploaded_image = st.file_uploader("Пожалуйста выберите фото", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
        except Exception:
            st.error("Error: Неверная фотография")
        else:
            image = image.convert("RGB")
            img_array = np.array(image)
            return img_array


if __name__ == '__main__':
    main()