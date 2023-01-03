import streamlit as st
import PIL.Image
import PIL.ImageOps
from io import BytesIO
from src.predictions import yolo_predictions
from src.image_load import transform_image
from src.model_load import load_model


def load_image(uploaded_file, col1):
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        img = PIL.Image.open(BytesIO(image_data))
        np_img = transform_image(img)
        with col1:
            st.image(np_img)
        return np_img
    else:
        return None


def main():

    st.title('License Plate Detection')
    model = load_model("./models/weights/best.onnx")
    uploaded_file = st.file_uploader(label='Pick an image to test')
    col1, col2 = st.columns(2)
    np_image = load_image(uploaded_file, col1)

    with col1:
        loaded_result = st.button('Run on loaded image')

    if loaded_result:
        if np_image is not None:
            result_img_dict = yolo_predictions(np_image,model)
            with col2:
                st.image(result_img_dict["full_image"])
                st.write(f'Confidence Level **{round(result_img_dict["conf"], 2)}**')
                st.write("**Extracted Plate Image**")
                st.image(result_img_dict["plate_img"])
                st.markdown(f'Extracted Plate Text: **{result_img_dict["text"]}**')
                
            
        else:
            st.write('Please Upload a Photo')


if __name__ == '__main__':

    main()