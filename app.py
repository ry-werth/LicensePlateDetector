import streamlit as st
from os import listdir
from PIL import Image
from src.predictions import yolo_predictions
from src.image_load import transform_image
from src.model_load import load_model

def get_test_image_names():
    files = [f for f in listdir("images")]
    return sorted(files)

def load_image(file, col1):
    if file is not None:
        image = Image.open(f'images/{file}')
        np_img = transform_image(image)
        with col1:
            st.image(np_img)
        return np_img
    else:
        return None


if __name__ == '__main__':
    st.set_page_config(page_title="Main Page", page_icon="")

    st.title('License Plate Detection')

    st.markdown('Using [**Yolov5**](https://pjreddie.com/darknet/yolo/) I utilized transfer learning to train a model to identify license plates in an image')
    st.markdown('[**Open CV DNN Module**](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) is used to run an image through the model')
    st.markdown('With [**Pytesseract**](https://pypi.org/project/pytesseract/) I _attempt to_ extract the text from the license plate')
    st.markdown("""
    For a quick example, simply select a test image below.
    To try it on your own image, navigate to the picture capture or picture upload page on the right.
    """)

    model = load_model("./models/weights/best.onnx")
    images_list = get_test_image_names()


    option = st.selectbox(
    'Select an image to test on',
    images_list)

    st.write('You selected:', option)

    col1, col2 = st.columns(2)

    np_image = load_image(option, col1)

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

    balls = st.button("Show me some balloons!")
    if balls:
        st.balloons()

    st.markdown("My [Github](https://github.com/ry-werth/LicensePlateDetector/tree/main)  	:nerd_face:")

