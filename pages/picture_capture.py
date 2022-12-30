import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import PIL.ImageOps
from io import BytesIO
import cv2
from src.predictions import get_detections, non_maximum_supression, drawings
from src.image_load import transform_image


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

def load_model(model_path):
    #model = torch.load(model_path, map_location='cpu')
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    final_dict = drawings(img,boxes_np,confidences_np,index)
    return final_dict


def main():

    st.title('License Plate Detection')
    model = load_model("./models/weights/best.onnx")
    img_file_buffer = st.camera_input("Take a picture")
    col1, col2 = st.columns(2)
    np_image = load_image(img_file_buffer, col1)

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