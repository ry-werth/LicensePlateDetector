import streamlit as st



if __name__ == '__main__':
    st.set_page_config(page_title="Main Page", page_icon="")

    st.title('License Plate Detection')

    st.markdown('Using **Yolov5** trained a model to identify license plates in an image')
    st.markdown('Using **Pytesseract** I extract the text from the license plate')
    st.markdown("""
    Try it yourself. Navigate to either the picture capture or upload page on the right.
    """)

    balls = st.button("I'm Just Here For Balloons")
    if balls:
        st.balloons()

