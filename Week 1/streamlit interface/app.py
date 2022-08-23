import streamlit as st
from PIL import Image
from utils import image_aligner_api


class Widget:
    def __init__(self) -> None:
        pass

    @staticmethod
    def upload_image(header: str, desc: str) -> any:
        st.markdown(f"### {header}")
        file = st.file_uploader(
            f"{desc}", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
        if file:
            read_image = Image.open(file).convert('RGB')
            st.image(read_image)
            return read_image

    @staticmethod
    def show_image(image):
        st.image(image)

    @staticmethod
    def header(name: str):
        st.markdown(f"### {name}")


def web_head():
    st.set_page_config(
        page_title="Document Image Alignment", page_icon=":shark:"
    )
    st.title("Document Image Alignment")
    st.markdown("---")


def web_body():
    mock_algined_document = Widget.upload_image("Step 1 : Upload your mock *aligned* document",
                                                "Upload an empty document image (aligned)!")

    filled_unaligned_document = Widget.upload_image("Step 2 : Upload your *unaligned* document",
                                                    "Upload a filled document image (unaligned)!")

    if mock_algined_document and filled_unaligned_document:
        aligned_image, estimated_homography = image_aligner_api(
            mock_algined_document, filled_unaligned_document)

        st.markdown("---")
        Widget.header("Result Aligned Document")
        Widget.show_image(aligned_image)
        st.markdown("\n")
        st.markdown("__Estimated Homography__ :")
        st.code(f"{estimated_homography}")


def document_aligned_app():
    web_head()
    web_body()


if __name__ == "__main__":
    document_aligned_app()
