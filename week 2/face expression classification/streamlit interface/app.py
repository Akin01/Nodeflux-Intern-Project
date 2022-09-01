import streamlit as st
from PIL import Image
from utils import classifier
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from stream_service import live_inference

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


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
        page_title="Expression Classifier", page_icon=":shark:"
    )
    st.title("Face Expression Classification")
    st.markdown("---")


def file_upload_page():
    face_image = Widget.upload_image(
        "Step 1 : Upload your face image!", "Your Face Expression")

    if face_image:
        label, inference_time, pred_score = classifier(face_image)

        st.markdown("---")
        Widget.header("Result")
        st.markdown(f"Expression : ```{label}```")
        st.markdown(f"Inference Time : ```{round(inference_time, 2)} s```")
        st.markdown(f"Confidence Score : ```{round(pred_score * 100, 2)} %```")


def stream_video_page():
    st.header("Webcam Live Inference")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="4h3huht4hj3", mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=live_inference.Faceemotion)


def web_body():
    sidebar_options = ["Upload File", "Webcam Stream"]
    choice = st.sidebar.selectbox("Select Options", sidebar_options)

    if choice == "Upload File":
        file_upload_page()

    elif choice == "Webcam Stream":
        stream_video_page()


def face_classification():
    web_head()
    web_body()


if __name__ == "__main__":
    face_classification()
