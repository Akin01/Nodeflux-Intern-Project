import requests
from PIL import Image
from io import BytesIO
from base64 import decodebytes, encodebytes
import numpy as np

inference_url = 'http://host.docker.internal:5000/image/process'


def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode(
        'ascii')  # encode as base64
    return encoded_img


def decode_image(image_bytes):
    image_bytes = image_bytes.encode('ascii')
    image_bytes = decodebytes(image_bytes)
    image_bytes = BytesIO(image_bytes)
    image_bytes = Image.open(image_bytes)
    return np.array(image_bytes)


def classifier(face_image):

    response = requests.post(
        inference_url, verify=False,
        files={"face_image": encode_image(face_image)},
    ).json()

    label = response["label"]
    inference_time = response["inference_time"]
    pred_score = response["pred_score"]

    return label, inference_time, pred_score
