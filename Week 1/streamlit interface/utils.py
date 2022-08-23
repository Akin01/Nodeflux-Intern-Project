import requests
from PIL import Image
from io import BytesIO
from base64 import decodebytes, encodebytes
import numpy as np

inference_url = 'http://127.0.0.1:7001/image/process'


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


def image_aligner_api(algined_mock_document, unaligned_filled_document):

    response = requests.post(
        inference_url,
        files={"mock_document": encode_image(algined_mock_document),
               "unaligned_doc": encode_image(unaligned_filled_document),
               },
    ).json()

    image_aligned = decode_image(response["aligned_image"])
    estimated_homography = response["estimated_homography"]

    return image_aligned, estimated_homography
