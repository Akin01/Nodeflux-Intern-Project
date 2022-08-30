from io import BytesIO
from base64 import encodebytes, decodebytes
from PIL import Image
import numpy as np


def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode(
        'ascii')  # encode as base64
    return encoded_img


def decode_image(image_bytes):
    image_bytes = decodebytes(image_bytes)
    image_bytes = BytesIO(image_bytes)
    image_bytes = Image.open(image_bytes)
    return np.array(image_bytes)
