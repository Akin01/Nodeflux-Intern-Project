from flask import Flask, jsonify, request
from utils import encode_image, decode_image
from images_aligner import process_align
from PIL import Image
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "<p>This is an API to process image as aligned document.</p>"


@app.route("/image/process", methods=["POST"])
def align_document():
    if request.method == "POST":
        if request.files:
            file = request.files
            mock_document = decode_image(file["mock_document"].read())
            unaligned_filled_doc = decode_image(file["unaligned_doc"].read())

            aligned_document, estimated_homography = process_align(
                unaligned_filled_doc, mock_document)

            return jsonify(
                {
                    "aligned_image": encode_image(
                        Image.fromarray(np.uint8(aligned_document))),
                    "estimated_homography": estimated_homography.tolist()
                }
            )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7001)
