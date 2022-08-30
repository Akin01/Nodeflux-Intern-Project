from flask import Flask, jsonify, request
from utils import decode_image
from classifier import expression_classifier

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "<p>Face Expression Classification API</p>"


@app.route("/image/process", methods=["POST"])
def expression_classifier():
    if request.method == "POST":
        if request.files:
            file = request.files
            face_image = decode_image(file["face_image"].read())

            label, inference_time, pred_score = expression_classifier(
                face_image)

            return jsonify(
                {
                    "label": label,
                    "inference_time": inference_time,
                    "pred_score": pred_score
                }
            )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7001)
