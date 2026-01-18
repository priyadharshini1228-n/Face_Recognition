from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)

REFERENCE_IMAGE_PATH = "static/reference.jpg"

if not os.path.exists(REFERENCE_IMAGE_PATH):
    raise FileNotFoundError("Reference image not found at static/reference.jpg")

# Load reference image with PIL
ref_img = np.array(Image.open(REFERENCE_IMAGE_PATH).convert("RGB"))

@app.route("/")
def health():
    return "OK"

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    try:
        result = DeepFace.verify(
            img,
            ref_img,
            enforce_detection=False
        )
        return jsonify({"verified": result["verified"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
