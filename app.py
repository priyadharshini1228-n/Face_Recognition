from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
from deepface import DeepFace


app = Flask(__name__)

# Load reference image ONCE at startup
REFERENCE_IMAGE_PATH = "static/reference.jpg"

if not os.path.exists(REFERENCE_IMAGE_PATH):
    raise FileNotFoundError("Reference image not found at static/reference.jpg")

ref_img = cv2.imread(REFERENCE_IMAGE_PATH)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Face Verification API running"}), 200

@app.route("/ui", methods=["GET"])
def ui():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify_face():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.verify(
            img,
            ref_img,
            enforce_detection=False
        )
        return jsonify({
            "verified": result["verified"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
