import os
import io
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        data = request.json
        prompt = data.get("prompt")
        scene_id = data.get("scene_id", 0)
        
        # Placeholder - will implement Flux later
        return jsonify({
            "scene_id": scene_id,
            "image_b64": "placeholder",
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
