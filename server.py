import os
import io
import base64
import torch
from flask import Flask, request, jsonify
from diffusers import FluxPipeline

app = Flask(__name__)
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("🔄 Loading FLUX 2 Dev...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("✅ FLUX 2 loaded!")
    return pipe

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    try:
        data = request.json
        scenes = data.get("scenes", [])
        
        print(f"📦 Generating {len(scenes)} images...")
        
        model = load_model()  # Loads on first request, not startup
        results = []
        
        for scene in scenes:
            prompt = scene.get("prompt")
            scene_id = scene.get("scene_id", 0)
            width = scene.get("width", 720)
            height = scene.get("height", 1280)
            
            print(f"🎨 Scene {scene_id}: {prompt[:50]}...")
            
            image = model(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            results.append({
                "scene_id": scene_id,
                "image_b64": image_b64,
                "status": "success"
            })
            
            print(f"✅ Scene {scene_id} done!")
        
        return jsonify({
            "results": results,
            "status": "complete"
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    # Don't preload - Flask starts immediately
    app.run(host="0.0.0.0", port=8000)
