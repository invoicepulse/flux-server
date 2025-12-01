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
        pipe.enable_model_cpu_offload()  # Memory optimization
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
        
        print(f"📦 Generating {len(scenes)} images in parallel batches...")
        
        model = load_model()
        results = []
        
        # H100 can do ~6-8 images in parallel (32GB VRAM)
        BATCH_SIZE = 6
        
        for i in range(0, len(scenes), BATCH_SIZE):
            batch = scenes[i:i+BATCH_SIZE]
            prompts = [s.get("prompt") for s in batch]
            scene_ids = [s.get("scene_id", idx) for idx, s in enumerate(batch)]
            
            # Get dimensions from first scene (all same)
            width = batch[0].get("width", 1080)
            height = batch[0].get("height", 1920)
            
            print(f"🎨 Batch {i//BATCH_SIZE + 1}: {len(prompts)} images @ {width}x{height}...")
            
            # PARALLEL generation - all images at once
            images = model(
                prompt=prompts,  # List of prompts
                width=width,
                height=height,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images
            
            # Convert all to base64
            for idx, image in enumerate(images):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)  # Higher quality
                image_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                results.append({
                    "scene_id": scene_ids[idx],
                    "image_b64": image_b64,
                    "status": "success"
                })
            
            print(f"✅ Batch {i//BATCH_SIZE + 1} done!")
        
        return jsonify({
            "results": results,
            "status": "complete"
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
