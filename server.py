import os
import io
import base64
import traceback
import torch
from flask import Flask, request, jsonify
from diffusers import FluxPipeline

app = Flask(__name__)
pipe = None

def load_model():
    global pipe
    if pipe is None:
        try:
            print("🔄 Loading FLUX 2 Dev...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16
            )
            pipe.to("cuda")
            print("✅ FLUX 2 loaded!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print(traceback.format_exc())
            raise
    return pipe

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    try:
        print("📥 Received batch request")
        data = request.json
        print(f"📦 Request data: {data}")
        
        scenes = data.get("scenes", [])
        print(f"🎬 Processing {len(scenes)} scenes")
        
        model = load_model()
        results = []
        
        BATCH_SIZE = 6
        
        for i in range(0, len(scenes), BATCH_SIZE):
            batch = scenes[i:i+BATCH_SIZE]
            prompts = [s.get("prompt") for s in batch]
            scene_ids = [s.get("scene_id", idx) for idx, s in enumerate(batch)]
            
            width = batch[0].get("width", 1080)
            height = batch[0].get("height", 1920)
            
            print(f"🎨 Batch {i//BATCH_SIZE + 1}: {len(prompts)} images @ {width}x{height}")
            print(f"📝 Prompts: {prompts}")
            
            try:
                images = model(
                    prompt=prompts,
                    width=width,
                    height=height,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                ).images
                
                print(f"✅ Generated {len(images)} images")
                
                for idx, image in enumerate(images):
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=95)
                    image_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    results.append({
                        "scene_id": scene_ids[idx],
                        "image_b64": image_b64,
                        "status": "success"
                    })
                
                print(f"✅ Batch {i//BATCH_SIZE + 1} completed")
                
            except Exception as batch_error:
                print(f"❌ Batch generation error: {batch_error}")
                print(traceback.format_exc())
                raise
        
        print(f"🎉 All {len(results)} images generated successfully")
        return jsonify({
            "results": results,
            "status": "complete"
        })
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ ERROR: {error_msg}")
        print(f"❌ TRACEBACK: {error_trace}")
        return jsonify({
            "error": error_msg,
            "traceback": error_trace,
            "status": "failed"
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
