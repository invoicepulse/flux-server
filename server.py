import os
import io
import base64
import traceback
import torch
from flask import Flask, request, jsonify
from diffusers import FluxPipeline
from huggingface_hub import login

app = Flask(__name__)
pipe = None

def load_model():
    global pipe
    if pipe is None:
        try:
            print("🔐 Logging into HuggingFace...")
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                print("✅ HF login successful")
            else:
                print("⚠️ No HF_TOKEN found in environment")
            
            print("🔄 Loading FLUX 2 Dev...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                use_auth_token=hf_token
            )
            pipe.to("cuda")
            print("✅ FLUX 2 loaded!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print(traceback.format_exc())
            raise
    return pipe

# ... rest of the code stays the same
