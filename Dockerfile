FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision \
    diffusers==0.32.1 \
    transformers==4.47.1 \
    accelerate \
    safetensors \
    pillow \
    flask \
    sentencepiece \
    huggingface_hub

COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
