FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    pillow

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
