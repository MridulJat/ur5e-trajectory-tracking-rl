FROM python:3.13-slim

# Headless MuJoCo needs an OpenGL implementation
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglew-dev \
        libosmesa6-dev \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch. Without this the image pulls CUDA and lands around 2.5GB
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY . .

CMD ["python", "evaluate.py"]
