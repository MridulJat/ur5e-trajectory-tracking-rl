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

# CPU-only torch. Without this the image pulls CUDA and lands around 2.5GB.
# Installed on its own so the PyTorch index is never consulted for other packages
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.15.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "evaluate.py"]
