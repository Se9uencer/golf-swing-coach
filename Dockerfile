# Public web front end (swingcoach/web/, PLAN.md section 10). The CLI
# pipeline itself has no business needing a container -- this exists only
# to run the FastAPI app on Render.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SWINGCOACH_STORAGE_DIR=/data \
    SWINGCOACH_MODEL_PATH=/app/models/pose_landmarker_full.task

# libgl1/libglib2.0-0: opencv-python-headless still dlopens these despite
# "headless" in the name. libgomp1: mediapipe's native ops use OpenMP.
# ffmpeg itself is NOT installed here -- imageio-ffmpeg (a pip dependency)
# ships its own static binary with libx264, see render/video_writer.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY swingcoach/ swingcoach/
RUN pip install --no-cache-dir ".[web]"

# The pose model isn't committed (9.4MB binary, see models/README.md) --
# baked into the image at build time so there's no cold-start download
# race the first time a request comes in.
RUN mkdir -p models && curl -fsSL -o models/pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn swingcoach.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
