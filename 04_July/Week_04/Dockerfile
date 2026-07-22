# Week 18 Capstone — Dockerized FastAPI CV Microservice
#
# NOTE: this image was NOT built or run in this environment — the dev sandbox this
# notebook ran in has no `docker` binary available (verified with `docker --version`
# in 05_CAPSTONE...ipynb, which failed with "command not found"). The Dockerfile below
# is written to the same standard as everything else in this repo and is ready to build
# on any machine with Docker installed; see the capstone notebook for the honest
# in-notebook substitute (fastapi.testclient.TestClient) used to prove app.py works.
#
# Build (on a machine with Docker):
#   docker build -t cv-microservice .
# Run:
#   docker run -p 8000:8000 cv-microservice
# Then:
#   curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the ONNX model this service serves (produced by
# 02_ONNX & TensorRT.ipynb — both the small graph file and its external weights file
# are required for onnxruntime to load the model)
COPY app.py .
COPY tiny_classifier.onnx .
COPY tiny_classifier.onnx.data .

EXPOSE 8000

# Basic container-level health check, hits the liveness probe defined in app.py
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
