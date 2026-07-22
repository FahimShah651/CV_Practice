"""
CV Microservice — FastAPI wrapper around the TinyCNN MNIST-digit classifier.

The model served here is the ONNX export produced by 02_ONNX & TensorRT.ipynb
(tiny_classifier.onnx + tiny_classifier.onnx.data). This file is the Week 18 Friday
capstone deliverable: a small, real, testable inference microservice — the same shape
of service you'd put behind the Dockerfile in this folder.

Run locally (outside Docker):
    pip install fastapi uvicorn onnxruntime numpy
    uvicorn app:app --host 0.0.0.0 --port 8000

Then:
    curl http://localhost:8000/health
    curl http://localhost:8000/info

Endpoints:
    GET  /health   -> liveness probe, no model load required
    GET  /info     -> model metadata (shape, providers, class count)
    POST /predict  -> {"image": [[...28x28 floats in [0,1]...]]}
                       -> {"predicted_class": int, "probabilities": [...], "latency_ms": float}
"""
import os
import time
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_classifier.onnx")

app = FastAPI(
    title="CV Microservice — TinyCNN MNIST Classifier",
    description="Week 18 Capstone: dockerized FastAPI service serving an ONNX-exported digit classifier.",
    version="1.0.0",
)

# Lazily-initialised ONNX Runtime session — loaded once, reused across requests.
_session = None
_input_name = None
_output_name = None


def get_session() -> ort.InferenceSession:
    global _session, _input_name, _output_name
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                f"Run 02_ONNX & TensorRT.ipynb first to produce tiny_classifier.onnx."
            )
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name
    return _session


class PredictRequest(BaseModel):
    image: List[List[float]] = Field(
        ..., description="28x28 grayscale MNIST-style digit image, pixel values in [0, 1]"
    )


class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]
    latency_ms: float


@app.get("/health")
def health():
    """Liveness probe — does not touch the model, so it stays fast even under load."""
    return {"status": "ok"}


@app.get("/info")
def info():
    """Readiness probe + model metadata — touches the model, so a broken model file surfaces here."""
    sess = get_session()
    return {
        "model": "TinyCNN (2x Conv-BN-ReLU-Pool + 2x Linear)",
        "task": "MNIST digit classification",
        "format": "ONNX",
        "input_shape": [1, 1, 28, 28],
        "num_classes": 10,
        "onnx_runtime_providers": sess.get_providers(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run one inference. Expects a 28x28 array; returns the predicted digit + full softmax."""
    arr = np.array(req.image, dtype=np.float32)
    if arr.shape != (28, 28):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a 28x28 image, got shape {list(arr.shape)}",
        )

    sess = get_session()
    x = arr.reshape(1, 1, 28, 28)

    t0 = time.perf_counter()
    logits = sess.run([_output_name], {_input_name: x})[0][0]
    latency_ms = (time.perf_counter() - t0) * 1000

    exp = np.exp(logits - logits.max())
    probs = (exp / exp.sum()).tolist()

    return PredictResponse(
        predicted_class=int(np.argmax(logits)),
        probabilities=probs,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
