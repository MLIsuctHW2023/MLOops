import os
import json
import logging
import time
from typing import Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import boto3
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRITON_HOST = os.getenv("TRITON_HOST", "triton-server")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MODEL_NAME = os.getenv("MODEL_NAME", "codegen")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "text-summaries")

class codegenRequest(BaseModel):
    text: str

class codegenResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TRITON_CLIENT, S3_CLIENT
    
    S3_CLIENT = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        verify=False,
    )
    
    TRITON_CLIENT = grpcclient.InferenceServerClient(url=f"{TRITON_HOST}:{TRITON_PORT}")
    
    for attempt in range(30):
        try:
            if TRITON_CLIENT.is_model_ready(model_name=MODEL_NAME):
                logger.info("Model %s ready", MODEL_NAME)
                break
            time.sleep(2)
        except Exception:
            time.sleep(2)
    
    yield
    
    if TRITON_CLIENT:
        TRITON_CLIENT.close()

app = FastAPI(lifespan=lifespan)

def process_text_with_triton(text: str) -> str:
    input_tokens = [ord(char) for char in text[:512]]
    if len(input_tokens) < 512:
        input_tokens.extend([0] * (512 - len(input_tokens)))
    
    attention_mask = [1 if token != 0 else 0 for token in input_tokens]
    
    inference_inputs = [
        grpcclient.InferInput("input_ids", [1, 512], "INT32"),
        grpcclient.InferInput("attention_mask", [1, 512], "INT32")
    ]
    
    inference_inputs[0].set_data_from_numpy(np.array([input_tokens], dtype=np.int32))
    inference_inputs[1].set_data_from_numpy(np.array([attention_mask], dtype=np.int32))
    
    result = TRITON_CLIENT.infer(MODEL_NAME, inference_inputs, outputs=[grpcclient.InferRequestedOutput("output_0")])
    output_array = result.as_numpy("output_0")
    
    summary_tokens = output_array[0].tolist()
    return "".join(chr(int(token)) for token in summary_tokens if 0 < token < 1114111).strip()

@app.post("/codegen", response_model=codegenResponse)
async def codegen_text(request: codegenRequest):
    original_text = request.text.strip()
    if not original_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    summary = process_text_with_triton(original_text)
    
    return codegenResponse(
        summary=summary,
        original_length=len(original_text),
        summary_length=len(summary)
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)