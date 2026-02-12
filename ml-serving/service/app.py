"""FastAPI service for model inference with Triton."""
from typing import Dict, Any, List, cast
import numpy as np
import requests  # type: ignore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Inference Service", version="1.0.0")


class InferenceRequest(BaseModel):
    """Request model for inference."""
    model_config = ConfigDict(protected_namespaces=())
    
    input: List[List[List[List[float]]]] = Field(
        ...,
        description="Input tensor as 4D list: [batch][channel][height][width]"
    )


class InferenceResponse(BaseModel):
    """Response model for inference."""
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[List[float]] = Field(
        ...,
        description="Output predictions for each batch item"
    )
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")


class HealthResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Service status")
    triton: str = Field(..., description="Triton server status")


class TritonClient:
    """Client for communicating with Triton Inference Server."""
    
    def __init__(self, url: str = None) -> None:
        """Initialize Triton client.
        
        Args:
            url: Triton server URL. If not provided, read from TRITON_URL environment variable.
        """
        self.url = url or os.getenv("TRITON_URL", "triton:8000")  
        self.model_name = "simple-net"
        self.model_version = "1"

    def is_ready(self) -> bool:
        """Check if Triton server is ready.
        
        Returns:
            True if Triton is ready, False otherwise
        """
        try:
            response = requests.get(
                f"http://{self.url}/v2/health/ready",  
                timeout=5.0
            )
            status_code: int = cast(int, response.status_code)
            return status_code == 200
        except requests.exceptions.RequestException as e:
            logger.warning(f"Triton connection failed: {e}")
            return False

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Perform inference using Triton.
        
        Args:
            input_tensor: Input tensor as numpy array
            
        Returns:
            Output predictions as numpy array
            
        Raises:
            HTTPException: If inference fails
        """
        if not self.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Triton server is not ready"
            )

        # Prepare request data
        inputs = [{
            "name": "input",
            "shape": input_tensor.shape,
            "datatype": "FP32",
            "data": input_tensor.tolist()
        }]

        outputs = [{"name": "output"}]

        request_data = {
            "inputs": inputs,
            "outputs": outputs
        }

        try:
            response = requests.post(
                f"http://{self.url}/v2/models/{self.model_name}/versions/{self.model_version}/infer",
                json=request_data,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            output_data = result["outputs"][0]["data"]
            output_shape = result["outputs"][0]["shape"]
            
            return np.array(output_data).reshape(output_shape)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            ) from e


# Initialize Triton client
triton_client = TritonClient()


def wait_for_triton(max_retries: int = 30, delay: float = 2.0) -> None:
    """Wait for Triton server to become ready."""
    for i in range(max_retries):
        if triton_client.is_ready():
            logger.info("Triton server is ready!")
            return
        else:
            logger.warning(f"Waiting for Triton... ({i+1}/{max_retries})")
            time.sleep(delay)
    raise Exception("Triton server did not become ready in time.")


@app.on_event("startup")
async def startup_event():
    """Wait for Triton to be ready on startup."""
    logger.info("Starting up...")
    wait_for_triton()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        Health status of the service and Triton
    """
    try:
        triton_status = "connected" if triton_client.is_ready() else "disconnected"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        triton_status = "error"
    
    return HealthResponse(status="healthy", triton=triton_status)


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    """Perform model inference.
    
    Args:
        request: Inference request with input tensor
        
    Returns:
        Inference response with predictions
    """
    try:
        # Convert input to numpy array
        input_array = np.array(request.input, dtype=np.float32)
        logger.info(f"Input shape: {input_array.shape}")
        
        # Validate input shape
        if input_array.ndim != 4:
            raise HTTPException(
                status_code=400,
                detail="Input must be 4D tensor: [batch][channel][height][width]"
            )
        
        if input_array.shape[1:] != (1, 28, 28):
            raise HTTPException(
                status_code=400,
                detail=f"Input shape must be [batch, 1, 28, 28], got {input_array.shape}"
            )

        # Perform inference
        predictions = triton_client.infer(input_array)
        
        return InferenceResponse(
            predictions=predictions.tolist(),
            model_name=triton_client.model_name,
            model_version=triton_client.model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        ) from e


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint.
    
    Returns:
        Welcome message
    """
    return {"message": "ML Inference Service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)