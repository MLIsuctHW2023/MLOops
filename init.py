import os
import time
import logging
import json
from typing import Optional
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "codegen")
MODELS_BUCKET_NAME = os.getenv("MODELS_BUCKET_NAME", "triton-models")
MAX_RETRIES = 12
RETRY_DELAY = 8

def create_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        verify=False,
    )

def wait_for_minio() -> boto3.client:
    logger.info("Waiting for MinIO to become available...")
    for attempt in range(MAX_RETRIES):
        try:
            s3_client = create_s3_client()
            s3_client.list_buckets()
            logger.info("MinIO is available")
            return s3_client
        except (EndpointConnectionError, ClientError) as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("MinIO not ready yet (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                time.sleep(RETRY_DELAY)
            else:
                logger.error("MinIO failed to become available after %d attempts", MAX_RETRIES)
                raise

def initialize_bucket(s3_client: boto3.client, bucket_name: str) -> bool:
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info("Bucket %s already exists", bucket_name)
        return False
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                logger.info("Bucket %s created successfully", bucket_name)
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                        }
                    ]
                }
                s3_client.put_bucket_policy(
                    Bucket=bucket_name,
                    Policy=json.dumps(bucket_policy)
                )
                logger.info("Bucket policy set for %s", bucket_name)
                return True
            except ClientError as create_error:
                logger.error("Failed to create bucket %s: %s", bucket_name, create_error)
                raise
        else:
            logger.error("Error checking bucket %s: %s", bucket_name, e)
            raise

def upload_all_files(s3_client: boto3.client) -> None:
    try:
        files_to_upload = [
            ("codegen/1/model.onnx", "model.onnx"),
            ("codegen/config.pbtxt", "config.pbtxt"),
            ("codegen_dali_ensemble/1/model.config", "dali.config"),
            ("text_preprocessor/1/model.config", "preprocessor.config"),
            ("text_postprocessor/1/model.config", "postprocessor.config")
        ]
        
        for s3_key, local_file in files_to_upload:
            if os.path.exists(local_file):
                with open(local_file, 'rb') as file:
                    s3_client.put_object(
                        Bucket=MODELS_BUCKET_NAME,
                        Key=s3_key,
                        Body=file,
                        ContentType='application/octet-stream' if s3_key.endswith('.onnx') else 'text/plain'
                    )
                logger.info("Uploaded %s to S3 as %s", local_file, s3_key)
            else:
                logger.warning("File %s not found, skipping", local_file)
                
    except Exception as e:
        logger.error("Failed to upload files to S3: %s", e)
        raise

def main() -> int:
    try:
        logger.info("Starting storage and model initialization...")
        s3_client = wait_for_minio()
        initialize_bucket(s3_client, BUCKET_NAME)
        initialize_bucket(s3_client, MODELS_BUCKET_NAME)
        upload_all_files(s3_client)
        logger.info("Storage and model initialization completed successfully")
        return 0
    except Exception as e:
        logger.error("Initialization failed: %s", e)
        return 1

if __name__ == "__main__":
    exit(main())