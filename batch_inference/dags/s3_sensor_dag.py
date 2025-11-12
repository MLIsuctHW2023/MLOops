from datetime import datetime
from typing import Any, List, Dict

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor


def check_size(files: List[Dict[str, Any]]) -> bool:
    return all([f["Size"] > 1 for f in files])


def upload_file(aws_conn_id: str, bucket_name: str, key: str, filepath: str) -> str:
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    filepath = s3_hook.download_file(key, bucket_name, filepath, preserve_file_name=True)
    return filepath


with DAG(
    "s3_check_files",
    schedule="0 12 * * *",
    default_args={
        "owner": "imdxd",
    },
    max_active_tasks=1,
    max_active_runs=1,
    start_date=datetime(2025, 10, 27, 0, 0, 0),
    end_date=None,
) as dag:

    sensor_task = S3KeySensor(
        task_id="wait_for_data",
        dag=dag,
        aws_conn_id="minio_connection",
        bucket_name="mlbucket",
        bucket_key="raw_data/{{ ds }}/data.csv",
        check_fn=check_size,
        poke_interval=10,
        timeout=60,
        soft_fail=True
    )

    upload_file_op = PythonVirtualenvOperator(
        task_id="upload_s3_file",
        dag=dag,
        python_callable=upload_file,
        op_kwargs={
            "aws_conn_id": "minio_connection",
            "bucket_name": "mlbucket",
            "key": "raw_data/{{ ds }}/data.csv",
            "filepath": "/data/loaded_data/{{ds}}"
        },
        requirements=["boto3==1.33.13"],
        retries=4
    )

    sensor_task >> upload_file_op

