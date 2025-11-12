from datetime import datetime

from airflow import DAG
from airflow.utils import trigger_rule
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator

from src.local_to_s3_with_prefix import LocalFilesystemToS3OperatorWithPrefix


def create_random_data(path_to_save: str) -> str:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path_to_file = Path(path_to_save) / "data.csv"
    path_to_file.parent.mkdir(exist_ok=True, parents=True)
    days = np.random.randint(0, 10, size=100)
    ages = np.random.randint(0, 10, size=100)
    df = pd.DataFrame({"days": days, "ages": ages})
    df.to_csv(path_to_file, index=False)
    return str(path_to_file)


with DAG(
    "s3_create_bucket",
    schedule="@once",
    default_args={
        "owner": "imdxd",
    },
    start_date=datetime(2025, 9, 21, 0, 0, 0),
    max_active_runs=1,
    max_active_tasks=3,
    end_date=None,
) as dag:

    create_bucket_task = S3CreateBucketOperator(
        task_id="create_bucket",
        bucket_name="mlbucket",
        aws_conn_id="minio_connection",
        dag=dag,
    )

    create_data = PythonOperator(
        task_id="create_data",
        dag=dag,
        python_callable=create_random_data,
        op_kwargs={
            "path_to_save": "/data/raw_data/{{ ds }}"
        }
    )

    upload_file_to_s3 = LocalFilesystemToS3OperatorWithPrefix.partial(
        task_id="upload_files",
        aws_conn_id="minio_connection",
        dest_bucket="mlbucket",
        replace=False,
        filename=create_data.output,
        prefix="raw_data/{{ ds }}"
    ).expand_kwargs(
        [
            {"dest_key": "data1.csv"},
            {"dest_key": "data2.csv"},
            {"dest_key": "data3.csv"},
        ]
    )

    [create_bucket_task, create_data] >> upload_file_to_s3
