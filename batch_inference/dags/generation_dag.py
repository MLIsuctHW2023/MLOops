"""
MNIST/FashionMNIST (любой датасет torchvision/torchaudio/transformers способный запустится на CPU)
download_data_op = from internet
create_bucket_op = создать бакет
upload_train_data_op = to S3
upload_test_data_op = to S3
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator


def get_data(train_save_path: str, test_save_path: str):
    from pathlib import Path

    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    train_save_p = Path(train_save_path)
    test_save_p = Path(test_save_path)
    train_save_p.parent.mkdir(exist_ok=True, parents=True)
    test_save_p.parent.mkdir(exist_ok=True, parents=True)
    dataset = load_iris()
    df = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    df["target"] = dataset["target"]
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    train_df.to_parquet(train_save_p)
    test_df.to_parquet(test_save_p)


with DAG(
    "generation_data_dag",
    schedule="0 3 * * *",
    default_args={
        "owner": "imdxd",
    },
    start_date=datetime(2025, 11, 11, 0, 0, 0),
    max_active_runs=1,
    max_active_tasks=1,
    end_date=None,
) as dag:

    get_data_op = PythonVirtualenvOperator(
        dag=dag,
        task_id="generate_data_op",
        python_callable=get_data,
        op_kwargs={
            "train_save_path": "/data/uploaded_data/{{ ds }}/train_data.parquet",
            "test_save_path": "/data/uploaded_data/{{ ds }}/test_data.parquet"
        },
        requirements=["scikit-learn==1.0.2", "pandas==1.3.5"],
    )

    create_bucket_task = S3CreateBucketOperator(
        task_id="create_bucket",
        bucket_name="mldata",
        aws_conn_id="minio_connection",
        dag=dag,
    )

    train_data_upload_op = LocalFilesystemToS3Operator(
        dag=dag,
        task_id="upload_train_data_op",
        aws_conn_id="minio_connection",
        dest_key="uploaded_data/{{ ds }}/train_data.parquet",
        dest_bucket="mldata",
        filename="/data/uploaded_data/{{ ds }}/train_data.parquet"
    )

    test_data_upload_op = LocalFilesystemToS3Operator(
        dag=dag,
        task_id="upload_test_data_op",
        aws_conn_id="minio_connection",
        dest_key="uploaded_data/{{ ds }}/test_data.parquet",
        dest_bucket="mldata",
        filename="/data/uploaded_data/{{ ds }}/test_data.parquet"
    )

    get_data_op >> create_bucket_task >> train_data_upload_op >> test_data_upload_op
