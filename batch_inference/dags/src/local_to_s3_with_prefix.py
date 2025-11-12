from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator


class LocalFilesystemToS3OperatorWithPrefix(LocalFilesystemToS3Operator):

    template_fields = tuple({"prefix"} | set(LocalFilesystemToS3Operator.template_fields))

    def __init__(self, prefix: str, **kwargs):
        self.prefix = prefix
        super().__init__(**kwargs)
        self.dest_key = self.prefix + "/" + self.dest_key
