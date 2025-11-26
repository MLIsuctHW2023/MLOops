"""
data_sensor = ждем на s3 сгенерированные данные s3 sensor
get_data_op = загружаем данные с s3 transfer s3 to local
split_data_op = делим на трейн и валидацию pythonVirtualEnv
train_op = Тренируем модель pythonVirtualEnv
model_upload_op = загружаем модель на S3 (доп балы то mlflow) transfer.LocalToS3
evaluate_data_op = Прогоняем валидацию pythonVirtualEnv
metric_upload_op = Грузим метрики по валидации на S3 (доп балы то mlflow) transfer.LocalToS3
"""