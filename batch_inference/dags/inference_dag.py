"""
model_sensor = ждем модель (s3 sensor, (доп балы то mlflow))
metric_sensor = ждем метрик валидации s3 sensor (доп балы то mlflow)
data_sensor = ждем данные s3 sensor
model_choose_op = выбираем лучшую модель по метрикам pythonVirtualEnv (выбираем из mlflow)
download_data_op = грузим данные transfer s3 to local
download_model_op = грузим модель transfer s3 to local (доп балы то mlflow)
inference_model_op = прогоняем модельку по данным pythonVirtualEnv
upload_result_op = загружаем результаты на S3 transfer local to s3
"""