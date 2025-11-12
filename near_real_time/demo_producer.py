import random
import socket
import time
import uuid

from confluent_kafka import Producer  # type: ignore
from src.data_model import PredictionRow

conf = {"bootstrap.servers": "localhost:9093", "client.id": socket.gethostname()}
possible_type_values = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
producer = Producer(**conf)

while True:
    choosen_type = random.choice(possible_type_values)
    choosen_amount = random.randint(0, 1_000_000)
    oldbalanceOrg = random.randint(0, 1_000_000)
    newbalanceOrig = random.randint(0, 1_000_000)
    oldbalanceDest = random.randint(0, 1_000_000)
    newbalanceDest = random.randint(0, 1_000_000)
    pred_row = PredictionRow(
        type=choosen_type,
        amount=choosen_amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
    )
    producer.produce(
        topic="mltopic",
        key=str(uuid.uuid4()),
        value=pred_row.json(),
    )
    time.sleep(5)
