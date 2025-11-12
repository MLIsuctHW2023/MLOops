import json

from confluent_kafka import KafkaError, KafkaException  # type: ignore
from src.data_model import PredictionRow

MIN_COMMIT_COUNT = 1
running = True


def consume_loop(consumer, topics):
    try:
        consumer.subscribe(topics)

        msg_count = 0
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    print(
                        "%% %s [%d] reached end at offset %d\n"
                        % (msg.topic(), msg.partition(), msg.offset())
                    )
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                row = PredictionRow(**json.loads(msg.value()))
                print(row.json())
                msg_count += 1
                if msg_count % MIN_COMMIT_COUNT == 0:
                    consumer.commit(asynchronous=False)
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()


if __name__ == "__main__":
    from confluent_kafka import Consumer

    conf = {
        "bootstrap.servers": "localhost:9093",
        "group.id": "only_one_consumer",
        "auto.offset.reset": "smallest",
    }
    consumer = Consumer(**conf)
    consume_loop(consumer, ["mltopic"])
