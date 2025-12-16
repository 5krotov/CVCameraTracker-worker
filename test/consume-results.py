from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'detection-results',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest'
)

print("Waiting for detection results...")
for message in consumer:
    result = message.value
    print(f"Camera: {result['camera_id']}")
    print(f"People detected: {len(result['objects'])}")
    print(f"Coordinates: {result['coordinates']}")
    break

consumer.close()
