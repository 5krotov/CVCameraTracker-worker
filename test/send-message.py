import json
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Create a test frame first
from PIL import Image
import numpy as np

img = Image.fromarray(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))
img.save('/tmp/frames/test_frame.jpg')

# Send task
task = {
    "camera_id": "test-camera",
    "frame_path": "/upload/2.jpg",
    "timestamp": datetime.now().isoformat()
}

future = producer.send('video-frames-tasks', value=task)
future.get(timeout=10)
print("Task sent!")

producer.close()
