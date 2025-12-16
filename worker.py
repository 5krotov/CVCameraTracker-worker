"""
DETR-based people detection worker for video frame processing.

Consumes FrameTask messages from Kafka, detects people using DETR,
and publishes DetectionResult messages back to Kafka.
"""

import json
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms as T
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FrameTask:
    """Represents a frame task from the backend service."""
    camera_id: str
    frame_path: str
    timestamp: str

    @classmethod
    def from_json(cls, data: dict) -> 'FrameTask':
        return cls(**data)


@dataclass
class DetectionResult:
    """Represents detection results to be sent back to backend."""
    camera_id: str
    frame_path: str
    timestamp: str
    objects: list[str]  # List of detected object types (e.g., ['person', 'person'])
    coordinates: list[list[float]]  # Bounding boxes [x_min, y_min, x_max, y_max]

    def to_json(self) -> dict:
        return asdict(self)


class DETRDetector:
    """Wrapper for DETR object detection model."""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize DETR detector.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            confidence_threshold: Minimum confidence for detections
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Loading DETR model: {model_name} on device: {self.device}")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # COCO class names - focus on person class (index 0)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

        logger.info("DETR detector initialized successfully")

    def detect(self, image_path: str) -> Optional[tuple[int, list, list]]:
        """
        Detect objects in image, filtering for people only.

        Args:
            image_path: Path to image file

        Returns:
            DetectionResult with detected people and coordinates, or None if error
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])  # [height, width]
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )

            detections = results[0]
            people_objects = []
            people_coordinates = []

            logger.info(
                f"Detected {len(detections['scores'])} objects: "
                f"{[self.class_names[i.item()] for i in detections['labels']]}",
            )

            if len(detections['scores']) > 0:
                for score, label, box in zip(
                    detections['scores'],
                    detections['labels'],
                    detections['boxes']
                ):
                    label_idx = label.item() - 1
                    class_name = self.class_names[label_idx]

                    # Filter for person class only (index 0)
                    if class_name == 'person':
                        confidence = score.item()
                        # Normalize box coordinates [x_min, y_min, x_max, y_max]
                        box_normalized = box.cpu().numpy().tolist()
                        people_objects.append('person')
                        people_coordinates.append(box_normalized)

                        logger.debug(
                            f"Detected person: confidence={confidence:.3f}, "
                            f"box={box_normalized}"
                        )

            logger.info(
                f"Image {image_path}: detected {len(people_objects)} people"
            )
            return len(people_objects), people_objects, people_coordinates

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            return None

    def __call__(self, image_path: str) -> Optional[tuple[int, list[str], list[list[float]]]]:
        """Make detector callable."""
        return self.detect(image_path)


class KafkaFrameWorker:
    """Worker that consumes frame tasks and produces detection results."""

    def __init__(
        self,
        kafka_brokers: str,
        input_topic: str = "video-frames-tasks",
        output_topic: str = "detection-results",
        group_id: str = "detr-worker-group",
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.7
    ):
        """
        Initialize Kafka worker.

        Args:
            kafka_brokers: Comma-separated broker addresses (e.g., "localhost:9092")
            input_topic: Topic to consume frame tasks from
            output_topic: Topic to produce detection results to
            group_id: Kafka consumer group ID
            model_name: DETR model identifier
            confidence_threshold: Detection confidence threshold
        """
        brokers = kafka_brokers.split(',') if isinstance(kafka_brokers, str) else kafka_brokers

        logger.info(f"Initializing Kafka worker with brokers: {brokers}")

        # Initialize consumer
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=brokers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            session_timeout_ms=30000,
            request_timeout_ms=40000
        )

        # Initialize producer
        self.producer = KafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            request_timeout_ms=40000
        )

        self.input_topic = input_topic
        self.output_topic = output_topic

        # Initialize detector
        self.detector = DETRDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

        self.running = False
        logger.info("Kafka worker initialized successfully")

    def process_frame_task(self, task_data: dict) -> bool:
        """
        Process a single frame task.

        Args:
            task_data: Frame task data dictionary

        Returns:
            True if processing successful, False otherwise
        """
        try:
            task = FrameTask.from_json(task_data)

            # Validate frame path exists
            if not os.path.exists(task.frame_path):
                logger.error(f"Frame file not found: {task.frame_path}")
                return False

            logger.info(f"Processing frame task: camera={task.camera_id}, frame={task.frame_path}")

            # Run detection
            detection_result = self.detector.detect(task.frame_path)

            if detection_result is None:
                logger.error(f"Detection failed for frame: {task.frame_path}")
                return False

            num_people, people_objects, people_coordinates = detection_result

            # Create result message
            result = DetectionResult(
                camera_id=task.camera_id,
                frame_path=task.frame_path,
                timestamp=task.timestamp,
                objects=people_objects,
                coordinates=people_coordinates
            )

            # Send to Kafka
            self._send_result(result)

            logger.info(
                f"Frame processed successfully: camera={task.camera_id}, "
                f"people_detected={num_people}"
            )
            return True

        except Exception as e:
            logger.error(f"Error processing frame task: {e}", exc_info=True)
            return False

    def _send_result(self, result: DetectionResult) -> None:
        """Send detection result to Kafka."""
        try:
            future = self.producer.send(
                self.output_topic,
                value=result.to_json(),
                key=result.camera_id.encode('utf-8')
            )

            # Wait for send to complete with timeout
            future.get(timeout=10)
            logger.debug(f"Result sent to Kafka: {result.camera_id}")

        except KafkaError as e:
            logger.error(f"Failed to send result to Kafka: {e}", exc_info=True)
            raise

    def run(self) -> None:
        """Start consuming and processing frame tasks."""
        self.running = True
        logger.info("Starting Kafka frame worker")

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            for message in self.consumer:
                if not self.running:
                    break

                logger.debug(f"Received message from topic {self.input_topic}")

                try:
                    self.process_frame_task(message.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Stopping Kafka frame worker")
        self.running = False
        try:
            self.consumer.close()
            self.producer.close()
            logger.info("Kafka connections closed")
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {e}")


def main():
    """Main entry point."""
    # Configuration from environment variables
    kafka_brokers = os.getenv('KAFKA_BROKERS', 'localhost:9092')
    input_topic = os.getenv('KAFKA_FRAME_TASK_TOPIC', 'video-frames-tasks')
    output_topic = os.getenv('KAFKA_DETECTION_RESULT_TOPIC', 'detection-results')
    group_id = os.getenv('KAFKA_WORKER_GROUP', 'detr-worker-group')
    model_name = os.getenv('DETR_MODEL', 'facebook/detr-resnet-50')
    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

    logger.info("Starting DETR Detection Worker")
    logger.info(f"Kafka brokers: {kafka_brokers}")
    logger.info(f"Input topic: {input_topic}")
    logger.info(f"Output topic: {output_topic}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Confidence threshold: {confidence_threshold}")

    worker = KafkaFrameWorker(
        kafka_brokers=kafka_brokers,
        input_topic=input_topic,
        output_topic=output_topic,
        group_id=group_id,
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )

    worker.run()


if __name__ == '__main__':
    main()
