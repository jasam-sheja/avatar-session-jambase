import json
import struct

import cv2
import numpy as np


def serialize_frame(frame: np.ndarray) -> bytes:
    """
    Serialize a frame to bytes.
    """
    # Encode the frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


def deserialize_frame(data: bytes) -> np.ndarray:
    """
    Deserialize bytes to a frame.
    """
    # First, unpack the length of the frame
    if len(data) == 0:
        raise ValueError("Received empty data")

    # Convert the byte data to a numpy array
    frame = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)


def serialize_json(data: dict) -> bytes:
    """
    Serialize a JSON object to bytes.
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    json_data = json.dumps(data)
    return json_data.encode("utf-8")


def deserialize_json(data: bytes) -> dict:
    """
    Deserialize bytes to a JSON object.
    """
    json_data = data.decode("utf-8")
    return json.loads(json_data)
