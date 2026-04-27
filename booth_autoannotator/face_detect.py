"""
Face detection with landmarks and confidence scoring.

Uses OpenCV DNN face detector for robust face presence detection.
Extracts landmarks for downstream head pose and blink analysis.
"""

# from retinaface import RetinaFace
import os
import pickle

from insightface.app import FaceAnalysis
from insightface.app.common import Face
import tensorflow as tf
from pathlib import Path
# from mtcnn import MTCNN
import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
__dir__ = Path(__file__).parent


@dataclass
class FaceDetection:
    """Face detection result at a specific timestamp."""

    # bbox <class 'numpy.ndarray'>
    # kps <class 'numpy.ndarray'>
    # det_score <class 'numpy.float32'>
    # landmark_3d_68 <class 'numpy.ndarray'>
    # pose <class 'numpy.ndarray'>
    # landmark_2d_106 <class 'numpy.ndarray'>
    # gender <class 'numpy.int64'>
    # age <class 'int'>
    # embedding <class 'numpy.ndarray'>

    timestamp: float
    bbox: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    det_score: Optional[float] = None
    landmark_3d_68: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    landmark_2d_106: Optional[np.ndarray] = None
    gender: Optional[int] = None
    age: Optional[int] = None
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_face(cls, timestamp: float, face: Face) -> "FaceDetection":
        """Create FaceDetection from insightface Face object."""
        if face is None:
            return cls(timestamp=timestamp)
        return cls(
            timestamp=timestamp,
            bbox=face.bbox,
            kps=face.kps,
            det_score=face.det_score,
            landmark_3d_68=face.landmark_3d_68,
            pose=face.pose,
            landmark_2d_106=face.landmark_2d_106,
            gender=int(face.gender),
            age=face.age,
            embedding=face.embedding,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "kps": self.kps.tolist() if self.kps is not None else None,
            "det_score": self.det_score,
            "landmark_3d_68": (
                self.landmark_3d_68.tolist()
                if self.landmark_3d_68 is not None
                else None
            ),
            "pose": self.pose.tolist() if self.pose is not None else None,
            "landmark_2d_106": (
                self.landmark_2d_106.tolist()
                if self.landmark_2d_106 is not None
                else None
            ),
            "gender": self.gender,
            "age": self.age,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
        }

    @property
    def detected(self) -> bool:
        """Whether a face was detected."""
        return self.bbox is not None

    # def __getattr__(self, name):
    #     # access face attributes directly from FaceDetection for convenience
    #     if self.face is not None and hasattr(self.face, name):
    #         return getattr(self.face, name)
    #     raise AttributeError(f"'FaceDetection' object has no attribute '{name}'")


# class MTCNNFaceDetector:
#     """Face detector using the `mtcnn` package (TensorFlow backend)."""

#     def __init__(self, confidence_threshold: float = 0.5):
#         """
#         Initialize MTCNN detector.

#         Args:
#             confidence_threshold: Minimum confidence for valid face detection.
#         """
#         self.confidence_threshold = confidence_threshold
#         self.detector = MTCNN(
#             device="GPU:0"
#         )  # Load on CPU by default; can be changed to 'cuda' if supported

#     def detect_faces(
#         self, frame: np.ndarray
#     ) -> List[Tuple[float, Tuple[int, int, int, int]]]:
#         """
#         Detect faces using MTCNN.

#         Args:
#             frame: Input frame (BGR).

#         Returns:
#             List of (confidence, bbox) tuples.
#         """

#         h, w = frame.shape[:2]
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         try:
#             detections = self.detector.detect_faces(
#                 rgb, limit_boundaries_landmarks=True
#             )
#         except Exception as e:
#             logger.warning(f"MTCNN inference failed: {e}")
#             return []

#         faces = []
#         for det in detections:
#             confidence = float(det.get("confidence", 0.0))
#             if confidence < self.confidence_threshold:
#                 continue

#             box = det.get("box", None)
#             if box is None or len(box) != 4:
#                 continue

#             x, y, bw, bh = [int(v) for v in box]
#             if x <= 0 or y <= 0 or bw <= 0 or bh <= 0 or x + bw > w or y + bh > h:
#                 continue

#             faces.append((confidence, box, det.get("keypoints", None)))

#         return faces

#     def detect_face(
#         self, frame: np.ndarray
#     ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
#         """
#         Detect if a face is present in the frame.

#         Args:
#             frame: Input frame (BGR).

#         Returns:
#             Tuple of (detected, confidence, bbox, landmarks).
#         """
#         faces = self.detect_faces(frame)
#         if not faces:
#             return False, 0.0, None, None
#         confidence, bbox, landmarks = max(faces, key=lambda x: x[0])
#         return True, confidence, bbox, landmarks


class RetinaFaceDetector:
    """Face detector using the `retinaface` package (PyTorch backend)."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize RetinaFace detector.

        Args:
            confidence_threshold: Minimum confidence for valid face detection.
        """
        self.confidence_threshold = confidence_threshold
        self.app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Detect a face using RetinaFace.
        Args:
            frame: Input frame (BGR).

        Returns:
            List of (confidence, bbox) tuples.
        """

        detections = self.app.get(frame)
        faces = []
        for det in detections:
            if float(det.det_score) < self.confidence_threshold:
                continue

            box = det.bbox
            if box is None or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            faces.append(det)

        return faces

    def detect_face(self, frame: np.ndarray) -> Face:
        """
        Detect if a face is present in the frame.
        Args:
            frame: Input frame (BGR).
        Returns:
            Tuple of (detected, confidence, bbox, landmarks).
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None
        return max(faces, key=lambda x: x.det_score)


class FaceDetector:
    """Face detector using OpenCV DNN models."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        sample_fps: float = 5.0,
        model_type: str = "retinaface",  # "caffe", "mtcnn", "retinaface", or "mediapipe"
        model_path: Optional[str] = __dir__.joinpath(
            "assets", "res10_300x300_ssd_iter_140000.caffemodel"
        ).as_posix(),
        config_path: Optional[str] = __dir__.joinpath(
            "assets", "deploy.prototxt"
        ).as_posix(),
    ):
        """
        Initialize face detector.

        Args:
            confidence_threshold: Minimum confidence for valid face detection
            sample_fps: Frame sampling rate for analysis
            model_type: Type of model to use ("caffe", "mtcnn", "retinaface", or "mediapipe")
            model_path: Path to model weights (if None, will try to use OpenCV's)
            config_path: Path to model config (for Caffe models)
        """
        self.confidence_threshold = confidence_threshold
        self.sample_fps = sample_fps
        self.model_type = model_type

        # Initialize detector
        self.detector = self._load_detector(model_path, config_path)

        # For landmark detection (optional, loaded on demand)
        self.landmark_detector = None

    def _load_detector(
        self, model_path: Optional[str], config_path: Optional[str]
    ) -> Optional[Any]:
        """
        Load face detection model.

        Args:
            model_path: Path to model weights
            config_path: Path to model config

        Returns:
            Loaded model or None if not available
        """
        try:
            if self.model_type == "caffe":
                # Try to load Caffe model
                if model_path and config_path:
                    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("Loaded custom Caffe face detection model")
                    return net
                else:
                    # Try OpenCV's built-in model
                    logger.warning(
                        "No custom model specified, face detection may be limited"
                    )
                    return None
            # elif self.model_type == "mtcnn":
            #     return MTCNNFaceDetector(confidence_threshold=self.confidence_threshold)
            if self.model_type == "retinaface":
                return RetinaFaceDetector(
                    confidence_threshold=self.confidence_threshold
                )
            else:
                logger.warning(f"Model type '{self.model_type}' not yet implemented")
                return None
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            return None

    # def detect_faces_dnn(
    #     self, frame: np.ndarray
    # ) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    #     """
    #     Detect faces using DNN model.

    #     Args:
    #         frame: Input frame (BGR)

    #     Returns:
    #         List of (confidence, bbox) tuples
    #     """
    #     # if self.model_type == "mtcnn" and isinstance(self.detector, MTCNNFaceDetector):
    #     #     return self.detector.detect_faces(frame)

    #     if self.detector is None:
    #         # Fallback to Haar cascades
    #         return self.detect_faces_haar(frame)

    #     h, w = frame.shape[:2]

    #     # Prepare blob
    #     blob = cv2.dnn.blobFromImage(
    #         frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
    #     )

    #     self.detector.setInput(blob)
    #     detections = self.detector.forward()

    #     faces = []
    #     for i in range(detections.shape[2]):
    #         confidence = detections[0, 0, i, 2]

    #         if confidence > self.confidence_threshold:
    #             # Get bounding box
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             x1, y1, x2, y2 = box.astype(int)

    #             # Convert to (x, y, w, h) format
    #             bbox = (x1, y1, x2 - x1, y2 - y1)
    #             faces.append((float(confidence), bbox))

    #     return faces

    # def detect_faces_haar(
    #     self, frame: np.ndarray
    # ) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    #     """
    #     Detect faces using Haar cascades (fallback method).

    #     Args:
    #         frame: Input frame (BGR)

    #     Returns:
    #         List of (confidence, bbox) tuples
    #     """
    #     # Load Haar cascade (cached)
    #     if not hasattr(self, "_haar_cascade"):
    #         self._haar_cascade = cv2.CascadeClassifier(
    #             cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    #         )

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces_rects = self._haar_cascade.detectMultiScale(
    #         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    #     )

    #     # Haar doesn't provide confidence, use fixed value
    #     faces = [(0.8, tuple(rect), None) for rect in faces_rects]
    #     return faces

    def detect_face(self, frame: np.ndarray) -> Optional[Face]:
        """
        Detect if a face is present in the frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Face object if a face is detected, None otherwise
        """
        # if self.model_type == "mtcnn" and isinstance(self.detector, MTCNNFaceDetector):
        #     return self.detector.detect_face(frame)
        if self.model_type == "retinaface" and isinstance(
            self.detector, RetinaFaceDetector
        ):
            return self.detector.detect_face(frame)
        # faces = self.detect_faces_dnn(frame)

        # if faces:
        #     # Take the most confident detection
        #     confidence, bbox = max(faces, key=lambda x: x[0])
        #     return True, confidence, bbox, None

        return None

    def analyze_video(
        self, video_path: str, progress_callback: Optional[callable] = None
    ) -> List[FaceDetection]:
        """
        Analyze video to detect face presence over time.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback function(current_frame, total_frames)

        Returns:
            List of FaceDetection objects with timestamps
        """
        logger.info(f"Analyzing face presence in {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip for sampling
        # frame_skip = max(1, int(fps / self.sample_fps))
        frame_skip = 0

        def apply(frame, timestamp) -> FaceDetection:
            face = self.detect_face(frame)
            return FaceDetection.from_face(timestamp=timestamp, face=face)

        face_detections = []
        stack = []
        frame_idx = 0
        pbar = tqdm(
            total=total_frames, desc="Analyzing Faces", unit="frame"
        )  # if progress_callback is None else None

        def update_progress(current, total):
            if progress_callback:
                progress_callback(current / total)
            if pbar is not None:
                pbar.update(current - pbar.n)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_skip > 1:
                    frame_idx += 1
                    frame_skip -= 1
                    stack.append((frame, timestamp))
                    continue
                timestamp = frame_idx / fps
                # Detect face
                result = apply(frame, timestamp)
                detected = result.detected

                if detected:
                    # If we just detected a face, we want to go back and fill in the previous frames in the stack
                    for framei, timestampi in stack:
                        face_detections.append(apply(framei, timestampi))
                else:
                    for framei, timestampi in stack:
                        face_detections.append(
                            FaceDetection(timestamp=timestampi)
                        )
                    frame_skip = max(1, int(fps / self.sample_fps))
                stack.clear()
                face_detections.append(result)
                update_progress(frame_idx, total_frames)

                frame_idx += 1

        finally:
            cap.release()

        logger.info(f"Analyzed {len(face_detections)} face detection samples")
        return face_detections

    def find_enter_time(
        self,
        face_detections: List[FaceDetection],
        door_closed_time: float,
        min_consecutive_frames: int = 3,
        search_window: float = 30.0,  # seconds after door closes
    ) -> Optional[Tuple[float, float]]:
        """
        Find enter_time: first stable face detection after door closes.

        Args:
            face_detections: List of face detections
            door_closed_time: When door closed (seconds)
            min_consecutive_frames: Minimum consecutive detections needed
            search_window: How long to search after door closes

        Returns:
            Tuple of (enter_time, confidence) or None
        """
        # Filter detections in search window
        candidates = [
            det
            for det in face_detections
            if door_closed_time <= det.timestamp <= door_closed_time + search_window
            and det.detected
            and det.det_score >= self.confidence_threshold
        ]

        if not candidates:
            return None

        # Find first sequence of consecutive detections
        consecutive_count = 0
        first_detection_time = None
        avg_confidence = 0.0

        for i, det in enumerate(candidates):
            # Check if this continues a sequence
            if i > 0:
                time_gap = det.timestamp - candidates[i - 1].timestamp
                expected_gap = 1.0 / self.sample_fps

                # Allow some tolerance in timing
                if time_gap <= expected_gap * 2:
                    consecutive_count += 1
                    avg_confidence += det.det_score
                else:
                    # Gap in sequence, restart
                    consecutive_count = 1
                    first_detection_time = det.timestamp
                    avg_confidence = det.det_score
            else:
                consecutive_count = 1
                first_detection_time = det.timestamp
                avg_confidence = det.det_score
            # Check if we have enough consecutive detections
            if consecutive_count >= min_consecutive_frames:
                avg_confidence /= consecutive_count
                return first_detection_time, avg_confidence

        # If we didn't find enough consecutive, return first detection with lower confidence
        if candidates:
            return candidates[0].timestamp, candidates[0].det_score * 0.5

        return None

    def find_exit_time(
        self,
        face_detections: List[FaceDetection],
        door_opened_time: float,
        search_window: float = 30.0,  # seconds before door opens
    ) -> Optional[Tuple[float, float]]:
        """
        Find exit_time: last face detection before door opens.

        Args:
            face_detections: List of face detections
            door_opened_time: When door opened (seconds)
            search_window: How long to search before door opens

        Returns:
            Tuple of (exit_time, confidence) or None
        """
        # Filter detections in search window
        candidates = [
            det
            for det in face_detections
            if door_opened_time - search_window <= det.timestamp <= door_opened_time
            and det.detected
            and det.det_score >= self.confidence_threshold
        ]

        if not candidates:
            return None

        # Return last detection
        last_detection = candidates[-1]
        return last_detection.timestamp, last_detection.det_score

    def visualize_detections(
        self,
        face_detections: List[FaceDetection],
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """
        Visualize face detection confidence over time.

        Args:
            face_detections: List of FaceDetection objects
        """
        timestamps = [det.timestamp for det in face_detections]
        confidences = [det.det_score for det in face_detections]
        if ax is None:
            plt.figure(figsize=(10, 5))
        elif ax:
            plt.sca(ax)
        plt.plot(timestamps, confidences, marker="o")
        plt.axhline(
            self.confidence_threshold,
            color="red",
            linestyle="--",
            label="Confidence Threshold",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Face Detection Confidence")
        plt.title("Face Detection Confidence Over Time")
        plt.legend()
        if show:
            plt.show()

        # face locations visualization (optional)
        # plt.figure(figsize=(10, 5))
        # plt.hist2d(
        #     [
        #         det.bbox[0] + det.bbox[2] // 2
        #         for det in face_detections
        #         if det.detected and det.bbox
        #     ],
        #     [
        #         det.bbox[1] + det.bbox[3] // 2
        #         for det in face_detections
        #         if det.detected and det.bbox
        #     ],
        #     bins=[50, 50],
        #     cmap="Reds",
        # )
        # plt.xlabel("Face Center X")
        # plt.ylabel("Face Center Y")
        # plt.title("Face Detection Locations")
        # plt.colorbar(label="Detection Count")
        # plt.show()

    def play_detections(self, video_path: str, face_detections: List[FaceDetection]):
        """
        Play video with face detection bounding boxes overlaid.

        Args:
            video_path: Path to video file
            face_detections: List of FaceDetection objects
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a mapping of timestamp to detection for quick lookup
        detection_map = {det.timestamp: det for det in face_detections}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Find closest detection for current time
            closest_time = min(
                detection_map.keys(), key=lambda t: abs(t - current_time)
            )
            if abs(closest_time - current_time) < (1.0 / self.sample_fps):
                det = detection_map[closest_time]
                if det.detected and det.bbox:
                    x0, y0, x1, y1 = det.bbox
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{det.det_score:.2f}",
                        (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                cv2.imshow("Face Detections", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def estimate_pose(self, face_detections: List[FaceDetection]) -> List[np.ndarray]:
        """
        Estimate head pose

        Args:
            face_detections: List of FaceDetection objects

        Returns:
                List of arrays containing head pose angles (yaw, pitch, roll) for each detection
        """
        estimates = []
        for det in face_detections:
            if det.detected:
                estimates.append(det.pose)
            else:
                estimates.append(None)
        return estimates

    def visualize_pose(
        self,
        face_detections: List[FaceDetection],
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """
        Visualize head pose over time.

        Args:
            face_detections: List of FaceDetection objects
        """
        timestamps = [det.timestamp for det in face_detections]
        poses = [det.pose for det in face_detections]

        yaw = [pose[0] if pose is not None else 0.0 for pose in poses]
        pitch = [pose[1] if pose is not None else 0.0 for pose in poses]
        roll = [pose[2] if pose is not None else 0.0 for pose in poses]

        if ax is None:
            plt.figure(figsize=(10, 5))
        elif ax:
            plt.sca(ax)

        plt.plot(timestamps, yaw, label="Yaw")
        plt.plot(timestamps, pitch, label="Pitch")
        plt.plot(timestamps, roll, label="Roll")
        plt.xlabel("Time (s)")
        plt.ylabel("Head Pose (degrees)")
        plt.title("Estimated Head Pose Over Time")
        plt.legend()
        if show:
            plt.show()
