"""
Door angle estimation using QR code detection and pose estimation.

The QR code is mounted on the door and its perspective changes as the door opens/closes.
This module detects the QR code and estimates door state (open/closed) and angle.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d

logger = logging.getLogger(__name__)


@dataclass
class DoorState:
    """Door state at a specific timestamp."""

    timestamp: float
    state: str  # "open", "closed", "unknown"
    confidence: float
    angle: Optional[float] = (
        None  # Estimated angle in degrees (0 = closed, 90 = fully open)
    )
    qr_corners: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp,
            "state": self.state,
            "confidence": self.confidence,
        }
        if self.angle is not None:
            result["angle"] = float(self.angle)
        if self.qr_corners is not None:
            result["qr_corners"] = self.qr_corners.tolist()
        return result

    def __str__(self):
        return f"DoorState(timestamp={self.timestamp:.2f}, state={self.state}, confidence={self.confidence:.2f}, angle={self.angle})"


class DoorAngleEstimator:
    """Estimate door angle and state using QR code detection."""

    def __init__(
        self,
        closed_threshold: float = 15.0,  # Angle below this is considered closed
        open_threshold: float = 45.0,  # Angle above this is considered open
        sample_fps: float = 5.0,  # Sample rate for analysis (frames per second)
        min_qr_area: float = 100.0,  # Minimum QR code area in pixels
    ):
        """
        Initialize door angle estimator.

        Args:
            closed_threshold: Maximum angle (degrees) to consider door closed
            open_threshold: Minimum angle (degrees) to consider door fully open
            sample_fps: Frame sampling rate for analysis
            min_qr_area: Minimum QR code area to consider valid detection
        """
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.sample_fps = sample_fps
        self.min_qr_area = min_qr_area
        self.qr_detector = cv2.QRCodeDetector()
        self.qr_edges = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype="float32"
        ).reshape((4, 1, 3))
        self.cmtx = np.array(
            [
                [1.41302865e03, 0.00000000e00, 9.34633235e02],
                [0.00000000e00, 1.41207149e03, 5.37667350e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        self.dist = np.array(
            [[0.05194222, -0.19840314, -0.0098829, -0.00885626, 0.33701845]]
        )

    def detect_qr_code(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect QR code in frame.

        Args:
            frame: Input video frame (BGR)

        Returns:
            Tuple of (detected, corners) where corners is a (4, 2) array if detected
        """
        # Convert to grayscale for better detection
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        # Detect QR code
        try:
            data, points, _ = self.qr_detector.detectAndDecode(gray)
        except cv2.error as e:
            print(f"OpenCV error during QR code detection: {e}")
            return False, None

        if data == "https://orcid.org/0000-0003-1912-804X":
            # Use first detected QR code
            corners = points[0].reshape(-1, 2)

            # Validate area
            area = cv2.contourArea(corners)
            if area >= self.min_qr_area:
                return True, corners

        return False, None

    def estimate_angle_from_corners(self, corners: np.ndarray) -> Tuple[float, float]:
        """
        Estimate door angle from QR code corners.

        Args:
            corners: QR code corners as (4, 2) array

        Returns:
            Tuple of (angle in degrees, confidence)
        """
        # determine the orientation of QR code coordinate system with respect to camera coorindate system.
        ret, rvec, tvec = cv2.solvePnP(self.qr_edges, corners, self.cmtx, self.dist)
        angle = 180.0  # Default to fully open if pose estimation fails
        if ret:
            R, _ = cv2.Rodrigues(rvec)
            z_axis = R[:, 2]  # Z-axis of the QR code in camera coordinates
            #  Calculate angle between Z-axis and camera's Z-axis (which is [0, 0, 1])
            angle = int(
                round(np.arccos(np.dot(z_axis, np.array([0, 0, -1]))) * (180.0 / np.pi))
            )

        else:
            logger.warning("Pose estimation failed for QR code corners")

        # Confidence can be based on well rvec and tvec can transform qr_edges to corners, which means the detected corners are consistent with the expected geometry of the QR code.
        projected_corners, _ = cv2.projectPoints(
            self.qr_edges, rvec, tvec, self.cmtx, self.dist
        )
        reprojection_error = np.linalg.norm(projected_corners.reshape(-1, 2) - corners)
        confidence = max(
            0.0, 1.0 - reprojection_error / 100.0
        )  # Simple confidence based on reprojection error

        return angle, confidence

    def analyze_video(
        self, video_path: str, progress_callback: Optional[callable] = None
    ) -> List[DoorState]:
        """
        Analyze video to extract door states over time.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback function(current_frame, total_frames)

        Returns:
            List of DoorState objects with timestamps
        """
        logger.info(f"Analyzing door states in {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip for sampling
        frame_skip = max(1, int(fps / self.sample_fps)) - 1

        door_states = []
        frame_idx = 0

        try:
            with tqdm(total=total_frames, desc="Processing video frames") as pbar:
                while True:
                    for _ in range(frame_skip):
                        if not cap.grab():  # Grab frame without decoding for efficiency
                            break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Sample at specified rate
                    timestamp = frame_idx / fps
                    # Detect QR code
                    detected, corners = self.detect_qr_code(frame)

                    if detected:
                        # Estimate angle
                        angle, confidence = self.estimate_angle_from_corners(corners)

                        # Determine state
                        if angle < self.closed_threshold:
                            state = "closed"
                        elif angle > self.open_threshold:
                            state = "open"
                        else:
                            state = "partial"

                        door_states.append(
                            DoorState(
                                timestamp=timestamp,
                                state=state,
                                confidence=confidence,
                                angle=angle,
                                qr_corners=corners,
                            )
                        )
                    else:
                        # No QR detected
                        door_states.append(
                            DoorState(
                                timestamp=timestamp, state="unknown", confidence=0.0
                            )
                        )

                    if progress_callback:
                        progress_callback(frame_idx, total_frames)
                    pbar.update(frame_skip + 1)
                    pbar.set_postfix({"Last state": door_states[-1].state})
                    frame_idx += 1 + frame_skip

        finally:
            cap.release()

        logger.info(f"Analyzed {len(door_states)} door state samples")
        return door_states

    def smooth_door_states(
        self, door_states: List[DoorState], kernel_size: int = 27
    ) -> List[DoorState]:
        """Apply median filter to smooth door angle estimates."""
        angles = np.array(
            [s.angle if s.angle is not None else 180 for s in door_states],
            dtype=np.uint8,
        )
        # plt.scatter(np.arange(len(angles)), angles, color="red", alpha=0.5, label="Raw Angles")
        eroded_angles = cv2.erode(angles.reshape(-1, 1), np.ones((kernel_size, 1), dtype=np.uint8), iterations=1)[:, 0]
        # plt.scatter(np.arange(len(eroded_angles)), eroded_angles, color="orange", alpha=0.5, label="Eroded Angles")
        smoothed_angles = cv2.medianBlur(eroded_angles.reshape(-1, 1), kernel_size)[:, 0]
        # plt.scatter(np.arange(len(smoothed_angles)), smoothed_angles, color="blue", alpha=0.5, label="Smoothed Angles")
        dilated_angles = cv2.dilate(smoothed_angles.reshape(-1, 1), np.ones((kernel_size, 1), dtype=np.uint8), iterations=1)[:, 0]
        # plt.scatter(np.arange(len(dilated_angles)), dilated_angles, color="green", alpha=0.5, label="Dilated Angles")
        # gaussian_angles = cv2.GaussianBlur(dilated_angles.reshape(-1, 1), (max(3, 2 * (kernel_size//4)+1), 1), 1)[:, 0]
        gaussian_angles = gaussian_filter1d(dilated_angles.astype('float32'), sigma=kernel_size/24, truncate=3.0)
        # plt.scatter(np.arange(len(gaussian_angles))+0.2, gaussian_angles, color="purple", alpha=0.5, label="Gaussian Angles")
        # missdetections = (angles > 55) 
        # relabeled_angles = angles.copy()
        # for i in np.where(missdetections)[0]:
        #     if i > 0 and i < len(angles) - 1:
        #         relabeled_angles[i] = max(
        #             np.min(angles[max(0, i - kernel_size // 2) : i]),
        #             np.min(angles[i + 1 : i + 1 + kernel_size // 2]),
        #         )
        # plt.scatter(np.arange(len(relabeled_angles)), relabeled_angles, color="green", alpha=0.6)
        # plt.xlabel("Sample Index")
        # plt.ylabel("Door Angle (degrees)")
        # plt.title("Smoothed Door Angle Estimates")
        # plt.legend()
        # plt.show()

        for i, s in enumerate(door_states):
            s.angle = float(gaussian_angles[i])
            s.state = "closed" if s.angle < self.closed_threshold else "open"
        return door_states

    def find_door_events(
        self,
        door_states: List[DoorState],
        min_duration: float = 10.0,  # Minimum duration in seconds for a state
        confidence_threshold: float = 0.5,
        max_sessions: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Find door open/close events from timeline of door states.

        Args:
            door_states: List of door states over time
            min_duration: Minimum duration for a state to be considered stable
            confidence_threshold: Minimum confidence for valid states

        Returns:
            List of events with type ("door_opened", "door_closed"), timestamp, and confidence
        """
        events = []

        if not door_states:
            return events

        # angle = np.array(
        #     [s.angle if s.angle is not None else 180 for s in door_states],
        #     dtype=np.uint8,
        # )
        # # apply a median filter to smooth out noise
        # angle = cv2.medianBlur(angle.reshape(-1, 1), 5)[:, 0]
        # for i, s in enumerate(door_states):
        #     s.angle = int(angle[i])
        #     s.state = "closed" if s.angle < self.closed_threshold else "open"
        # # plot with thickness as confidence
        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     [s.timestamp for s in door_states],
        #     [s.angle for s in door_states],
        #     # s=[s.confidence * 100 for s in door_states],
        #     color=[
        #         (
        #             "red"
        #             if s.state == "closed"
        #             else ("green" if s.state == "open" else "orange")
        #         )
        #         for s in door_states
        #     ],
        #     alpha=0.6,
        # )
        # plt.xlabel("Time (s)")
        # plt.ylabel("Door Angle (degrees)")
        # plt.title("Door Angle Over Time")
        # plt.show()

        # # Filter by confidence
        # valid_states = [s for s in door_states if s.confidence >= confidence_threshold]

        # if not valid_states:
        #     return events

        # Detect state transitions
        current_state = door_states[0]
        state_start_time = None
        history = []

        for state in door_states:
            if current_state.state != state.state:
                if history and state.state == history[-1].state:
                    history.append(state)
                else:
                    # New state, reset history or state is not stable yet, start new history
                    history = [state]
                # State changed
                duration = state.timestamp - current_state.timestamp

                # Check if previous state was stable enough
                if duration >= min_duration:
                    # Record transition
                    if current_state.state == "closed" and state.state == "open" and len(history) > 2:
                        events.append(
                            {
                                "type": "door_opened",
                                "timestamp": history[0].timestamp,
                                "confidence": history[0].confidence,
                                "method": "door_angle_qr",
                            }
                        )
                    elif current_state.state == "open" and state.state == "closed" and len(history) > 2:
                        events.append(
                            {
                                "type": "door_closed",
                                "timestamp": history[-1].timestamp,
                                "confidence": history[-1].confidence,
                                "method": "door_angle_qr",
                            }
                        )
                    elif len(history) <= 2:
                        # don't make decision if the state is not stable enough
                        continue
                    current_state = state
                    history = [] # reset history after recording event
        if events[0]["type"] == "door_opened":
            events = events[1:]
        if events and events[-1]["type"] == "door_closed":
            events.append(
                {
                    "type": "door_opened",
                    "timestamp": door_states[-1].timestamp,
                    "confidence": 0.0,
                    "method": "heuristic_noend",
                }
            )
        if len(events) > max_sessions * 2:
            logger.warning(
                f"Detected {len(events)} door events, which exceeds expected maximum of {max_sessions * 2}. Consider adjusting thresholds."
            )
            # keep longest duration events
            lengths = []
            for close, open in zip(events[::2], events[1::2]):
                lengths.append(open["timestamp"] - close["timestamp"])
            sorted_lengths = sorted(lengths, reverse=True)
            threshold_length = sorted_lengths[max_sessions - 1]
            filtered_events = []
            for close, open in zip(events[::2], events[1::2]):
                if open["timestamp"] - close["timestamp"] >= threshold_length:
                    filtered_events.extend([close, open])
            events = filtered_events

        logger.info(f"Found {len(events)} door events")
        # # visualize events on the angle plot
        # for event in events:
        #     plt.axvline(event["timestamp"], color="blue", linestyle="--", alpha=0.7)
        #     plt.text(
        #         event["timestamp"],
        #         90,
        #         event["type"],
        #         rotation=90,
        #         verticalalignment="bottom",
        #         horizontalalignment="right",
        #         color="blue",
        #     )
        # plt.show()
        return events

    def visualize_door_states(
        self, door_states: List[DoorState], offset: float = 0.0, ax=None, show=True
    ):
        """Visualize door states over time."""
        if ax is None:
            plt.figure(figsize=(10, 5))
        elif ax:
            plt.sca(ax)
        plt.scatter(
            [s.timestamp + offset for s in door_states],
            [s.angle if s.angle is not None else 180 for s in door_states],
            color=[
                (
                    "red"
                    if s.state == "closed"
                    else ("green" if s.state == "open" else "orange")
                )
                for s in door_states
            ],
            alpha=0.6,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Door Angle (degrees)")
        plt.title("Door Angle Over Time")
        if show:
            plt.show()

    def visualize_door_events(
        self, events: List[Dict[str, Any]], offset: float = 0.0, ax=None, show=True
    ):
        """Visualize door events on top of door states."""
        if ax is None:
            plt.figure(figsize=(10, 5))
        elif ax:
            plt.sca(ax)
        for event in events:
            plt.axvline(
                event["timestamp"] + offset, color="red", linestyle="--", alpha=0.7
            )
            plt.text(
                event["timestamp"] + offset,
                90,
                event["type"],
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="right",
                color="red",
            )
        if show:
            plt.show()
