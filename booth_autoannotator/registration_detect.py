"""
Registration phase detection using head motion and blink analysis.

The registration phase is characterized by:
- Elevated head motion (yaw, pitch, roll changes)
- Repeated blinking patterns
- Short duration (~10 seconds per segment)
- May consist of multiple sub-segments separated by pauses
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionMetrics:
    """Head motion metrics at a timestamp."""
    timestamp: float
    motion_energy: float  # Overall motion metric
    yaw_velocity: float = 0.0
    pitch_velocity: float = 0.0
    roll_velocity: float = 0.0
    blink_detected: bool = False
    eye_aspect_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "motion_energy": self.motion_energy,
            "yaw_velocity": self.yaw_velocity,
            "pitch_velocity": self.pitch_velocity,
            "roll_velocity": self.roll_velocity,
            "blink_detected": self.blink_detected,
            "eye_aspect_ratio": self.eye_aspect_ratio
        }


@dataclass
class RegistrationSegment:
    """A registration sub-segment."""
    start: float
    end: float
    confidence: float
    motion_score: float
    blink_score: float
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "motion_score": self.motion_score,
            "blink_score": self.blink_score,
            "tags": self.tags
        }


class RegistrationDetector:
    """Detect registration phase using head motion and blink analysis."""
    
    def __init__(
        self,
        motion_threshold: float = 0.5,  # Threshold for high motion
        blink_threshold: float = 0.3,   # Eye aspect ratio threshold for blink
        min_segment_duration: float = 5.0,  # Minimum registration segment duration (seconds)
        max_segment_duration: float = 20.0,  # Maximum registration segment duration (seconds)
        pause_threshold: float = 3.0,   # Pause duration to split segments (seconds)
        sample_fps: float = 10.0,       # Sampling rate for analysis
    ):
        """
        Initialize registration detector.
        
        Args:
            motion_threshold: Threshold for high motion energy
            blink_threshold: EAR threshold for blink detection
            min_segment_duration: Minimum duration for a registration segment
            max_segment_duration: Maximum duration for a registration segment
            pause_threshold: Pause duration to consider segment boundary
            sample_fps: Frame sampling rate
        """
        self.motion_threshold = motion_threshold
        self.blink_threshold = blink_threshold
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.pause_threshold = pause_threshold
        self.sample_fps = sample_fps
        
        # Face landmark detector
        self.face_detector = None
        self.landmark_detector = None
    
    def _load_face_detector(self):
        """Load face detector (Haar cascade)."""
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_face_landmarks(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """
        Detect facial landmarks.
        
        For simplicity, we'll estimate key points rather than using a full landmark detector.
        In production, use dlib or MediaPipe for accurate landmarks.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary with estimated landmarks
        """
        x, y, w, h = face_bbox
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Estimate eye regions (simple heuristic)
        # Eyes are typically in the upper half, centered horizontally
        eye_y = int(h * 0.3)
        eye_h = int(h * 0.15)
        
        left_eye_x = int(w * 0.25)
        right_eye_x = int(w * 0.65)
        eye_w = int(w * 0.2)
        
        left_eye_roi = face_roi[eye_y:eye_y+eye_h, left_eye_x:left_eye_x+eye_w]
        right_eye_roi = face_roi[eye_y:eye_y+eye_h, right_eye_x:right_eye_x+eye_w]
        
        # Compute eye aspect ratios (simplified)
        left_ear = self._compute_eye_aspect_ratio(left_eye_roi)
        right_ear = self._compute_eye_aspect_ratio(right_eye_roi)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        return {
            "left_eye": (x + left_eye_x, y + eye_y, eye_w, eye_h),
            "right_eye": (x + right_eye_x, y + eye_y, eye_w, eye_h),
            "eye_aspect_ratio": avg_ear
        }
    
    def _compute_eye_aspect_ratio(self, eye_roi: np.ndarray) -> float:
        """
        Compute eye aspect ratio (EAR) from eye ROI.
        
        This is a simplified version. In production, use landmark-based EAR.
        Here we use pixel intensity as a proxy (closed eyes are darker).
        
        Args:
            eye_roi: Eye region of interest
            
        Returns:
            Eye aspect ratio estimate
        """
        if eye_roi.size == 0:
            return 0.3  # Default value
        
        # Convert to grayscale
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi
        
        # Compute mean intensity (normalized)
        mean_intensity = np.mean(gray) / 255.0
        
        # Higher intensity suggests open eye
        # Map intensity to EAR-like value (0.15-0.4 range)
        ear = 0.15 + mean_intensity * 0.25
        
        return ear
    
    def compute_optical_flow_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Compute motion energy using optical flow.
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            face_bbox: Optional face bounding box to focus on face region
            
        Returns:
            Motion energy value
        """
        # If face bbox provided, compute flow only in face region
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Ensure within bounds
            h_img, w_img = prev_gray.shape
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            prev_roi = prev_gray[y:y+h, x:x+w]
            curr_roi = curr_gray[y:y+h, x:x+w]
        else:
            prev_roi = prev_gray
            curr_roi = curr_gray
        
        if prev_roi.size == 0 or curr_roi.size == 0:
            return 0.0
        
        # Compute optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, curr_roi,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Compute motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_energy = np.mean(magnitude)
            
            return float(motion_energy)
        except Exception as e:
            logger.warning(f"Optical flow computation failed: {e}")
            return 0.0
    
    def analyze_video(
        self,
        video_path: str,
        enter_time: float,
        search_duration: float = 60.0,  # Search for registration in first 60s after enter
        progress_callback: Optional[callable] = None
    ) -> List[MotionMetrics]:
        """
        Analyze video to extract motion metrics for registration detection.
        
        Args:
            video_path: Path to video file
            enter_time: Subject enter time (start searching after this)
            search_duration: How long to search for registration
            progress_callback: Optional progress callback
            
        Returns:
            List of motion metrics
        """
        logger.info(f"Analyzing motion for registration detection in {video_path}")
        
        self._load_face_detector()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Start from enter_time
        start_frame = int(enter_time * fps)
        end_frame = int((enter_time + search_duration) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Frame skip for sampling
        frame_skip = max(1, int(fps / self.sample_fps))
        
        motion_metrics = []
        prev_gray = None
        prev_face_bbox = None
        frame_idx = start_frame
        
        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample at specified rate
                if (frame_idx - start_frame) % frame_skip == 0:
                    timestamp = frame_idx / fps
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    faces = self.face_detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                    )
                    
                    face_bbox = tuple(faces[0]) if len(faces) > 0 else prev_face_bbox
                    
                    # Compute motion if we have previous frame
                    motion_energy = 0.0
                    if prev_gray is not None:
                        motion_energy = self.compute_optical_flow_motion(
                            prev_gray, gray, face_bbox
                        )
                    
                    # Detect blink if face detected
                    blink_detected = False
                    eye_aspect_ratio = 0.3
                    
                    if face_bbox is not None and len(face_bbox) == 4:
                        try:
                            landmarks = self.detect_face_landmarks(frame, face_bbox)
                            eye_aspect_ratio = landmarks["eye_aspect_ratio"]
                            blink_detected = eye_aspect_ratio < self.blink_threshold
                        except Exception as e:
                            logger.debug(f"Landmark detection failed: {e}")
                    
                    motion_metrics.append(MotionMetrics(
                        timestamp=timestamp,
                        motion_energy=motion_energy,
                        blink_detected=blink_detected,
                        eye_aspect_ratio=eye_aspect_ratio
                    ))
                    
                    prev_gray = gray
                    if face_bbox is not None:
                        prev_face_bbox = face_bbox
                    
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(motion_metrics)} motion metric samples")
        return motion_metrics
    
    def detect_registration_segments(
        self,
        motion_metrics: List[MotionMetrics],
        enter_time: float
    ) -> List[RegistrationSegment]:
        """
        Detect registration segments from motion metrics.
        
        Args:
            motion_metrics: Motion metrics timeline
            enter_time: Subject enter time
            
        Returns:
            List of registration segments
        """
        if not motion_metrics:
            return []
        
        # Compute motion and blink scores over sliding windows
        window_size = 5  # Number of samples in window
        segments = []
        
        i = 0
        while i < len(motion_metrics) - window_size:
            window = motion_metrics[i:i + window_size]
            
            # Compute window scores
            avg_motion = np.mean([m.motion_energy for m in window])
            blink_count = sum([m.blink_detected for m in window])
            blink_rate = blink_count / len(window)
            
            # Check if this looks like registration
            is_high_motion = avg_motion > self.motion_threshold
            has_blinks = blink_rate > 0.2  # At least 20% of samples have blinks
            
            if is_high_motion or has_blinks:
                # Found potential registration start
                segment_start = window[0].timestamp
                
                # Extend segment while motion remains high
                j = i + window_size
                while j < len(motion_metrics):
                    if motion_metrics[j].motion_energy > self.motion_threshold * 0.5:
                        j += 1
                    else:
                        # Check if this is a pause or end
                        # Look ahead for more motion
                        has_more_motion = False
                        for k in range(j, min(j + 5, len(motion_metrics))):
                            if motion_metrics[k].motion_energy > self.motion_threshold:
                                has_more_motion = True
                                break
                        
                        if has_more_motion:
                            j += 1  # Continue
                        else:
                            break  # End of segment
                
                segment_end = motion_metrics[j-1].timestamp if j > i else window[-1].timestamp
                duration = segment_end - segment_start
                
                # Validate segment duration
                if self.min_segment_duration <= duration <= self.max_segment_duration:
                    # Compute final scores
                    segment_metrics = motion_metrics[i:j]
                    motion_score = np.mean([m.motion_energy for m in segment_metrics])
                    blink_score = sum([m.blink_detected for m in segment_metrics]) / len(segment_metrics)
                    
                    confidence = min((motion_score / self.motion_threshold) * 0.5 + blink_score * 0.5, 1.0)
                    
                    tags = []
                    if motion_score > self.motion_threshold:
                        tags.append("head_motion")
                    if blink_score > 0.2:
                        tags.append("blink")
                    
                    segments.append(RegistrationSegment(
                        start=segment_start,
                        end=segment_end,
                        confidence=confidence,
                        motion_score=motion_score,
                        blink_score=blink_score,
                        tags=tags
                    ))
                    
                    # Skip past this segment
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        # Filter and merge segments
        filtered_segments = self._filter_and_merge_segments(segments)
        
        logger.info(f"Detected {len(filtered_segments)} registration segments")
        return filtered_segments
    
    def _filter_and_merge_segments(
        self,
        segments: List[RegistrationSegment]
    ) -> List[RegistrationSegment]:
        """
        Filter and merge registration segments.
        
        Args:
            segments: Raw detected segments
            
        Returns:
            Filtered and merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start)
        
        # Merge segments close together
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            gap = next_seg.start - current.end
            
            if gap < self.pause_threshold:
                # Merge
                current = RegistrationSegment(
                    start=current.start,
                    end=next_seg.end,
                    confidence=(current.confidence + next_seg.confidence) / 2,
                    motion_score=(current.motion_score + next_seg.motion_score) / 2,
                    blink_score=(current.blink_score + next_seg.blink_score) / 2,
                    tags=list(set(current.tags + next_seg.tags))
                )
            else:
                # Gap too large, finalize current
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        
        return merged
