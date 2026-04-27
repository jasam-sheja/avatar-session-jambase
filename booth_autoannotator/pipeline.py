"""
Main pipeline orchestrator for automatic annotation.

Coordinates all analysis modules, manages caching, and produces draft annotations.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
import logging
import wave

import cv2
import librosa
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from .models import (
    AnnotationDocument,
    VideoMetadata,
    AudioMetadata,
    Session,
    AnnotationValue,
    SessionEvents,
    SessionTimes,
    SessionPhases,
    RegistrationPhase,
    RegistrationSegment as ModelRegistrationSegment,
    SoloPhase,
    QAPhase,
    QAUtterance,
)
from .door_angle import DoorAngleEstimator
from .face_detect import FaceDetector, FaceDetection
from .speech_vad import SpeechSegment, VoiceActivityDetector
from .registration_detect import RegistrationDetector

logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    """Settings for the annotation pipeline."""

    # Door angle settings
    door_closed_threshold: float = 15.0
    door_open_threshold: float = 45.0
    door_sample_fps: float = 5.0

    # Face detection settings
    face_confidence_threshold: float = 0.5
    face_sample_fps: float = 5.0

    # VAD settings
    vad_backend: str = "silero"  # "energy", "webrtc", "silero"
    vad_energy_threshold: float = 0.01
    min_speech_duration_ms: int = 250

    # Registration settings
    registration_motion_threshold: float = 0.5
    registration_min_duration: float = 5.0
    registration_max_duration: float = 20.0

    # Solo speech heuristics
    solo_min_duration: float = 10.0
    solo_max_gap: float = 2.0

    # QA utterance settings
    qa_max_utterance_duration: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "door_closed_threshold": self.door_closed_threshold,
            "door_open_threshold": self.door_open_threshold,
            "door_sample_fps": self.door_sample_fps,
            "face_confidence_threshold": self.face_confidence_threshold,
            "face_sample_fps": self.face_sample_fps,
            "vad_backend": self.vad_backend,
            "vad_energy_threshold": self.vad_energy_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "registration_motion_threshold": self.registration_motion_threshold,
            "registration_min_duration": self.registration_min_duration,
            "registration_max_duration": self.registration_max_duration,
            "solo_min_duration": self.solo_min_duration,
            "solo_max_gap": self.solo_max_gap,
            "qa_max_utterance_duration": self.qa_max_utterance_duration,
        }


class AnnotationPipeline:
    """Main pipeline for automatic annotation."""

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize annotation pipeline.

        Args:
            settings: Pipeline settings
            cache_dir: Directory for caching intermediate results
        """
        self.settings = settings or PipelineSettings()
        self.cache_dir = cache_dir

        # Initialize analyzers
        self.door_estimator = DoorAngleEstimator(
            closed_threshold=self.settings.door_closed_threshold,
            open_threshold=self.settings.door_open_threshold,
            sample_fps=self.settings.door_sample_fps,
        )

        self.face_detector = FaceDetector(
            confidence_threshold=self.settings.face_confidence_threshold,
            sample_fps=self.settings.face_sample_fps,
        )

        self.vad = VoiceActivityDetector(
            backend=self.settings.vad_backend,
            energy_threshold=self.settings.vad_energy_threshold,
            min_speech_duration_ms=self.settings.min_speech_duration_ms,
        )

        self.registration_detector = RegistrationDetector(
            motion_threshold=self.settings.registration_motion_threshold,
            min_segment_duration=self.settings.registration_min_duration,
            max_segment_duration=self.settings.registration_max_duration,
        )

    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of file.

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read

        Returns:
            Hex digest of hash
        """
        hasher = hashlib.sha256()
        # pbar = tqdm(total=os.path.getsize(file_path), desc=f"Hashing {os.path.basename(file_path)}", unit="B", unit_scale=True, leave=False)
        with open(file_path, "rb") as f:
            hasher.update(f.read(chunk_size))
            f.seek(-chunk_size, os.SEEK_END)
            hasher.update(f.read(chunk_size))
            # while chunk := f.read(chunk_size):
            #     hasher.update(chunk)
            #     pbar.update(len(chunk))
        # pbar.close()
        return hasher.hexdigest()

    def _get_cache_path(self, video_path: str, cache_name: str) -> Optional[Path]:
        """
        Get cache file path for a specific analysis.

        Args:
            video_path: Path to video file
            cache_name: Name of the cache (e.g., "door_states", "face_detections")

        Returns:
            Path to cache file or None if caching disabled
        """
        if self.cache_dir is None:
            return None

        # Create cache directory
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use video filename as base
        video_name = Path(video_path).stem
        cache_file = cache_dir / f"{video_name}_{cache_name}.pkl"

        return cache_file

    def _save_cache(self, cache_path: Path, data: Any):
        """Save data to cache."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self, cache_path: Path) -> Optional[Any]:
        """Load data from cache."""
        try:
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded cache from {cache_path}")
                return data
            else:
                logger.info(f"No cache found at {cache_path}")
                exit(1)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            exit(1)
        return None

    def analyze1(
        self,
        video_path: Tuple[str, str],
        audio_path: Tuple[str, str],
        video_timestamps_path: Optional[Tuple[str, str]] = None,
        audio_timestamps_path: Optional[Tuple[str, str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        doc1: Optional[AnnotationDocument] = None,
        doc2: Optional[AnnotationDocument] = None,
        plot: bool = True,
        show: bool = True,
    ) -> Tuple[AnnotationDocument, AnnotationDocument]:
        """
        Run complete analysis pipeline.

        Args:
            video_path: Tuple of paths to video files
            audio_path: Tuple of paths to audio files
            video_timestamps_path: Tuple of paths to video timestamps files (optional)
            audio_timestamps_path: Tuple of paths to audio timestamps files (optional)
            progress_callback: Optional callback(stage_name, progress_fraction)

        Returns:
            Tuple of AnnotationDocuments with draft annotations
        """
        logger.info(f"Starting analysis pipeline for {video_path}")

        # Create annotation document
        if doc1 is None:
            doc1 = AnnotationDocument()
        if doc2 is None:
            doc2 = AnnotationDocument()

        def read_initial_timestamp(path: Optional[str]) -> float:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    while True:
                        line = f.readline().strip()
                        try:
                            return float(line)
                        except ValueError:
                            continue
            return 0.0

        def get_video_metadata(
            video_path: str, video_timestamps_path: Optional[str]
        ) -> VideoMetadata:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            time_stamp = read_initial_timestamp(video_timestamps_path)

            return VideoMetadata(
                file_path=video_path,
                file_hash=self._compute_file_hash(video_path),
                duration_sec=duration,
                fps=fps,
                resolution=[width, height],
                time_stamp=time_stamp,
            )

        doc1.video = get_video_metadata(video_path[0], video_timestamps_path[0])
        doc2.video = get_video_metadata(video_path[1], video_timestamps_path[1])

        # Get audio duration (simplified - assumes WAV)
        def get_audio_metadata(
            audio_path: str, audio_timestamps_path: Optional[str]
        ) -> AudioMetadata:
            audio_offset = read_initial_timestamp(audio_timestamps_path)
            try:
                with wave.open(audio_path, "rb") as wf:
                    audio_frames = wf.getnframes()
                    audio_rate = wf.getframerate()
                    audio_duration = (
                        audio_frames / audio_rate if audio_rate > 0 else 0.0
                    )
            except:
                audio_duration = doc1.video.duration_sec  # Fallback to video duration

            return AudioMetadata(
                file_path=audio_path,
                file_hash=self._compute_file_hash(audio_path),
                duration_sec=audio_duration,
                offset_sec=audio_offset,
            )

        doc1.audio = get_audio_metadata(audio_path[0], audio_timestamps_path[0])
        doc1.audio.offset_sec = doc1.video.time_stamp - doc1.audio.offset_sec
        doc2.audio = get_audio_metadata(audio_path[1], audio_timestamps_path[1])
        doc2.audio.offset_sec = doc2.video.time_stamp - doc2.audio.offset_sec

        logger.info(
            f"Video duration: {doc1.video.duration_sec:.2f}s, Audio duration: {doc1.audio.duration_sec:.2f}s, Audio offset: {doc1.audio.offset_sec:.2f}s"
        )
        logger.info(
            f"Video hash: {doc1.video.file_hash}, Audio hash: {doc1.audio.file_hash}"
        )
        logger.info(f"Video metadata: {doc1.video}")
        logger.info(f"Audio metadata: {doc1.audio}")

        # Update tool settings
        doc1.tool.settings = self.settings.to_dict()
        doc2.tool.settings = self.settings.to_dict()

        # Step 1: Door angle analysis
        door_states = [
            self._load_cache(self._get_cache_path(video_path[i], "door_states"))
            for i in range(2)
        ]
        # self.door_estimator.visualize_door_states(door_states[0])
        # self.door_estimator.visualize_door_states(door_states[1])
        door_states = list(map(self.door_estimator.smooth_door_states, door_states))
        door_events = list(map(self.door_estimator.find_door_events, door_states))
        # self.door_estimator.visualize_door_states(door_states, show=False)
        # self.door_estimator.visualize_door_events(door_events, ax=False)

        # Step 2: Face detection
        face_detections = [
            self._load_cache(self._get_cache_path(video_path[i], "face_detections"))
            for i in range(2)
        ]
        # self.face_detector.visualize_detections(face_detections)
        # self.face_detector.play_detections(video_path, face_detections)

        # # Step 3: Speech analysis
        # speech_segments = [None, None]
        speech_segments, audio_data = zip(
            *[
                self._load_cache(self._get_cache_path(audio_path, "speech_segments"))
                for audio_path in audio_path
            ]
        )
        speech_segments = list(map(self.vad.merge_segments, speech_segments))
        speech_segments[0] = self.vad.offset_segments(speech_segments[0], -doc1.audio.offset_sec)
        speech_segments[1] = self.vad.offset_segments(speech_segments[1], -doc2.audio.offset_sec)
        # self.vad.visualize_segments(speech_segments)

        # Step 4: Segment sessions
        doc1.sessions = self._segment_sessions(
            door_events[0], face_detections[0], speech_segments[0]
        )
        doc2.sessions = self._segment_sessions(
            door_events[1], face_detections[1], speech_segments[1]
        )
        if len(doc1.sessions) != len(doc2.sessions):
            logger.warning(
                f"Number of sessions in video 1 ({len(doc1.sessions)}) does not match video 2 ({len(doc2.sessions)})"
            )

        if plot:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for (
                audio_data_i,
                doc_i,
                speech_segments_i,
                door_events_i,
                face_detections_i,
                door_states_i,
                axes_i,
                sessions_i,
            ) in zip(
                audio_data,
                [doc1, doc2],
                speech_segments,
                door_events,
                face_detections,
                door_states,
                # [[axes[0], axes[1]], [axes[2], axes[3]]],
                [[axes[0], axes[1]], [axes[0], axes[2]]],
                [doc1.sessions, doc2.sessions],
            ):
                # print(axes_i); exit()
                video_offset = (- doc1.video.time_stamp + doc_i.video.time_stamp) / 1000.0
                self.vad.visualize_segments(speech_segments_i, offset=video_offset, ax=axes_i[0], show=False, zorder=0.1, color=("#0072B2" if doc_i == doc1 else '#D55E00'))
                librosa.display.waveshow(
                    audio_data_i.astype(np.float32),
                    sr=16000,
                    ax=axes_i[0],
                    color="#E69F00" if doc_i == doc1 else '#009E73',
                    offset=video_offset - doc_i.audio.offset_sec / 1000.0,
                    alpha=0.5,
                )
                self.door_estimator.visualize_door_events(
                    door_events_i, offset=video_offset, ax=axes_i[0], show=False
                )

                # self.vad.visualize_segments(speech_segments_i, offset=video_offset, ax=axes_i[1], show=False, zorder=0.1, color=("#0072B2" if doc_i == doc1 else '#D55E00'))
                self.face_detector.visualize_pose(face_detections_i, ax=axes_i[1], show=False)
                other_y_axis = axes_i[1].twinx()
                self.door_estimator.visualize_door_states(door_states_i, offset=video_offset, ax=other_y_axis, show=False)
                self.visualize_sessions(sessions_i, offset=video_offset, ax=axes_i[1], show=False)
            if show:
                plt.tight_layout()
                plt.show()
        # fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        # self.visualize_sessions(doc1.sessions, ax=axes[0], show=False)
        # self.visualize_sessions(doc2.sessions, ax=axes[1], show=False)
        # if show:
        #     plt.show()

        # pair_sessions = self._pair_sessions(doc1.sessions, doc2.sessions)
        # # cut the data of each session
        # data = []
        # for s1, s2 in pair_sessions:
        #     session_data = [{
        #         "session": s1,
        #         "face_detections": [d for d in face_detections[0] if s1.times.enter_time.t <= d.timestamp <= (s1.times.exit_time.t if s1.times.exit_time else float("inf"))],
        #         "speech_segments": [s for s in speech_segments[0] if s1.times.enter_time.t <= s.start <= (s1.times.exit_time.t if s1.times.exit_time else float("inf"))],
        #     },
        #     {
        #         "session": s2,
        #         "face_detections": [d for d in face_detections[1] if s2.times.enter_time.t <= d.timestamp <= (s2.times.exit_time.t if s2.times.exit_time else float("inf"))],
        #         "speech_segments": [s for s in speech_segments[1] if s2.times.enter_time.t <= s.start <= (s2.times.exit_time.t if s2.times.exit_time else float("inf"))],
        #     }
        #     ]
        #     data.append(session_data)

        
        return doc1, doc2

    def analyze2(
        self,
        video_path: Tuple[str, str],
        audio_path: Tuple[str, str],
        video_timestamps_path: Optional[Tuple[str, str]] = None,
        audio_timestamps_path: Optional[Tuple[str, str]] = None,
        doc1: Optional[AnnotationDocument] = None,
        doc2: Optional[AnnotationDocument] = None,
        plot=False,
        show=False,
    ) -> AnnotationDocument:

        sessions1 = doc1.sessions
        sessions2 = doc2.sessions
        # sort by enter time
        sessions1 = sorted(sessions1, key=lambda s: s.times.enter_time.t if s.times.enter_time else float("inf"))
        sessions2 = sorted(sessions2, key=lambda s: s.times.enter_time.t if s.times.enter_time else float("inf"))
        # load cache
        face_detections: List[List[FaceDetection]] = [
            self._load_cache(self._get_cache_path(video_path[i], "face_detections"))
            for i in range(2)
        ]
        speech_segments: List[List[SpeechSegment]] = None
        speech_segments, audio_data = zip(
            *[
                self._load_cache(self._get_cache_path(audio_path, "speech_segments"))
                for audio_path in audio_path
            ]
        )
        speech_segments = list(map(self.vad.merge_segments, speech_segments))
        speech_segments[0] = self.vad.offset_segments(speech_segments[0], -doc1.audio.offset_sec)
        speech_segments[1] = self.vad.offset_segments(speech_segments[1], -doc2.audio.offset_sec)
        # Step 5: For each session, detect registration

        for i, (session1, session2) in enumerate(zip(sessions1, sessions2)):
            logger.info(f"Processing session {session1.session_id}")

            # cut the data of each session
            face_detections1 = [d for d in face_detections[0] if session1.times.enter_time.t <= d.timestamp <= (session1.times.exit_time.t if session1.times.exit_time else float("inf"))]
            face_detections2 = [d for d in face_detections[1] if session2.times.enter_time.t <= d.timestamp <= (session2.times.exit_time.t if session2.times.exit_time else float("inf"))]
            speech_segments1 = [s for s in speech_segments[0] if session1.times.enter_time.t <= s.start <= (session1.times.exit_time.t if session1.times.exit_time else float("inf"))]
            speech_segments2 = [s for s in speech_segments[1] if session2.times.enter_time.t <= s.start <= (session2.times.exit_time.t if session2.times.exit_time else float("inf"))]

            # syncronize to video1
            offset = (doc2.video.time_stamp - doc1.video.time_stamp) / 1000
            for det in face_detections2:
                det.timestamp += offset 
            for seg in speech_segments2:
                seg.start += offset 
                seg.end += offset

            timestamp0 = min(
                session1.times.enter_time.t if session1.times.enter_time else float("inf"),
                session2.times.enter_time.t + offset if session2.times.enter_time else float("inf"),
            )
            print(f"Session {session1.session_id}: timestamp0={timestamp0}, session1 enter={session1.times.enter_time.t if session1.times.enter_time else 'None'}, session1 exit={session1.times.exit_time.t if session1.times.exit_time else 'None'}, session2 enter={session2.times.enter_time.t if session2.times.enter_time else 'None'}, session2 exit={session2.times.exit_time.t if session2.times.exit_time else 'None'}")
            print(f"Session {session1.session_id}: {len(face_detections1)} face detections in video 1, {len(face_detections2)} face detections in video 2")
            print(f"Session {session1.session_id}: {len(speech_segments1)} speech segments in audio 1, {len(speech_segments2)} speech segments in audio 2")
            duration = max(
                session1.times.exit_time.t if session1.times.exit_time else 0,
                session2.times.exit_time.t if session2.times.exit_time else 0,
            ) - timestamp0 + 1
            print(f"Session {session1.session_id}: duration={duration:.2f}s")
            is_speech = np.zeros((2, int(duration)), dtype=np.uint8)
            for seg in speech_segments1:
                start_idx = int(seg.start - timestamp0)
                end_idx = int(seg.end - timestamp0)
                is_speech[0, start_idx:end_idx] = 255
            for seg in speech_segments2:
                start_idx = int(seg.start - timestamp0)
                end_idx = int(seg.end - timestamp0)
                is_speech[1, start_idx:end_idx] = 255
            # roll,pitch,yaw magnitudes
            rot_magnitudes = np.zeros((2, int(duration)), dtype=np.float32)
            for det in face_detections1:
                idx = int(det.timestamp - timestamp0)
                rot_magnitudes[0, idx] = max(rot_magnitudes[0, idx], np.linalg.norm(det.pose))
            for det in face_detections2:
                idx = int(det.timestamp - timestamp0)
                rot_magnitudes[1, idx] = max(rot_magnitudes[1, idx], np.linalg.norm(det.pose))

            monolog = cv2.morphologyEx(is_speech, cv2.MORPH_CLOSE, np.ones((1, 11), dtype=np.uint8))
            monolog = cv2.morphologyEx(monolog, cv2.MORPH_OPEN, np.ones((1, 21), dtype=np.uint8))
            monolog_idx = np.argmax(monolog, axis=1)
            monolog_timestamp = timestamp0 + monolog_idx

            convo = (is_speech[0] | is_speech[1])[None, :]
            convo = cv2.morphologyEx(convo, cv2.MORPH_CLOSE, np.ones((1, 11), dtype=np.uint8))
            convo = cv2.morphologyEx(convo, cv2.MORPH_OPEN, np.ones((1, 21), dtype=np.uint8))
            convo_start_idx = np.argmax(convo, axis=1)
            convo_start_timestamp = [timestamp0 + convo_start_idx]*2
            convo_end_idx = convo.shape[1] - np.argmax(convo[:, ::-1], axis=1) - 1
            convo_end_timestamp = [timestamp0 + convo_end_idx]*2

            heuristic = np.ones_like(is_speech, dtype=np.float32)
            heuristic[:, :3*heuristic.shape[1]//100] = 0  # Only consider time after session start for registration
            heuristic[:, monolog_idx.min():] = 0  # Only consider time before monolog starts for registration
            registration = rot_magnitudes * heuristic  # Only consider non-speech segments for registration
            # registration = cv2.dilate(registration, np.ones((1, 11), dtype=np.float32), iterations=2)
            registration = cv2.morphologyEx(registration, cv2.MORPH_CLOSE, np.ones((1, 11), dtype=np.float32))
            registration = cv2.morphologyEx(registration, cv2.MORPH_OPEN, np.ones((1, 15), dtype=np.float32))
            registration = registration > 0
            registration_idx = np.argmax(registration, axis=1) # take the first registration point
            registration_timestamp = timestamp0 + registration_idx - 10  # shift back a bit to account for smoothing
            

            # visualize for debugging
            fig, ax = plt.subplots(5, 1, figsize=(12, 6), sharex=True)
            ax[0].plot(rot_magnitudes[0], label="Video 1 Rotation Magnitude")
            ax[0].plot(rot_magnitudes[1], label="Video 2 Rotation Magnitude")
            ax[0].legend()
            ax[1].imshow(is_speech, aspect="auto", cmap="gray", alpha=0.5, interpolation="nearest")
            ax[1].set_ylabel("Speech Activity")
            ax[2].imshow(registration, aspect="auto", cmap="gray", alpha=0.5, interpolation="nearest")
            ax[2].set_ylabel("Registration Heuristic")
            ax[2].legend()
            ax[3].imshow(monolog, aspect="auto", cmap="gray", alpha=0.5, interpolation="nearest")
            ax[3].set_ylabel("Monolog Heuristic")
            ax[4].imshow(convo, aspect="auto", cmap="gray", alpha=0.5, interpolation="nearest")
            ax[4].set_ylabel("Conversation Heuristic")

            # undo synchronization 
            registration_timestamp[1] -= offset
            monolog_timestamp[1] -= offset
            convo_start_timestamp[1] -= offset
            convo_end_timestamp[1] -= offset

            def _sec_to_time(sec: float) -> str:
                m = int(sec // 60)
                s = int(sec % 60)
                return f"{m:02d}:{s:02d}"

            print(f"Session {session1.session_id}: registration timestamp={_sec_to_time(registration_timestamp[0])} (video 1), {_sec_to_time(registration_timestamp[1])} (video 2)")
            print(f"Session {session1.session_id}: monolog timestamp={_sec_to_time(monolog_timestamp[0])} (video 1), {_sec_to_time(monolog_timestamp[1])} (video 2)")
            print(f"Session {session1.session_id}: conversation timestamp={_sec_to_time(convo_start_timestamp[0])} (video 1), {_sec_to_time(convo_start_timestamp[1])} (video 2) to {_sec_to_time(convo_end_timestamp[0])} (video 1), {_sec_to_time(convo_end_timestamp[1])} (video 2)")

            plt.show()
            exit()


            # Detect registration
            if session.times.enter_time:
                enter_t = session.times.enter_time.t

                reg_cache = self._get_cache_path(video_path, f"registration_s{i}")
                motion_metrics = self._load_cache(reg_cache) if reg_cache else None

                if motion_metrics is None:
                    motion_metrics = self.registration_detector.analyze_video(
                        video_path, enter_t
                    )
                    if reg_cache:
                        self._save_cache(reg_cache, motion_metrics)

                reg_segments = self.registration_detector.detect_registration_segments(
                    motion_metrics, enter_t
                )

                # Add to session
                if reg_segments:
                    session.phases.registration.start = AnnotationValue(
                        t=reg_segments[0].start,
                        source="auto",
                        confidence=reg_segments[0].confidence,
                        method="motion+blink",
                    )

                    for seg in reg_segments:
                        session.phases.registration.segments.append(
                            ModelRegistrationSegment(
                                start=AnnotationValue(
                                    t=seg.start,
                                    source="auto",
                                    confidence=seg.confidence,
                                    method="motion+blink",
                                ),
                                end=AnnotationValue(
                                    t=seg.end,
                                    source="auto",
                                    confidence=seg.confidence,
                                    method="motion+blink",
                                ),
                                tags=seg.tags,
                            )
                        )

            # Detect solo and QA phases
            self._detect_speech_phases(session, speech_segments)

        doc1.sessions = sessions

        if progress_callback:
            progress_callback("complete", 1.0)

        logger.info(f"Analysis complete. Found {len(sessions)} sessions.")
        return doc1

    def _segment_sessions(
        self,
        door_events: List[Dict[str, Any]],
        face_detections: List,
        speech_segments: List,
    ) -> List[Session]:
        """
        Segment video into individual sessions based on door events.

        Args:
            door_events: Door open/close events
            face_detections: Face detection timeline
            speech_segments: Speech segments from audio

        Returns:
            List of Session objects
        """
        sessions = []

        # Find door close events (session starts)
        door_closed_events = [e for e in door_events if e["type"] == "door_closed"]
        door_opened_events = [e for e in door_events if e["type"] == "door_opened"]

        for i, close_event in enumerate(door_closed_events):
            session_id = f"S{i+1:03d}"
            session = Session(session_id=session_id)

            # Set door events
            session.events.door_closed = AnnotationValue(
                t=close_event["timestamp"],
                source="auto",
                confidence=close_event["confidence"],
                method=close_event["method"],
            )

            # Find corresponding door open event
            next_open = None
            for open_event in door_opened_events:
                if open_event["timestamp"] > close_event["timestamp"]:
                    next_open = open_event
                    break

            if next_open:
                session.events.door_opened = AnnotationValue(
                    t=next_open["timestamp"],
                    source="auto",
                    confidence=next_open["confidence"],
                    method=next_open["method"],
                )

            # Find enter_time
            enter_result = self.face_detector.find_enter_time(
                face_detections, close_event["timestamp"]
            )

            if enter_result:
                enter_time, enter_conf = enter_result
                session.times.enter_time = AnnotationValue(
                    t=enter_time,
                    source="auto",
                    confidence=enter_conf,
                    method="door+face",
                )

            # Find exit_time if we have door open event
            if next_open:
                exit_result = self.face_detector.find_exit_time(
                    face_detections, next_open["timestamp"]
                )

                if exit_result:
                    exit_time, exit_conf = exit_result
                    session.times.exit_time = AnnotationValue(
                        t=exit_time,
                        source="auto",
                        confidence=exit_conf,
                        method="door+face",
                    )

            sessions.append(session)

        return sessions
    
    def visualize_sessions(self, sessions: List[Session], offset: float = 0.0, ax: Optional[plt.Axes] = None, show: bool = True):
        """
        Visualize sessions on video timeline.

        Args:
            sessions: List of Session objects
            offset: Time offset for visualization
            ax: Optional matplotlib Axes to plot on
            show: Whether to call plt.show()
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        for session in sessions:
            enter_t = session.times.enter_time.t if session.times.enter_time else None
            exit_t = session.times.exit_time.t if session.times.exit_time else None

            if enter_t is not None:
                ax.axvline(enter_t + offset, color="green", linestyle="--", label="Enter Time")
            if exit_t is not None:
                ax.axvline(exit_t + offset, color="red", linestyle="--", label="Exit Time")

            if enter_t is not None and exit_t is not None:
                ax.axvspan(
                    enter_t + offset,
                    exit_t + offset,
                    color="green",
                    alpha=0.1,
                    label="Session Duration",
                )
        if show:
            plt.show()

    def _pair_sessions(self, sessions1: List[Session], sessions2: List[Session]) -> List[Tuple[Session, Session]]:
        """
        Pair sessions from two videos based on enter times.

        Args:
            sessions1: List of sessions from video 1
            sessions2: List of sessions from video 2
        Returns:
            List of paired sessions (session1, session2)
        """
        paired = []
        for s1 in sessions1:
            best_match = None
            best_iou = 0
            for s2 in sessions2:
                min_enter = min(s1.times.enter_time.t, s2.times.enter_time.t)
                max_enter = max(s1.times.enter_time.t, s2.times.enter_time.t)
                min_exit = min(s1.times.exit_time.t, s2.times.exit_time.t)
                max_exit = max(s1.times.exit_time.t, s2.times.exit_time.t)
                if max_enter >= min_exit:  # No overlap
                    continue # iou = 0
                iou = (min_exit - max_enter) / (max_exit - min_enter)
                if iou > best_iou:
                    best_iou = iou
                    best_match = s2
            if best_match:
                s1.pair_session_id = best_match.session_id
                best_match.pair_session_id = s1.session_id
                paired.append((s1, best_match))
        return paired
    def _detect_speech_phases(self, session: Session, speech_segments: List):
        """
        Detect solo and QA speech phases within a session.

        Args:
            session: Session to analyze
            speech_segments: All speech segments from audio
        """
        if not session.times.enter_time:
            return

        # Get session time bounds
        enter_t = session.times.enter_time.t
        exit_t = session.times.exit_time.t if session.times.exit_time else float("inf")

        # Filter speech segments to this session
        session_speech = [
            seg for seg in speech_segments if enter_t <= seg.start <= exit_t
        ]

        if not session_speech:
            return

        # Heuristic: solo speech is typically the first continuous speech after registration
        # Find registration end
        reg_end = enter_t
        if session.phases.registration.segments:
            reg_end = max(seg.end.t for seg in session.phases.registration.segments)

        # Find first significant speech after registration
        solo_candidates = [
            seg
            for seg in session_speech
            if seg.start >= reg_end + 2.0  # Small gap after registration
        ]

        if solo_candidates:
            # Group consecutive segments as solo speech
            solo_start = solo_candidates[0].start
            solo_end = solo_candidates[0].end

            for seg in solo_candidates[1:]:
                gap = seg.start - solo_end
                if gap <= self.settings.solo_max_gap:
                    solo_end = seg.end
                else:
                    break

            # Check minimum duration
            if (solo_end - solo_start) >= self.settings.solo_min_duration:
                session.phases.solo.start = AnnotationValue(
                    t=solo_start, source="auto", confidence=0.7, method="vad"
                )
                session.phases.solo.end = AnnotationValue(
                    t=solo_end, source="auto", confidence=0.7, method="vad"
                )

                # Remaining speech segments are QA
                qa_start = solo_end + self.settings.solo_max_gap
                qa_segments = [seg for seg in session_speech if seg.start >= qa_start]

                # Split into utterances
                utterances = self.vad.split_into_utterances(
                    qa_segments,
                    max_utterance_duration=self.settings.qa_max_utterance_duration,
                )

                for utt in utterances:
                    session.phases.qa.utterances.append(
                        QAUtterance(
                            start=AnnotationValue(
                                t=utt.start,
                                source="auto",
                                confidence=utt.confidence,
                                method="vad",
                            ),
                            end=AnnotationValue(
                                t=utt.end,
                                source="auto",
                                confidence=utt.confidence,
                                method="vad",
                            ),
                            label="TBD",
                            confidence_label=0.0,
                            method_label="manual",
                        )
                    )

    def precache(
        self,
        video_path: str,
        audio_path: str,
    ) -> AnnotationDocument:
        """
        Run complete analysis pipeline.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            video_timestamps_path: Path to video timestamps file (optional)
            audio_timestamps_path: Path to audio timestamps file (optional)
            progress_callback: Optional callback(stage_name, progress_fraction)

        Returns:
            AnnotationDocument with draft annotations
        """
        assert self.cache_dir is not None, "Cache directory must be set for precaching"
        logger.info(f"Starting precaching pipeline for {video_path}")

        # Step 1: Door angle analysis
        door_cache = self._get_cache_path(video_path, "door_states")
        if not door_cache.exists():
            door_states = self.door_estimator.analyze_video(video_path)
            self._save_cache(door_cache, door_states)

        # Step 2: Face detection
        face_cache = self._get_cache_path(video_path, "face_detections")
        if not face_cache.exists():
            face_detections = self.face_detector.analyze_video(video_path)
            self._save_cache(face_cache, face_detections)

        # Step 3: Speech analysis
        speech_cache = self._get_cache_path(audio_path, "speech_segments")
        if not speech_cache.exists():
            speech_segments, audio_data = self.vad.analyze_audio(audio_path)
            self._save_cache(speech_cache, (speech_segments, audio_data))


def export_json(doc: AnnotationDocument, output_path: str, indent: int = 2):
    """
    Export annotation document to JSON file.

    Args:
        doc: Annotation document
        output_path: Output JSON file path
        indent: JSON indentation
    """
    doc.save(output_path, indent=indent)
    logger.info(f"Exported annotations to {output_path}")
