"""
Voice Activity Detection (VAD) for speech segmentation.

Processes separate audio file to detect speech segments.
Supports multiple VAD backends (WebRTC VAD, Silero VAD, energy-based).
"""

from matplotlib import pyplot as plt
import numpy as np
import wave
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import resampy
import torch
import librosa
from tqdm import tqdm
# import webrtcvad
from noisereduce.torchgate import TorchGate
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A continuous speech segment."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0

    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "duration": self.duration(),
        }


class SileroVADWrapper:
    """Wrapper for Silero VAD to match our SpeechSegment format."""

    def __init__(self):
        self.model = load_silero_vad()
        self.tg = TorchGate(16000).cuda()  # For noise reduction if needed

    def detect_speech(self, audio_path: str) -> List[SpeechSegment]:
        """Detect speech segments in an audio file."""
        audio_data = read_audio(audio_path).cuda()
        audio_data = self.tg(audio_data.unsqueeze(0)).squeeze()  # Apply noise reduction if needed
        speech_timestamps = get_speech_timestamps(audio_data.cpu(), self.model, return_seconds=True)

        segments = []
        for ts in speech_timestamps:
            start_time = ts["start"]
            end_time = ts["end"]
            segments.append(SpeechSegment(start=start_time, end=end_time))

        return segments, audio_data

class VoiceActivityDetector:
    """Voice Activity Detection for speech segmentation."""

    def __init__(
        self,
        backend: str = "silero",  # "energy", "webrtc", "silero"
        frame_duration_ms: int = 30,  # Frame duration in milliseconds
        aggressiveness: int = 2,  # VAD aggressiveness (0-3 for WebRTC)
        energy_threshold: float = 0.01,  # Energy threshold for energy-based VAD
        min_speech_duration_ms: int = 250,  # Minimum speech segment duration
        min_silence_duration_ms: int = 300,  # Minimum silence between segments
        merge_gap_ms: int = 300,  # Merge segments separated by gaps smaller than this
    ):
        """
        Initialize VAD.

        Args:
            backend: VAD backend to use
            frame_duration_ms: Duration of each analysis frame
            aggressiveness: VAD aggressiveness (higher = more strict)
            energy_threshold: Threshold for energy-based VAD
            min_speech_duration_ms: Minimum duration for a speech segment
            min_silence_duration_ms: Minimum silence duration between segments
            merge_gap_ms: Merge segments with gaps smaller than this
        """
        self.backend = backend
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.merge_gap_ms = merge_gap_ms

    @property
    def vad(self):
        """Get VAD instance."""
        if not hasattr(self, "_vad"):
            self._vad = self._load_vad_backend()
        return self._vad

    def _load_vad_backend(self):
        """Load VAD backend."""
        self.noise_gate = TorchGate(16000).cuda()  # For noise reduction if needed
        if self.backend == "webrtc":
            try:
                import webrtcvad

                vad = webrtcvad.Vad(self.aggressiveness)
                logger.info("Loaded WebRTC VAD")
                return vad
            except ImportError:
                logger.warning(
                    "webrtcvad not available, falling back to energy-based VAD"
                )
                self.backend = "energy"
                return None
        elif self.backend == "silero":
            try:
                self.silero_vad = SileroVADWrapper()
                logger.info("Loaded Silero VAD")
            except ImportError:
                logger.warning(
                    "torch not available for Silero VAD, falling back to energy-based VAD"
                )
                self.backend = "energy"
                return None
        else:
            # Energy-based VAD (no external dependencies)
            logger.info("Using energy-based VAD")
            return None

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file (WAV format)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            with wave.open(audio_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()

                # Read audio data
                audio_bytes = wf.readframes(n_frames)

                # Convert to numpy array
                if wf.getsampwidth() == 2:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                else:
                    raise ValueError(f"Unsupported sample width: {wf.getsampwidth()}")

                # Convert stereo to mono if needed
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def compute_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Compute energy of audio chunk.

        Args:
            audio_chunk: Audio data

        Returns:
            Normalized energy value
        """
        # Normalize to [-1, 1]
        audio_float = audio_chunk.astype(np.float32) / 32768.0

        # Compute RMS energy
        energy = np.sqrt(np.mean(audio_float**2))
        return energy

    def detect_speech_energy(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float, bool]]:
        """
        Detect speech using energy-based method.

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            List of (start_time, end_time, is_speech) tuples
        """
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        speech_frames = []

        # Process audio in frames
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i : i + frame_size]
            energy = self.compute_energy(frame)

            is_speech = energy > self.energy_threshold
            start_time = i / sample_rate
            end_time = (i + frame_size) / sample_rate

            speech_frames.append((start_time, end_time, is_speech))

        return speech_frames

    def detect_speech_webrtc(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float, bool]]:
        """
        Detect speech using WebRTC VAD.

        Args:
            audio_data: Audio data (must be 8kHz, 16kHz, or 32kHz)
            sample_rate: Sample rate

        Returns:
            List of (start_time, end_time, is_speech) tuples
                start_time and end_time are in seconds, is_speech is a boolean
        """
        if self.vad is None:
            return self.detect_speech_energy(audio_data, sample_rate)
        
        # fig, axes = plt.subplots(3, 1, figsize=(12, 6))
        # librosa.display.waveshow(audio_data.astype(np.float32), sr=sample_rate, ax=axes[0])
        # axes[0].set_title("Audio Waveform")

        # WebRTC VAD requires specific sample rates
        if sample_rate not in [8000, 16000, 32000]:
            logger.warning(
                f"WebRTC VAD requires 8/16/32 kHz, got {sample_rate}. Resampling..."
            )
            target_rate = 16000
            audio_data = resampy.resample(audio_data, sample_rate, target_rate, filter='kaiser_fast', parallel=True).astype(np.int16)
            sample_rate = target_rate
            # librosa.display.waveshow(audio_data.astype(np.float32), sr=sample_rate, ax=axes[1])
            # axes[1].set_title(f"Resampled Audio ({sample_rate} Hz)")

        # audio_torch = torch.from_numpy(audio_data).float().cuda() / 32768.0
        # audio_torch = self.noise_gate(audio_torch.unsqueeze(0))  # Apply noise reduction if needed
        # audio_data = (audio_torch.squeeze().cpu().numpy() * 32768).astype(np.int16)
        # librosa.display.waveshow(audio_data.astype(np.float32), sr=sample_rate, ax=axes[2])
        # axes[2].set_title("Denoised Audio (if noise gate applied)")
        # plt.tight_layout()
        # plt.show()

        frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        speech_frames = []

        # Process audio in frames
        for i in tqdm(range(0, len(audio_data) - frame_size, frame_size)):
            frame = audio_data[i : i + frame_size]
            frame_bytes = frame.tobytes()

            try:
                is_speech = self.vad.is_speech(frame_bytes, sample_rate)
            except Exception as e:
                logger.warning(f"WebRTC VAD error: {e}, falling back to energy")
                print(f"len(frame_bytes)={len(frame_bytes)}, sample_rate={sample_rate}, frame_size={frame_size}")
                raise e
                is_speech = self.compute_energy(frame) > self.energy_threshold

            start_time = i / sample_rate
            end_time = (i + frame_size) / sample_rate

            speech_frames.append((start_time, end_time, is_speech))

        # for seg in self.merge_segments(speech_frames):
        #     axes[2].axvspan(seg.start, seg.end, alpha=0.3)
        # plt.tight_layout()
        # plt.show()

        return speech_frames

    def _resample_simple(
        self, audio_data: np.ndarray, orig_rate: int, target_rate: int
    ) -> np.ndarray:
        """Simple resampling (nearest neighbor). For production, use scipy or librosa."""
        duration = len(audio_data) / orig_rate
        target_length = int(duration * target_rate)
        indices = np.linspace(0, len(audio_data) - 1, target_length).astype(int)
        return audio_data[indices]

    def merge_segments(
        self, speech_frames: List[Tuple[float, float, bool]]
    ) -> List[SpeechSegment]:
        """
        Merge speech frames into continuous segments.

        Args:
            speech_frames: List of frame-level speech detections

        Returns:
            List of merged speech segments
        """
        if not speech_frames:
            return []

        if isinstance(speech_frames[0], SpeechSegment):
            speech_frames = [(seg.start, seg.end, True) for seg in speech_frames]

        segments = []
        def try_add_segment(start, end):
            duration_ms = (end - start) * 1000
            if duration_ms >= self.min_speech_duration_ms:
                segments.append(
                    SpeechSegment(
                        start=start,
                        end=end,
                        confidence=0.9 if self.backend == "webrtc" else 0.7,
                    )
                )
        current_start = None
        current_end = None
        last_speech_end = None

        for start_time, end_time, is_speech in speech_frames:
            if is_speech:
                if current_start is None:
                    # Start new segment
                    current_start = start_time
                elif start_time - current_end > self.min_silence_duration_ms / 1000.0:
                    # End of previous segment, start new one
                    try_add_segment(current_start, current_end)
                    current_start = start_time
                current_end = end_time

        # Handle final segment
        if current_start is not None and last_speech_end is not None:
            try_add_segment(current_start, last_speech_end)


        return segments

    
    def offset_segments(self, segments: List[SpeechSegment], offset_ms: float) -> List[SpeechSegment]:
        """
        Apply time offset to segments.

        Args:
            segments: List of speech segments
            offset_ms: Time offset in milliseconds (positive or negative)

        Returns:
            List of offset segments
        """
        offset_seconds = offset_ms / 1000.0
        offset_segments = [
            SpeechSegment(
                start=seg.start + offset_seconds,
                end=seg.end + offset_seconds,
                confidence=seg.confidence,
            )
            for seg in segments
        ]
        return offset_segments

    def analyze_audio(
        self, audio_path: str, progress_callback: Optional[callable] = None
    ) -> List[SpeechSegment]:
        """
        Analyze audio file to extract speech segments.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
            List of speech segments
        """
        logger.info(f"Analyzing speech in {audio_path}")

        if self.backend == "silero":
            segments, audio_data = self.silero_vad.detect_speech(audio_path)
            logger.info(f"Detected {len(segments)} speech segments with Silero VAD")
            audio_data = audio_data.cpu().numpy() # float32 tensor -> numpy array
            audio_data = (audio_data * 32768).astype(np.int16)  # Convert back to int16
            return segments, audio_data

        # Load audio
        audio_data, sample_rate = self.load_audio(audio_path)
        logger.info(
            f"Loaded audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz"
        )

        # Detect speech frames
        if self.backend == "webrtc":
            # WebRTC VAD requires specific sample rates, resample if needed
            audio_data = resampy.resample(audio_data, sample_rate, 16000, filter='kaiser_fast', parallel=True).astype(np.int16)
            sample_rate = 16000
            # clean audio with noise gate if needed
            audio_torch = torch.from_numpy(audio_data).float().cuda() / 32768.0
            audio_torch = self.noise_gate(audio_torch.unsqueeze(0))  # Apply noise reduction if needed
            audio_data = (audio_torch.squeeze().cpu().numpy() * 32768).astype(np.int16)
            speech_frames = self.detect_speech_webrtc(audio_data, 16000)
        else:
            raise NotImplementedError(
                f"VAD backend '{self.backend}' not implemented yet"
            )
            speech_frames = self.detect_speech_energy(audio_data, sample_rate)

        # Merge into segments
        # segments = self.merge_segments(speech_frames)
        segments = [
            SpeechSegment(
                start, end, confidence=0.9 if self.backend == "webrtc" else 0.7
            )
            for start, end, is_speech in speech_frames
            if is_speech
        ]

        logger.info(f"Detected {len(segments)} speech frames")
        return segments, audio_data

    def split_into_utterances(
        self,
        segments: List[SpeechSegment],
        max_utterance_duration: float = 30.0,  # seconds
        min_pause_for_split: float = 0.5,  # seconds
    ) -> List[SpeechSegment]:
        """
        Split long speech segments into utterance-sized chunks.

        Args:
            segments: Speech segments
            max_utterance_duration: Maximum duration for an utterance
            min_pause_for_split: Minimum pause duration to consider a split point

        Returns:
            List of utterance-sized segments
        """
        utterances = []

        for segment in segments:
            duration = segment.duration()

            if duration <= max_utterance_duration:
                # Segment is already utterance-sized
                utterances.append(segment)
            else:
                # Split long segment
                # For now, use simple time-based splitting
                # In production, analyze pauses within the segment
                n_splits = int(np.ceil(duration / max_utterance_duration))
                chunk_duration = duration / n_splits

                for i in range(n_splits):
                    chunk_start = segment.start + i * chunk_duration
                    chunk_end = min(
                        segment.start + (i + 1) * chunk_duration, segment.end
                    )

                    utterances.append(
                        SpeechSegment(
                            start=chunk_start,
                            end=chunk_end,
                            confidence=segment.confidence
                            * 0.8,  # Lower confidence for splits
                        )
                    )

        return utterances

    def visualize_segments(self, segments: List[SpeechSegment], offset: float = 0.0, ax: Optional[plt.Axes] = None, show: bool = True, **kwargs):
        """Visualize speech segments (for debugging)."""
        if ax is None:
            plt.figure(figsize=(10, 2))
        elif ax:
            plt.sca(ax)
        for seg in segments:
            plt.axvspan(seg.start + offset, seg.end + offset, alpha=0.3, **kwargs)
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.title("Detected Speech Segments")
        if show:
            plt.show()
