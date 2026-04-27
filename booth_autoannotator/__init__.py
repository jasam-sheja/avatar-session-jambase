"""
Booth Auto-Annotator Package

Automatic annotation tools for booth video sessions.
Provides modules for:
- Door state detection via QR code
- Face presence detection
- Voice activity detection
- Registration phase detection
- Complete annotation pipeline

Usage:
    from booth_autoannotator import AnnotationPipeline
    
    pipeline = AnnotationPipeline()
    doc = pipeline.analyze(video_path, audio_path)
    doc.save("annotations.json")
"""

__version__ = "0.1.0"
__author__ = "Booth Annotator Team"

# Import main classes for easy access
from .models import (
    AnnotationDocument,
    AnnotationValue,
    Session,
    VideoMetadata,
    AudioMetadata,
)
from .pipeline import AnnotationPipeline, PipelineSettings, export_json
from .door_angle import DoorAngleEstimator, DoorState
from .face_detect import FaceDetector, FaceDetection
from .speech_vad import VoiceActivityDetector, SpeechSegment
from .registration_detect import RegistrationDetector, MotionMetrics

__all__ = [
    # Main API
    "AnnotationPipeline",
    "PipelineSettings",
    "export_json",
    
    # Data models
    "AnnotationDocument",
    "AnnotationValue",
    "Session",
    "VideoMetadata",
    "AudioMetadata",
    
    # Individual analyzers
    "DoorAngleEstimator",
    "DoorState",
    "FaceDetector",
    "FaceDetection",
    "VoiceActivityDetector",
    "SpeechSegment",
    "RegistrationDetector",
    "MotionMetrics",
]
