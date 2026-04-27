"""
Data models for booth annotation JSON schema v1.1.

These dataclasses represent the structured annotation format for video sessions.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import json


@dataclass
class AnnotationValue:
    """Base annotation with timestamp, source, confidence, and method."""

    t: float
    source: Literal["auto", "manual"] = "auto"
    confidence: float = 1.0
    method: str = ""

    def __post_init__(self):
        if not isinstance(self.t, (int, float)):
            try:
                self.t = float(self.t)
            except ValueError:
                raise ValueError(f"Invalid timestamp value: {self.t}")
            
        if not isinstance(self.confidence, (int, float)):
            try:
                self.confidence = float(self.confidence)
            except ValueError:
                raise ValueError(f"Invalid confidence value: {self.confidence}")
            
        if self.source not in ["auto", "manual"]:
            raise ValueError(f"Invalid source value: {self.source}. Must be 'auto' or 'manual'.")
        
        if not isinstance(self.method, str):
            raise ValueError(f"Invalid method value: {self.method}. Must be a string.")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence value must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class VideoMetadata:
    """Video file metadata."""

    file_path: str
    file_hash: str
    duration_sec: float
    fps: float
    resolution: List[int]  # [width, height]
    time_stamp: float = 0.0  # Optional timestamp for synchronization

    def __post_init__(self):
        if isinstance(self.file_path, Path):
            self.file_path = str(self.file_path.resolve())
        if len(self.resolution) != 2:
            raise ValueError("Resolution must be a list of [width, height]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self):
        return f"VideoMetadata(path={self.file_path}, hash={self.file_hash}, duration={self.duration_sec:.2f}s, fps={self.fps:.2f}, resolution={self.resolution})"


@dataclass
class AudioMetadata:
    """Audio file metadata."""

    file_path: str
    file_hash: str
    duration_sec: float
    offset_sec: float = 0.0

    def __post_init__(self):
        if isinstance(self.file_path, Path):
            self.file_path = str(self.file_path.resolve())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self):
        return f"AudioMetadata(path={self.file_path}, hash={self.file_hash}, duration={self.duration_sec:.2f}s, offset={self.offset_sec:.2f}s)"


@dataclass
class ToolInfo:
    """Tool and auto-package version information."""

    name: str = "BoothAnnotator"
    version: str = "0.2"
    auto_package: Dict[str, str] = field(
        default_factory=lambda: {"name": "booth_autoannotator", "version": "0.1"}
    )
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubjectInfo:
    """Subject identification."""

    subject_id: str = "[MANUAL_REQUIRED]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionEvents:
    """Session-level events (door states)."""

    door_closed: Optional[AnnotationValue] = None
    door_opened: Optional[AnnotationValue] = None

    def __post_init__(self):
        if isinstance(self.door_closed, dict):
            self.door_closed = AnnotationValue(**self.door_closed)
        if isinstance(self.door_opened, dict):
            self.door_opened = AnnotationValue(**self.door_opened)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.door_closed:
            result["door_closed"] = self.door_closed.to_dict()
        if self.door_opened:
            result["door_opened"] = self.door_opened.to_dict()
        return result


@dataclass
class SessionTimes:
    """Key session timestamps."""

    enter_time: Optional[AnnotationValue] = None
    exit_time: Optional[AnnotationValue] = None

    def __post_init__(self):
        if isinstance(self.enter_time, dict):
            self.enter_time = AnnotationValue(**self.enter_time)
        if isinstance(self.exit_time, dict):
            self.exit_time = AnnotationValue(**self.exit_time)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.enter_time:
            result["enter_time"] = self.enter_time.to_dict()
        if self.exit_time:
            result["exit_time"] = self.exit_time.to_dict()
        return result


@dataclass
class RegistrationSegment:
    """A single registration sub-segment."""

    start: AnnotationValue
    end: AnnotationValue
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "tags": self.tags,
        }


@dataclass
class RegistrationPhase:
    """Registration phase with multiple possible segments."""

    start: Optional[AnnotationValue] = None
    segments: List[RegistrationSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.start:
            result["start"] = self.start.to_dict()
        result["segments"] = [seg.to_dict() for seg in self.segments]
        return result


@dataclass
class SoloPhase:
    """Solo speech phase."""

    start: Optional[AnnotationValue] = None
    end: Optional[AnnotationValue] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.start:
            result["start"] = self.start.to_dict()
        if self.end:
            result["end"] = self.end.to_dict()
        return result


@dataclass
class QAUtterance:
    """A single Q&A utterance."""

    start: AnnotationValue
    end: AnnotationValue
    label: str = "TBD"  # "Q", "A", or "TBD"
    confidence_label: float = 0.0
    method_label: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "label": self.label,
            "confidence_label": self.confidence_label,
            "method_label": self.method_label,
        }


@dataclass
class QAPhase:
    """Q&A conversation phase with utterances."""

    utterances: List[QAUtterance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"utterances": [utt.to_dict() for utt in self.utterances]}


@dataclass
class SessionPhases:
    """All phases within a session."""

    registration: RegistrationPhase = field(default_factory=RegistrationPhase)
    solo: SoloPhase = field(default_factory=SoloPhase)
    qa: QAPhase = field(default_factory=QAPhase)

    def __post_init__(self):
        if isinstance(self.registration, dict):
            self.registration = RegistrationPhase(**self.registration)
        if isinstance(self.solo, dict):
            self.solo = SoloPhase(**self.solo)
        if isinstance(self.qa, dict):
            self.qa = QAPhase(**self.qa)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registration": self.registration.to_dict(),
            "solo": self.solo.to_dict(),
            "qa": self.qa.to_dict(),
        }


@dataclass
class SessionValidation:
    """Validation status for a session."""

    status: Literal["incomplete", "complete", "needs_review"] = "incomplete"
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    """Complete session annotation."""

    session_id: str
    pair_session_id: Optional[str] = None  # For linking sessions across videos
    subject: SubjectInfo = field(default_factory=SubjectInfo)
    events: SessionEvents = field(default_factory=SessionEvents)
    times: SessionTimes = field(default_factory=SessionTimes)
    phases: SessionPhases = field(default_factory=SessionPhases)
    validation: SessionValidation = field(default_factory=SessionValidation)
    freeform_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "pair_session_id": self.pair_session_id,
            "subject": self.subject.to_dict(),
            "events": self.events.to_dict(),
            "times": self.times.to_dict(),
            "phases": self.phases.to_dict(),
            "validation": self.validation.to_dict(),
            "freeform_notes": self.freeform_notes,
        }


@dataclass
class AnnotationDocument:
    """Top-level annotation document for a video."""

    schema_version: str = "1.1"
    video: Optional[VideoMetadata] = None
    audio: Optional[AudioMetadata] = None
    tool: ToolInfo = field(default_factory=ToolInfo)
    sessions: List[Session] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": self.schema_version,
            "tool": self.tool.to_dict(),
            "sessions": [session.to_dict() for session in self.sessions],
            "notes": self.notes,
        }
        if self.video:
            result["video"] = self.video.to_dict()
        if self.audio:
            result["audio"] = self.audio.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationDocument":
        """Create from dictionary (for loading JSON)."""
        # This is a simplified loader - in production, add full deserialization
        doc = cls()
        doc.schema_version = data.get("schema_version", "1.1")
        doc.notes = data.get("notes", "")

        # Load video metadata
        if "video" in data:
            v = data["video"]
            doc.video = VideoMetadata(**v)

        # Load audio metadata
        if "audio" in data:
            a = data["audio"]
            doc.audio = AudioMetadata(**a)

        # Load tool info
        if "tool" in data:
            doc.tool = ToolInfo(**data["tool"])

        # Load sessions (simplified - full implementation would recursively load all nested objects)
        doc.sessions = []
        for sess_data in data.get("sessions", []):
            session = Session(session_id=sess_data["session_id"],
                              pair_session_id=sess_data.get("pair_session_id"),
                              subject=SubjectInfo(**sess_data.get("subject", {})),
                              events=SessionEvents(**sess_data.get("events", {})),
                              times=SessionTimes(**sess_data.get("times", {})),
                              phases=SessionPhases(**sess_data.get("phases", {})),
                              validation=SessionValidation(**sess_data.get("validation", {})),
                              freeform_notes=sess_data.get("freeform_notes", "")
                              )

            # Add more detailed loading as needed
            doc.sessions.append(session)

        return doc

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, output_path: str, indent: int = 2):
        """Save to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load(cls, input_path: str) -> "AnnotationDocument":
        """Load from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationDocument":
        """Create from dictionary (for loading JSON)."""
        # This is a simplified loader - in production, add full deserialization
        doc = cls()
        doc.schema_version = data.get("schema_version", "1.1")
        doc.notes = data.get("notes", "")

        # Load video metadata
        if "video" in data:
            v = data["video"]
            doc.video = VideoMetadata(**v)

        # Load audio metadata
        if "audio" in data:
            a = data["audio"]
            doc.audio = AudioMetadata(**a)

        # Load tool info
        if "tool" in data:
            doc.tool = ToolInfo(**data["tool"])

        # Load sessions (simplified - full implementation would recursively load all nested objects)
        doc.sessions = []
        for sess_data in data.get("sessions", []):
            session = Session(session_id=sess_data["session_id"],
                              pair_session_id=sess_data.get("pair_session_id"),
                              subject=SubjectInfo(**sess_data.get("subject", {})),
                              events=SessionEvents(**sess_data.get("events", {})),
                              times=SessionTimes(**sess_data.get("times", {})),
                              phases=SessionPhases(**sess_data.get("phases", {})),
                              validation=SessionValidation(**sess_data.get("validation", {})),
                              freeform_notes=sess_data.get("freeform_notes", "")
                              )
            # Add more detailed loading as needed
            doc.sessions.append(session)

        return doc
