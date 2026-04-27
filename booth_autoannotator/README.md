# Booth Auto-Annotator

Automatic annotation package for booth video sessions. This package provides automated first-pass annotations for long recordings where multiple subjects sequentially enter a booth, complete structured sessions, and leave.

## Features

- **Door State Detection**: QR code-based door angle estimation to detect when subjects enter/exit
- **Face Detection**: Robust face presence detection with confidence scoring
- **Voice Activity Detection**: Speech segmentation from audio with multiple VAD backends
- **Registration Detection**: Head motion and blink analysis to identify registration phases
- **Complete Pipeline**: Orchestrated analysis with caching support for efficient re-processing

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced VAD performance:

```bash
# WebRTC VAD (recommended for production)
pip install webrtcvad

# Silero VAD (neural network-based)
pip install torch
```

## Usage

### Python API

```python
from booth_autoannotator import AnnotationPipeline

# Create pipeline with default settings
pipeline = AnnotationPipeline(cache_dir="./cache")

# Analyze video and audio
doc = pipeline.analyze(
    video_path="session_recording.mp4",
    audio_path="session_audio.wav"
)

# Save annotations
doc.save("annotations.json")

# Access sessions
for session in doc.sessions:
    print(f"Session {session.session_id}")
    print(f"  Enter time: {session.times.enter_time.t if session.times.enter_time else 'N/A'}")
    print(f"  Registration segments: {len(session.phases.registration.segments)}")
    print(f"  QA utterances: {len(session.phases.qa.utterances)}")
```

### Command-Line Interface

```bash
# Basic usage
python -m booth_autoannotator.cli analyze video.mp4 audio.wav

# With custom output and settings
python -m booth_autoannotator.cli analyze video.mp4 audio.wav \
    -o annotations.json \
    --vad-backend webrtc \
    --door-threshold 20 \
    --cache-dir ./cache

# Disable caching
python -m booth_autoannotator.cli analyze video.mp4 audio.wav --no-cache

# Verbose mode
python -m booth_autoannotator.cli analyze video.mp4 audio.wav -v
```

### Custom Settings

```python
from booth_autoannotator import AnnotationPipeline, PipelineSettings

# Create custom settings
settings = PipelineSettings(
    door_closed_threshold=20.0,  # More lenient door detection
    face_confidence_threshold=0.6,  # Higher confidence required
    vad_backend="webrtc",  # Use WebRTC VAD
    registration_motion_threshold=0.4,  # Lower motion threshold
)

# Use with pipeline
pipeline = AnnotationPipeline(settings=settings)
doc = pipeline.analyze(video_path, audio_path)
```

## Architecture

### Modules

- **models.py**: Data models for JSON annotation schema (v1.1)
- **door_angle.py**: QR-based door angle estimation
- **face_detect.py**: Face detection with confidence scoring
- **speech_vad.py**: Voice activity detection (energy/WebRTC/Silero)
- **registration_detect.py**: Registration phase detection via motion analysis
- **pipeline.py**: Main orchestrator with caching
- **cli.py**: Command-line interface

### Annotation Schema

The package produces JSON files following schema version 1.1:

```json
{
  "schema_version": "1.1",
  "video": { "file_path": "...", "fps": 30.0, ... },
  "audio": { "file_path": "...", "duration_sec": 14400.0, ... },
  "sessions": [
    {
      "session_id": "S001",
      "subject": { "subject_id": "[MANUAL_REQUIRED]" },
      "times": {
        "enter_time": { "t": 123.45, "source": "auto", "confidence": 0.92 }
      },
      "phases": {
        "registration": { "segments": [...] },
        "solo": { "start": {...}, "end": {...} },
        "qa": { "utterances": [...] }
      }
    }
  ]
}
```

### Caching

The pipeline caches intermediate results to speed up re-analysis:

- Door state timeline
- Face detection timeline
- Speech segments
- Motion metrics per session

Cache files are stored as pickle files in the specified cache directory.

## Performance Considerations

- **Sampling Rate**: Analysis runs at reduced FPS (default 5 fps) for efficiency
- **Caching**: Enable caching for large videos to avoid re-computation
- **VAD Backend**: 
  - `energy`: Fastest, no dependencies, good for clean audio
  - `webrtc`: Good balance of speed and accuracy
  - `silero`: Best accuracy, requires PyTorch, slower

## Configuration

### Door Detection

- `door_closed_threshold`: Angle threshold for closed state (default: 15°)
- `door_open_threshold`: Angle threshold for open state (default: 45°)
- `door_sample_fps`: Sampling rate for door analysis (default: 5 fps)

### Face Detection

- `face_confidence_threshold`: Minimum confidence for valid detection (default: 0.5)
- `face_sample_fps`: Sampling rate for face analysis (default: 5 fps)

### Speech VAD

- `vad_backend`: VAD algorithm ("energy", "webrtc", "silero")
- `vad_energy_threshold`: Energy threshold (default: 0.01)
- `min_speech_duration_ms`: Minimum speech segment duration (default: 250 ms)

### Registration Detection

- `registration_motion_threshold`: Motion energy threshold (default: 0.5)
- `registration_min_duration`: Minimum segment duration (default: 5 s)
- `registration_max_duration`: Maximum segment duration (default: 20 s)

## Requirements

- **Python**: 3.7+
- **Video Format**: MP4 (H.264)
- **Audio Format**: WAV (PCM)
- **Video Specs**: 30 fps, 1920x1080 (configurable)
- **QR Code**: Must be visible on door for state detection

## Limitations

- Subject ID must be manually entered (not detected automatically)
- Q/A labeling defaults to "TBD" (requires manual review)
- QR-based door detection requires proper QR code mounting and visibility
- Face landmark detection uses simplified heuristics (consider MediaPipe for production)
- Registration detection based on optical flow (may need calibration per setup)

## Future Enhancements

- MediaPipe integration for better landmark detection
- Deep learning-based registration detection
- Automatic Q/A classification using audio features
- Multi-camera support
- Real-time processing support

## License

See LICENSE file in parent project.

## Authors

Booth Annotator Team - 2026
