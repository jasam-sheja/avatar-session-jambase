# Booth Auto-Annotator Installation Guide

## Quick Start

### 1. Install Core Dependencies

```bash
cd /mnt/HD-8.0TB/Moonshot/avatar-session-jambase
pip install -r booth_autoannotator/requirements.txt
```

### 2. Install Package (Development Mode)

```bash
# Basic installation
pip install -e .

# Or with the setup script
python setup_booth_autoannotator.py develop
```

### 3. Verify Installation

```bash
# Check if package can be imported
python -c "from booth_autoannotator import AnnotationPipeline; print('✓ Installation successful')"

# Check CLI
python -m booth_autoannotator.cli --help
```

## Optional Dependencies

### WebRTC VAD (Recommended for Production)

```bash
pip install webrtcvad
```

**Benefits**: More accurate VAD, better noise handling  
**Requirements**: Requires C compiler for installation

### Silero VAD (Best Accuracy)

```bash
pip install torch torchvision torchaudio
```

**Benefits**: Neural network-based, highest accuracy  
**Requirements**: Large dependency (~1GB), requires PyTorch

### Development Tools

```bash
pip install pytest black flake8 mypy
```

## Usage

### Python API

```python
from booth_autoannotator import AnnotationPipeline

pipeline = AnnotationPipeline(cache_dir="./cache")
doc = pipeline.analyze("video.mp4", "audio.wav")
doc.save("annotations.json")
```

### Command Line

```bash
# Basic usage
python -m booth_autoannotator.cli analyze video.mp4 audio.wav

# With options
python -m booth_autoannotator.cli analyze video.mp4 audio.wav \
    -o annotations.json \
    --vad-backend webrtc \
    --cache-dir ./cache
```

### Example Script

```bash
cd booth_autoannotator
python example.py
```

## Testing Installation

### Test with Sample Data

Create a simple test:

```python
# test_installation.py
import numpy as np
from booth_autoannotator import (
    AnnotationPipeline,
    DoorAngleEstimator,
    FaceDetector,
    VoiceActivityDetector,
)

print("Testing imports...")
print("✓ All modules imported successfully")

print("\nTesting instantiation...")
pipeline = AnnotationPipeline()
print("✓ Pipeline created")

door_est = DoorAngleEstimator()
print("✓ Door estimator created")

face_det = FaceDetector()
print("✓ Face detector created")

vad = VoiceActivityDetector()
print("✓ VAD created")

print("\n✅ All tests passed!")
```

Run it:
```bash
python test_installation.py
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the correct directory
cd /mnt/HD-8.0TB/Moonshot/avatar-session-jambase

# Try reinstalling
pip uninstall booth_autoannotator
pip install -e .
```

### OpenCV Issues

If OpenCV doesn't work:
```bash
# Try different OpenCV package
pip uninstall opencv-python
pip install opencv-python-headless
```

### WebRTC VAD Installation Fails

If webrtcvad won't install (needs C compiler):
```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev

# Then retry
pip install webrtcvad
```

Or use energy-based VAD (no dependencies):
```python
settings = PipelineSettings(vad_backend="energy")
pipeline = AnnotationPipeline(settings=settings)
```

### Memory Issues

For large videos:
```bash
# Reduce sampling rates
settings = PipelineSettings(
    door_sample_fps=3.0,
    face_sample_fps=3.0
)
```

## System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended for 4-hour videos
- **Storage**: ~1GB for cache per 4-hour video
- **CPU**: Multi-core recommended for faster processing

## Next Steps

1. ✅ Install package
2. ✅ Test with sample data
3. 📝 Calibrate thresholds for your booth setup
4. 🎨 Integrate with PyQt6 UI (if building annotation tool)
5. 📊 Process your first real video

## Documentation

- **Full documentation**: See `booth_autoannotator/README.md`
- **Implementation details**: See `booth_autoannotator/IMPLEMENTATION.md`
- **Examples**: See `booth_autoannotator/example.py`
- **API reference**: Use Python's `help()` function

```python
from booth_autoannotator import AnnotationPipeline
help(AnnotationPipeline)
```

## Support

For issues or questions, check:
1. Error messages and logs
2. README.md for usage patterns
3. IMPLEMENTATION.md for architecture details
4. Example scripts for working code

---

**Package Version**: 0.1.0  
**Last Updated**: 2026-02-16
