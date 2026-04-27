"""
Example script demonstrating booth_autoannotator usage.

This script shows how to use the auto-annotation pipeline
to process a booth session recording.
"""
import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from booth_autoannotator import AnnotationPipeline, PipelineSettings


def example_basic():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")
    
    # Set up paths (update these to your actual files)
    video_path = "path/to/session_recording.mp4"
    audio_path = "path/to/session_audio.wav"
    output_path = "annotations.json"
    
    # Create pipeline with default settings
    pipeline = AnnotationPipeline(cache_dir="./cache")
    
    # Progress callback
    def progress(stage, pct):
        print(f"  {stage}: {pct*100:.0f}%")
    
    # Run analysis
    print(f"Analyzing: {video_path}")
    doc = pipeline.analyze(video_path, audio_path, progress_callback=progress)
    
    # Save results
    doc.save(output_path)
    print(f"\nSaved to: {output_path}")
    
    # Print summary
    print(f"\nFound {len(doc.sessions)} session(s):")
    for session in doc.sessions:
        print(f"\n  Session {session.session_id}:")
        
        if session.times.enter_time:
            print(f"    Enter: {session.times.enter_time.t:.2f}s (conf: {session.times.enter_time.confidence:.2f})")
        
        if session.times.exit_time:
            print(f"    Exit: {session.times.exit_time.t:.2f}s (conf: {session.times.exit_time.confidence:.2f})")
        
        reg_segs = len(session.phases.registration.segments)
        print(f"    Registration segments: {reg_segs}")
        
        if session.phases.solo.start:
            print(f"    Solo speech: {session.phases.solo.start.t:.2f}s - {session.phases.solo.end.t:.2f}s")
        
        qa_utts = len(session.phases.qa.utterances)
        print(f"    QA utterances: {qa_utts}")


def example_custom_settings():
    """Example with custom settings."""
    print("\n=== Custom Settings Example ===\n")
    
    # Create custom settings
    settings = PipelineSettings(
        # Door detection
        door_closed_threshold=20.0,  # More lenient
        door_sample_fps=10.0,  # Higher sampling rate
        
        # Face detection
        face_confidence_threshold=0.6,  # Higher confidence
        
        # VAD
        vad_backend="webrtc",  # Use WebRTC (if installed)
        
        # Registration
        registration_motion_threshold=0.4,  # More sensitive
        registration_min_duration=3.0,  # Shorter minimum
    )
    
    print("Custom settings:")
    print(f"  Door threshold: {settings.door_closed_threshold}°")
    print(f"  Face confidence: {settings.face_confidence_threshold}")
    print(f"  VAD backend: {settings.vad_backend}")
    print(f"  Registration motion threshold: {settings.registration_motion_threshold}")
    
    # Use with pipeline
    pipeline = AnnotationPipeline(settings=settings, cache_dir="./cache")
    
    print("\nPipeline configured with custom settings.")


def example_access_data():
    """Example of accessing annotation data."""
    print("\n=== Data Access Example ===\n")
    
    from booth_autoannotator import AnnotationDocument
    
    # Load existing annotations
    doc = AnnotationDocument.load("annotations.json")
    
    print(f"Schema version: {doc.schema_version}")
    print(f"Video: {doc.video.file_path}")
    print(f"  Duration: {doc.video.duration_sec:.1f}s")
    print(f"  FPS: {doc.video.fps}")
    print(f"  Resolution: {doc.video.resolution}")
    
    print(f"\nAudio: {doc.audio.file_path}")
    print(f"  Duration: {doc.audio.duration_sec:.1f}s")
    
    print(f"\nSessions: {len(doc.sessions)}")
    
    # Access first session details
    if doc.sessions:
        session = doc.sessions[0]
        print(f"\nSession {session.session_id} details:")
        print(f"  Subject ID: {session.subject.subject_id}")
        print(f"  Validation status: {session.validation.status}")
        
        # Door events
        if session.events.door_closed:
            print(f"  Door closed at: {session.events.door_closed.t:.2f}s")
        if session.events.door_opened:
            print(f"  Door opened at: {session.events.door_opened.t:.2f}s")
        
        # Registration segments
        print(f"\n  Registration segments:")
        for i, seg in enumerate(session.phases.registration.segments):
            print(f"    Segment {i+1}: {seg.start.t:.2f}s - {seg.end.t:.2f}s")
            print(f"      Tags: {', '.join(seg.tags)}")
            print(f"      Confidence: {seg.start.confidence:.2f}")
        
        # QA utterances
        print(f"\n  QA utterances:")
        for i, utt in enumerate(session.phases.qa.utterances[:5]):  # Show first 5
            print(f"    Utterance {i+1}: {utt.start.t:.2f}s - {utt.end.t:.2f}s")
            print(f"      Label: {utt.label}")
            print(f"      Confidence: {utt.start.confidence:.2f}")
        
        if len(session.phases.qa.utterances) > 5:
            print(f"    ... and {len(session.phases.qa.utterances) - 5} more")


def main():
    """Run all examples."""
    print("Booth Auto-Annotator Examples")
    print("=" * 60)
    
    # Note: Update file paths before running
    print("\nNote: Update file paths in the script before running!\n")
    
    # Run examples
    try:
        # example_basic()  # Uncomment and update paths
        example_custom_settings()
        # example_access_data()  # Uncomment after running analysis
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please update the file paths in this script to point to your data.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
