#!/usr/bin/env python3
"""
Command-line interface for booth auto-annotator.

Usage:
    booth-annotator analyze <recording> --output <dir> --cache-dir <dir> [options]
    booth-annotator precache <video> <audio> --cache-dir <dir>
    booth-annotator --help

python -m booth_autoannotator.cli analyze /mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/avatar-fix/2025-12-19/recording-000 \
    --output /mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/annotation/data-10/2025-12-19/recording-000 \
    --cache-dir /mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/annotation/cache/2025-12-19/recording-000

Dataset pre-caching:
for path in /media/alsherfawi/Crucial\ X10/JAMBASE-moonshot3/Avatar/avatar/????-??-??/{left,right}; do
    for stream in "000" "001" "002"; do
        video=$path/stream-$stream.mp4
        audio=$path/audio-$stream.wav
        if [[ -f "$video" && -f "$audio" ]]; then
            # echo "Pre-caching: $video"
            curdir=$(dirname "$path")
            cache_dir="/media/alsherfawi/Crucial X10/JAMBASE-moonshot3/Avatar/annotation/cache"/$(basename "$curdir")/$(basename "$path")
            python -m booth_autoannotator.cli precache "$video" "$audio" --cache-dir "$cache_dir"
        fi
    done
done

root=/mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/avatar-fix
for path in "${root}"/????-??-??/recording-???; do
    ts python -m booth_autoannotator.cli analyze ${path} \
    --output /mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/annotation/data-10/$(realpath --relative-to="$root" "$path") \
    --cache-dir /mnt/HD-8.0TB/Datasets/JAMBASE-moonshot3/Avatar/annotation/cache/$(realpath --relative-to="$root" "$path") \
    --door-threshold 10
done
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from matplotlib import pyplot as plt

from .models import AnnotationDocument
from .pipeline import AnnotationPipeline, PipelineSettings, export_json


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def progress_callback(stage: str, progress: float):
    """Simple progress callback for CLI."""
    bar_length = 40
    filled = int(bar_length * progress)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r{stage:20s} [{bar}] {progress*100:.1f}%", end="", flush=True)
    if progress >= 1.0:
        print()  # New line when complete


def cmd_analyze(args):
    """Run analysis command."""
    # Validate inputs
    path = Path(args.recording)
    if not path.exists():
        print(f"Error: Recording directory not found: {path}", file=sys.stderr)
        return exit(1)
    video_path = path / "stream-001.mp4"
    audio_path = path / "audio-001.wav"
    video_timestamps_path = path / "stream-001.timestamps.txt"
    audio_timestamps_path = path / "audio-001.timestamps.txt"

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    # Setup output path
    args.output.mkdir(parents=True, exist_ok=True)

    # Setup cache directory
    cache_dir = args.cache_dir

    # Create settings
    settings = PipelineSettings()

    # Override settings from args
    if args.door_threshold:
        settings.door_closed_threshold = args.door_threshold
    if args.face_threshold:
        settings.face_confidence_threshold = args.face_threshold
    if args.vad_backend:
        settings.vad_backend = args.vad_backend
    if args.energy_threshold:
        settings.vad_energy_threshold = args.energy_threshold

    # Create pipeline
    pipeline = AnnotationPipeline(settings=settings, cache_dir=cache_dir)

    # Run analysis
    print(f"Analyzing video: {video_path}")
    print(f"Audio file: {audio_path}")
    print(f"Output: {args.output}")
    if cache_dir:
        print(f"Cache directory: {cache_dir}")
    print()

    try:
        skip_analysis = False
        args.output.joinpath("001.json").touch(exist_ok=False)
    except FileExistsError:
        print(f"✓ Existing annotations found at {args.output}, loading...")
        skip_analysis = True

    if (
        not skip_analysis or args.debug
    ):
        print("✓ No existing annotations found, starting analysis...")
        try:
            docs = pipeline.analyze1(
                (path / "stream-001.mp4", path / "stream-002.mp4"),
                (path / "audio-001.wav", path / "audio-002.wav"),
                (
                    path / "stream-001.timestamps.txt",
                    path / "stream-002.timestamps.txt",
                ),
                (path / "audio-001.timestamps.txt", path / "audio-002.timestamps.txt"),
                plot=True, show=args.debug
            )
            if not args.debug:
                plt.savefig(args.output / "analysis_plots.pdf", bbox_inches="tight", dpi=300)

            # Export results
            for i, doc in enumerate(docs):
                output_path_i = args.output / f"{i+1:03d}.json"
                export_json(doc, str(output_path_i), indent=args.indent)

            print()
            print(f"✓ Analysis complete!")
            print(f"  Found {sum(len(d.sessions) for d in docs)} sessions")
            print(f"  Results saved to: {args.output}")
        except Exception as e:
            print(f"\n✗ Analysis failed: {e}", file=sys.stderr)
            if args.verbose or True:
                import traceback

                traceback.print_exc()
            return 1

    with open(args.output / "001.json", "r") as f:
        doc1 = AnnotationDocument.from_dict(json.load(f))
    # print(json.dumps(doc1.to_dict(), indent=2))

    with open(args.output / "002.json", "r") as f:
        doc2 = AnnotationDocument.from_dict(json.load(f))

    pipeline.analyze2(
        (path / "stream-001.mp4", path / "stream-002.mp4"),
        (path / "audio-001.wav", path / "audio-002.wav"),
        (
            path / "stream-001.timestamps.txt",
            path / "stream-002.timestamps.txt",
        ),
        (path / "audio-001.timestamps.txt", path / "audio-002.timestamps.txt"),
        doc1=doc1,
        doc2=doc2,
        plot=True, show=args.debug
    )

    print(f"✓ Existing annotations loaded.")


def cmd_precache(args):
    """Run pre-caching command."""
    # Validate inputs
    video_path = Path(args.video)
    audio_path = Path(args.audio)
    cache_dir = args.cache_dir

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    # Create settings
    settings = PipelineSettings()

    # Create pipeline
    pipeline = AnnotationPipeline(settings=settings, cache_dir=cache_dir)

    # Run pre-caching
    print(f"Pre-caching video: {video_path}")
    print(f"Audio file: {audio_path}")
    print(f"Cache directory: {cache_dir}")
    print()

    try:
        pipeline.precache(
            str(video_path),
            str(audio_path),
        )
    except Exception as e:
        print(f"\n✗ Pre-caching failed: {e}", file=sys.stderr)
        if args.verbose or True:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Booth Auto-Annotator: Automatic video annotation for booth sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video with default settings
  booth-annotator analyze video.mp4 audio.wav
  
  # Specify output file and cache directory
  booth-annotator analyze video.mp4 audio.wav -o annotations.json --cache-dir /tmp/cache
  
  # Use WebRTC VAD with custom thresholds
  booth-annotator analyze video.mp4 audio.wav --vad-backend webrtc --door-threshold 20
  
  # Disable caching
  booth-annotator analyze video.mp4 audio.wav --no-cache
        """,
    )

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze video and generate draft annotations"
    )

    analyze_parser.add_argument(
        "recording", help="Path to recording directory containing video and audio files"
    )

    analyze_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for JSON files",
    )

    analyze_parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory for caching intermediate results (default: <video_dir>/.cache)",
    )

    analyze_parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation (default: 2)"
    )

    # Analysis settings
    analyze_parser.add_argument(
        "--door-threshold",
        type=float,
        default=15.0,
        help="Door closed angle threshold in degrees (default: 15.0)",
    )

    analyze_parser.add_argument(
        "--face-threshold",
        type=float,
        help="Face detection confidence threshold (default: 0.5)",
    )

    analyze_parser.add_argument(
        "--vad-backend",
        choices=["energy", "webrtc", "silero"],
        help="VAD backend to use (default: energy)",
    )

    analyze_parser.add_argument(
        "--energy-threshold",
        type=float,
        help="Energy threshold for energy-based VAD (default: 0.01)",
    )

    # General options
    analyze_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    analyze_parser.add_argument(
        "--debug", action="store_true", help="Enable debug"
    )

    analyze_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    precache_parser = subparsers.add_parser(
        "precache",
        help="Pre-cache intermediate results for a video without full analysis",
    )

    precache_parser.add_argument("video", help="Path to input video file (MP4)")

    precache_parser.add_argument("audio", help="Path to input audio file (WAV)")

    precache_parser.add_argument(
        "--cache-dir", required=True, help="Directory for caching intermediate results"
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    # Run command
    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "precache":
        return cmd_precache(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
