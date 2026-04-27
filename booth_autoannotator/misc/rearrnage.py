'''
Rearrange the dataset to have a more consistent structure.

!note: the current structure has mismatch between video and audio files, left audio should be paired with right video and vice versa.

The current structure is:
- dataset/
  - avatar/
    - 2025-12-24/
      - left/
        - stream-000.mp4
        - stream-000.timestamps.txt
        - stream-001.mp4
        - stream-001.timestamps.txt
        - stream-002.mp4
        - stream-002.timestamps.txt
        - ...
      - right/
        - audio-000.wav
        - audio-000.timestamps.txt
        - audio-001.wav
        - audio-001.timestamps.txt
        - audio-002.wav
        - audio-002.timestamps.txt
        - ...

The desired structure is:
- dataset/
  - avatar/
    - 2025-12-24/
        - recording-000/
            - audio.001.wav # right audio
            - audio.001.timestamps.txt
            - video.001.mp4 # left video
            - video.001.timestamps.txt
            - audio.002.wav # left audio
            - audio.002.timestamps.txt
            - video.002.mp4 # right video
            - video.002.timestamps.txt
            - ...
        - recording-001/
            - audio.001.wav
            - audio.001.timestamps.txt
            - video.001.mp4
            - video.001.timestamps.txt
            - audio.002.wav
            - audio.002.timestamps.txt
            - video.002.mp4
            - video.002.timestamps.txt
            - ...
        ...

same with the cache

current:
- dataset
    - 2025-12-24/
        - left/
            - audio-000_speech_segments.pkl
            - audio-001_speech_segments.pkl
            - stream-000_door_states.pkl
            - stream-001_door_states.pkl
            - stream-000_face_detections.pkl
            - stream-001_face_detections.pkl
        - right/
            - audio-000_speech_segments.pkl
            - audio-001_speech_segments.pkl
            - stream-000_door_states.pkl
            - stream-001_door_states.pkl
            - stream-000_face_detections.pkl
            - stream-001_face_detections.pkl
desired:
- dataset
    - 2025-12-24/
        - recording-000/
            - audio.001_speech_segments.pkl
            - audio.002_speech_segments.pkl
            - video.001_door_states.pkl
            - video.002_door_states.pkl
            - video.001_face_detections.pkl
            - video.002_face_detections.pkl
        - recording-001/
            - audio.001_speech_segments.pkl # right audio
            - audio.002_speech_segments.pkl # left audio
            - video.001_door_states.pkl # left video
            - video.002_door_states.pkl # right video
            - video.001_face_detections.pkl # left video
            - video.002_face_detections.pkl # right video
        ...
'''

import os
from pathlib import Path
import shutil

from tqdm import tqdm

def rearrange_dataset(root_dir, output_dir=None, dry_run=True):

    root = Path(root_dir)
    if output_dir is None:
        output_dir = root.parent / (root.name + "_rearranged")
    else:
        output_dir = Path(output_dir)

    for date_dir in tqdm(root.glob("*/")):
        if not date_dir.is_dir():
            continue
        left_dir = date_dir / "left"
        right_dir = date_dir / "right"
        if not left_dir.is_dir() or not right_dir.is_dir():
            print(f"Skipping {date_dir} - missing left or right directory")
            continue
        
        # Get sorted lists of video and audio files
        video_files = list(zip(sorted(left_dir.glob("stream-*.mp4")), sorted(right_dir.glob("stream-*.mp4"))))
        audio_files = list(zip(sorted(right_dir.glob("audio-*.wav")), sorted(left_dir.glob("audio-*.wav"))))
        video_timestamps = list(zip(sorted(left_dir.glob("stream-*.timestamps.txt")), sorted(right_dir.glob("stream-*.timestamps.txt"))))
        audio_timestamps = list(zip(sorted(right_dir.glob("audio-*.timestamps.txt")), sorted(left_dir.glob("audio-*.timestamps.txt"))))

        if len(video_files) != len(audio_files):
            print(f"Warning: {date_dir} has {len(video_files)} videos but {len(audio_files)} audios")
            continue
        for i, ((left_video, right_video), (right_audio, left_audio), (left_video_timestamp, right_video_timestamp), (right_audio_timestamp, left_audio_timestamp)) in enumerate(zip(video_files, audio_files, video_timestamps, audio_timestamps)):
            
            if not left_video.is_file() or not right_video.is_file() or not right_audio.is_file() or not left_audio.is_file():
                print(f"Warning: missing files for recording {i} in {date_dir}")
                continue
            if ''.join(filter(str.isdigit, left_video.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, right_audio.with_suffix('').as_posix())):
                print(f"Warning: video {left_video} does not match audio {right_audio}")
                continue
            if ''.join(filter(str.isdigit, right_video.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, left_audio.with_suffix('').as_posix())):
                print(f"Warning: video {right_video} does not match audio {left_audio}")
                continue
            if ''.join(filter(str.isdigit, left_video_timestamp.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, left_video.with_suffix('').as_posix())):
                print(f"Warning: video timestamp {left_video_timestamp} does not match video {left_video}")
                continue
            if ''.join(filter(str.isdigit, right_video_timestamp.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, right_video.with_suffix('').as_posix())):
                print(f"Warning: video timestamp {right_video_timestamp} does not match video {right_video}")
                continue
            if ''.join(filter(str.isdigit, right_audio_timestamp.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, right_audio.with_suffix('').as_posix())):
                print(f"Warning: audio timestamp {right_audio_timestamp} does not match audio {right_audio}")
                continue
            if ''.join(filter(str.isdigit, left_audio_timestamp.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, left_audio.with_suffix('').as_posix())):
                print(f"Warning: audio timestamp {left_audio_timestamp} does not match audio {left_audio}")
                continue
            recording_dir = output_dir / date_dir.name / f"recording-{i:03d}"
            if not dry_run:
                recording_dir.mkdir(parents=True, exist_ok=True)
                # Copy files to new location
                if not (recording_dir / f"stream-001.mp4").exists():
                    shutil.copy(left_video, recording_dir / f"stream-001.mp4")
                if not (recording_dir / f"stream-002.mp4").exists():
                    shutil.copy(right_video, recording_dir / f"stream-002.mp4")
                if not (recording_dir / f"audio.001.wav").exists():
                    shutil.copy(right_audio, recording_dir / f"audio.001.wav")
                if not (recording_dir / f"audio.002.wav").exists():
                    shutil.copy(left_audio, recording_dir / f"audio.002.wav")
                if not (recording_dir / f"stream-001.timestamps.txt").exists():
                    shutil.copy(left_video_timestamp, recording_dir / f"stream-001.timestamps.txt")
                if not (recording_dir / f"stream-002.timestamps.txt").exists():
                    shutil.copy(right_video_timestamp, recording_dir / f"stream-002.timestamps.txt")
                if not (recording_dir / f"audio.001.timestamps.txt").exists():
                    shutil.copy(right_audio_timestamp, recording_dir / f"audio.001.timestamps.txt")
                if not (recording_dir / f"audio.002.timestamps.txt").exists():
                    shutil.copy(left_audio_timestamp, recording_dir / f"audio.002.timestamps.txt")
            else:
                print(f"Prepared recording-{i:03d} with {left_video} <==> {right_video}, {right_audio} <==> {left_audio}")

def rearrange_cache(root_dir, output_dir=None, dry_run=True):
    root = Path(root_dir)
    if output_dir is None:
        output_dir = root.parent / (root.name + "_rearranged")
    else:
        output_dir = Path(output_dir)

    for date_dir in tqdm(root.glob("*/")):
        if not date_dir.is_dir():
            continue
        # print(date_dir); exit()
        left_dir = date_dir / "left"
        right_dir = date_dir / "right"
        if not left_dir.is_dir() or not right_dir.is_dir():
            print(f"Skipping {date_dir} - missing left or right directory")
            continue
        # print(f"Processing cache for {date_dir}"); exit()
        
        # Get sorted lists of video and audio files
        audio_files = list(zip(sorted(right_dir.glob("audio-*.pkl")), sorted(left_dir.glob("audio-*.pkl"))))
        door_files = list(zip(sorted(left_dir.glob("stream-*_door_states.pkl")), sorted(right_dir.glob("stream-*_door_states.pkl"))))
        face_files = list(zip(sorted(left_dir.glob("stream-*_face_detections.pkl")), sorted(right_dir.glob("stream-*_face_detections.pkl"))))

        if len(audio_files) != len(door_files) or len(audio_files) != len(face_files):
            print(f"Warning: {date_dir} has {len(audio_files)} audio caches but {len(door_files)} door caches and {len(face_files)} face caches")
            continue
        # print(f"Processing cache for {date_dir} with {len(audio_files)} recordings"); exit()
        for i, ((right_audio, left_audio), (left_door, right_door), (left_face, right_face)) in enumerate(zip(audio_files, door_files, face_files)):
            
            if not right_audio.is_file() or not left_audio.is_file() or not left_door.is_file() or not right_door.is_file() or not left_face.is_file() or not right_face.is_file():
                print(f"Warning: missing cache files for recording {i} in {date_dir}")
                continue
            if ''.join(filter(str.isdigit, left_door.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, right_audio.with_suffix('').as_posix())):
                print(f"Warning: door cache {left_door} does not match audio cache {right_audio}")
                continue
            if ''.join(filter(str.isdigit, right_door.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, left_audio.with_suffix('').as_posix())):
                print(f"Warning: door cache {right_door} does not match audio cache {left_audio}")
                continue
            if ''.join(filter(str.isdigit, left_face.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, right_audio.with_suffix('').as_posix())):
                print(f"Warning: face cache {left_face} does not match audio cache {right_audio}")
                continue
            if ''.join(filter(str.isdigit, right_face.with_suffix('').as_posix())) != ''.join(filter(str.isdigit, left_audio.with_suffix('').as_posix())):
                print(f"Warning: face cache {right_face} does not match audio cache {left_audio}")
                continue
            recording_dir = output_dir / date_dir.name / f"recording-{i:03d}"
            # print(recording_dir); exit()
            if not dry_run:
                recording_dir.mkdir(parents=True, exist_ok=True)
                # Copy files to new location
                if not (recording_dir / f"audio-001_speech_segments.pkl").exists():
                    shutil.copy(right_audio, recording_dir / f"audio-001_speech_segments.pkl")
                if not (recording_dir / f"audio-002_speech_segments.pkl").exists():
                    shutil.copy(left_audio, recording_dir / f"audio-002_speech_segments.pkl")
                if not (recording_dir / f"video-001_door_states.pkl").exists():
                    shutil.copy(left_door, recording_dir / f"video-001_door_states.pkl")
                if not (recording_dir / f"video-002_door_states.pkl").exists():
                    shutil.copy(right_door, recording_dir / f"video-002_door_states.pkl")
                if not (recording_dir / f"video-001_face_detections.pkl").exists():
                    shutil.copy(left_face, recording_dir / f"video-001_face_detections.pkl")
                if not (recording_dir / f"video-002_face_detections.pkl").exists():
                    shutil.copy(right_face, recording_dir / f"video-002_face_detections.pkl")
            else:
                print(f"Prepared recording-{i:03d} with {left_door} <==> {right_audio}, {right_door} <==> {left_audio}, {left_face} <==> {right_audio}, {right_face} <==> {left_audio}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rearrange dataset structure for booth annotator")
    parser.add_argument("root_dir", help="Root directory of the dataset to rearrange")
    parser.add_argument("--output-dir", help="Output directory for rearranged dataset (default: <root_dir>_rearranged)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without making any changes")
    args = parser.parse_args()

    # rearrange_dataset(args.root_dir, args.output_dir, args.dry_run)
    rearrange_cache(args.root_dir, args.output_dir, args.dry_run)


