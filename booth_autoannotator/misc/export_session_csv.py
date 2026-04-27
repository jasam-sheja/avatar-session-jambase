"""
Given a directory containing the output JSON files from `booth_autoannotator analyze`, export a CSV file summarizing the sessions and events.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export session annotations to CSV.")
    parser.add_argument("input", type=Path, help="Directory containing annotation JSON files.")
    parser.add_argument("output", type=Path, help="Output CSV file path.")
    return parser.parse_args()


def export_start_end_times(path: Path) -> pd.DataFrame:
    records = []
    for json_file in path.glob("*-*-*/recording-*/*.json"):
        with open(json_file, "r") as f:
            doc = json.load(f)
            for session in doc["sessions"]:
                try:
                    record = {
                        # "video_path": json_file.relative_to(path).with_name(f"stream-{json_file.stem}.mp4"),
                        "subject_id": session['subject']['subject_id'],
                        "video_path": '/'.join(doc["video"]['file_path'].split("/")[-3:]),
                        "start_time": session['times']["enter_time"]['t'],
                        "end_time": session['times']["exit_time"]['t'],
                    }
                    records.append(record)
                    # 
                except KeyError as e:
                    print(f"Error processing session in {json_file}: {e}")
                    print(session['times']); exit()
    return pd.DataFrame(records)

def main():
    args = parse_args()
    df = export_start_end_times(args.input)
    df.to_csv(args.output, index=False)
    print(f"Exported session start/end times to {args.output}")

if __name__ == "__main__":
    main()