#!/usr/bin/python3
"""
Extract user requests from Copilot CLI chat logs.

Usage:
  python extract_user_requests.py computer1-CLI computer2-CLI
  python extract_user_requests.py computer1-CLI --combined all_requests.jsonl

Outputs:
  - By default, writes <input_dir>/user_requests.jsonl for each input directory.
  - With --combined, writes a single combined file (no per-directory files).

The input files can be .jsonl (one JSON object per line) or .json (array).
The output is newline-delimited JSON with fields: {"timestamp": "...", "text": "..."}.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def parse_timestamp(ts: str):
    """Return a sortable key for ISO timestamps; fall back to raw string."""
    try:
        return (0, datetime.fromisoformat(ts.replace("Z", "+00:00")))
    except Exception:
        return (1, ts)


def load_json_objects(path: Path) -> Iterable[Dict]:
    """Yield JSON objects from a .jsonl or .json file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line in {path}: {exc}", file=sys.stderr)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed file {path}: {exc}", file=sys.stderr)
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data


def iter_events(input_path: Path) -> Iterable[Dict]:
    """Iterate all events from a file or directory."""
    if input_path.is_dir():
        for file in sorted(input_path.glob("*.json*")):
            yield from load_json_objects(file)
    else:
        yield from load_json_objects(input_path)


def extract_requests(input_path: Path) -> List[Dict[str, str]]:
    """Extract user messages with timestamps, sorted chronologically."""
    requests: List[Dict[str, str]] = []
    for event in iter_events(input_path):
        if event.get("type") != "user.message":
            continue
        ts = event.get("timestamp")
        text = event.get("data", {}).get("content")
        if not ts or text is None:
            continue
        requests.append({"timestamp": ts, "text": text})
    requests.sort(key=lambda r: parse_timestamp(r["timestamp"]))
    return requests


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract user requests from Copilot CLI chat logs.")
    parser.add_argument("inputs", nargs="+", help="Input directories or files to process.")
    parser.add_argument("--combined", help="Write all requests from every input into this file.")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    if args.combined:
        combined: List[Dict[str, str]] = []
        for path in input_paths:
            combined.extend(extract_requests(path))
        combined.sort(key=lambda r: parse_timestamp(r["timestamp"]))
        out_path = Path(args.combined)
        write_jsonl(out_path, combined)
        print(f"Wrote {len(combined)} requests → {out_path}")
    else:
        for path in input_paths:
            requests = extract_requests(path)
            out_path = path / "user_requests.jsonl" if path.is_dir() else path.with_suffix(".user_requests.jsonl")
            write_jsonl(out_path, requests)
            print(f"Wrote {len(requests)} requests → {out_path}")


if __name__ == "__main__":
    main()
