#!/usr/bin/env python3
"""
Extract user requests from VS Code Copilot chat logs (panel JSON format).

Usage:
  python extract_user_requests_vscode.py /path/to/log.json [more.json ...]

Output:
  - For each input file, writes <input>.user_requests.jsonl beside it.
  - Each line: {"timestamp": "<ISO-8601 UTC>", "text": "<user message>"}.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def parse_timestamp(ts: Union[int, float, str]) -> Tuple[int, str]:
    """Return (sort_key, iso_timestamp)."""
    if isinstance(ts, (int, float)):
        try:
            iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            return int(ts), iso
        except Exception:
            pass
    if isinstance(ts, str):
        try:
            # Handle ISO-like strings
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000), dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return sys.maxsize, ts
    return sys.maxsize, str(ts)


def extract_requests(path: Path) -> List[Dict[str, str]]:
    """Extract user request text and timestamp from a VS Code Copilot log."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Skipping malformed file {path}: {exc}", file=sys.stderr)
        return []

    requests = data.get("requests", [])
    results: List[Tuple[int, Dict[str, str]]] = []

    for req in requests:
        ts_raw: Optional[Union[int, float, str]] = req.get("timestamp") or req.get("createdAt")
        message = req.get("message", {})
        text = message.get("text")
        if ts_raw is None or text is None:
            continue
        sort_key, iso_ts = parse_timestamp(ts_raw)
        results.append((sort_key, {"timestamp": iso_ts, "text": text}))

    results.sort(key=lambda item: item[0])
    return [item[1] for item in results]


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract user requests from VS Code Copilot chat logs.")
    parser.add_argument("inputs", nargs="+", help="Input JSON log files.")
    args = parser.parse_args()

    for input_path_str in args.inputs:
        path = Path(input_path_str)
        if not path.is_file():
            print(f"Skipping non-file input: {path}", file=sys.stderr)
            continue

        requests = extract_requests(path)
        out_path = path.with_suffix(path.suffix + ".user_requests.jsonl")
        write_jsonl(out_path, requests)
        print(f"Wrote {len(requests)} requests → {out_path}")


if __name__ == "__main__":
    main()
