#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class Entry:
    timestamp: str
    text: str


def _format_timestamp(raw: str) -> str:
    """Return a human-readable UTC timestamp, falling back to the original string on parse errors."""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%d %H:%M:%S %Z")


def _load_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON in {path} at line {line_no}") from exc

        timestamp = _format_timestamp(str(obj.get("timestamp", "")))
        text = str(obj.get("text", ""))
        entries.append(Entry(timestamp=timestamp, text=text))
    return entries


def _render_html(entries: Iterable[Entry], title: str) -> str:
    styles = """
    :root { color-scheme: light; }
    body { font-family: "Inter", -apple-system, system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 24px; line-height: 1.6; }
    h1 { font-size: 1.3rem; margin-bottom: 12px; }
    .entries { display: flex; flex-direction: column; gap: 10px; }
    .entry { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
    .timestamp { font-size: 0.9rem; font-weight: 600; color: #64748b; margin-bottom: 6px; }
    .text { white-space: pre-wrap; word-break: break-word; }
    """

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        f"<title>{html.escape(title)}</title>",
        f"<style>{styles}</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class=\"entries\">",
    ]

    for entry in entries:
        text_html = html.escape(entry.text)
        html_lines.extend(
            [
                "<div class=\"entry\">",
                f"<div class=\"timestamp\">{html.escape(entry.timestamp)}</div>",
                f"<div class=\"text\">{text_html}</div>",
                "</div>",
            ]
        )

    html_lines.extend(["</div>", "</body>", "</html>"])
    return "\n".join(html_lines)


def convert_file(path: Path) -> Path:
    entries = _load_entries(path)
    output_path = path.with_suffix(".html")
    output_path.write_text(_render_html(entries, title=path.name), encoding="utf-8")
    return output_path


def main() -> None:
    root = Path(__file__).resolve().parent
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit("No .jsonl files found.")

    for file_path in jsonl_files:
        output = convert_file(file_path)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
