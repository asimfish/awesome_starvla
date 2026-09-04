#!/usr/bin/env python3
"""Assemble README.md = assets/readme_head.md + paper sections of assets/papers_curated.md + assets/readme_tail.md."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
head = (root / "assets" / "readme_head.md").read_text(encoding="utf-8")
tail = (root / "assets" / "readme_tail.md").read_text(encoding="utf-8")
curated = (root / "assets" / "papers_curated.md").read_text(encoding="utf-8")
start = curated.index("### [StarVLA Family](#content)")
body = curated[start:].rstrip() + "\n"
(root / "README.md").write_text(head + body + tail, encoding="utf-8")
print(f"README.md written: {len((head + body + tail).splitlines())} lines")
