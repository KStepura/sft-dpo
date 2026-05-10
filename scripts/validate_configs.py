#!/usr/bin/env python3
"""Load every configs/*.json to catch syntax errors before a long GPU run."""
from pathlib import Path
import json
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs"
    bad = 0
    for path in sorted(cfg_dir.glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            print(f"OK {path.name}")
        except Exception as exc:
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)
            bad += 1
    if bad:
        raise SystemExit(1)
    print(f"All {len(list(cfg_dir.glob('*.json')))} JSON configs valid.")


if __name__ == "__main__":
    main()
