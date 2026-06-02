from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a CMKC summary file.")
    parser.add_argument("--summary", required=True, help="Path to artifacts/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.summary, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
