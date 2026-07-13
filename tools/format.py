#!/usr/bin/env python3

import argparse
import shutil
import subprocess
from pathlib import Path

EXTENSIONS = {".cpp", ".hpp", ".h"}


def main():
    parser = argparse.ArgumentParser(
        description="Recursively run clang-format on C/C++ source files."
    )
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--style", default="file")
    args = parser.parse_args()

    clang_format = shutil.which("clang-format")
    if clang_format is None:
        print("Error: clang-format not found in PATH.")
        return 1

    root = Path(args.directory)

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not files:
        print("No source files found.")
        return 0

    print(f"Formatting {len(files)} files...")

    for f in files:
        print(f"  {f}")
        subprocess.run([clang_format, "-i", str(f)], check=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
