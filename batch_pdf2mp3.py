#!/usr/bin/env python3
"""Batch convert all PDF files in a directory to MP3 using pdf2mp3.py."""

import argparse
import subprocess
import sys
from pathlib import Path


def build_pdf2mp3_command(pdf: Path, args, base_output: Path, script_path: Path) -> list[str]:
    cmd: list[str] = [sys.executable, str(script_path), str(pdf)]

    if args.chapters:
        outdir = base_output / pdf.stem
        cmd.extend(["--chapters", "--outdir", str(outdir)])
    else:
        base_output.mkdir(parents=True, exist_ok=True)
        output_mp3 = base_output / f"{pdf.stem}.mp3"
        cmd.extend(["-o", str(output_mp3)])

    if args.chunk_size is not None:
        cmd.extend(["--chunk-size", str(args.chunk_size)])
    if args.rate is not None:
        cmd.extend(["--rate", str(args.rate)])
    if args.voice:
        cmd.extend(["--voice", args.voice])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert every PDF in a directory to MP3 using pdf2mp3.py"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for generated audio (defaults to <directory>/audio_output)",
    )
    parser.add_argument(
        "--chapters",
        action="store_true",
        help="Use chapter mode for each PDF (creates one subfolder per PDF)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size to forward to pdf2mp3.py",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="Speech rate to forward to pdf2mp3.py",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice substring forwarded to pdf2mp3.py",
    )

    args = parser.parse_args()

    pdf_dir = args.directory.resolve()
    if not pdf_dir.is_dir():
        print(f"[error] Not a directory: {pdf_dir}", file=sys.stderr)
        return 1

    script_path = Path(__file__).resolve().parent / "pdf2mp3.py"
    if not script_path.exists():
        print(f"[error] pdf2mp3.py not found at {script_path}", file=sys.stderr)
        return 1

    pdf_files = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf" and p.is_file())
    if not pdf_files:
        print(f"[info] No PDF files found in {pdf_dir}")
        return 0

    base_output = (args.output_dir or (pdf_dir / "audio_output")).resolve()

    failures: list[tuple[Path, int]] = []
    for pdf in pdf_files:
        cmd = build_pdf2mp3_command(pdf, args, base_output, script_path)
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((pdf, result.returncode))

    if failures:
        print("[warn] Some conversions failed:")
        for pdf, code in failures:
            print(f"  - {pdf} (exit {code})")
        return 1

    print("[ok] Converted all PDFs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
