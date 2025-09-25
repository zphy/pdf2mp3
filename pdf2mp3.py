#!/usr/bin/env python3
"""
pdf2mp3.py — Convert a PDF to QuickTime-compatible MP3.

- Extracts text from a PDF
- Splits into manageable chunks for TTS
- Renders WAV with offline TTS (pyttsx3)
- Encodes MP3 with libmp3lame via pydub/ffmpeg
- Optional chapter-sized MP3s by simple heading detection

Usage:
  python pdf2mp3.py input.pdf -o output.mp3

More examples:
  # One continuous MP3
  python pdf2mp3.py qip2026-paper457.pdf -o qip2026.mp3

  # Chapter-sized MP3s into a folder
  python pdf2mp3.py qip2026-paper457.pdf --chapters --outdir ./chapters

  # Adjust chunk size (characters) and TTS rate (words/min)
  python pdf2mp3.py input.pdf -o out.mp3 --chunk-size 3000 --rate 180
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

from pydub import AudioSegment
from pypdf import PdfReader
import pyttsx3
from tqdm import tqdm


_MATH_CHAR_REPLACEMENTS = {
    "\u27E8": "bra",  # left angle bracket -> CJK left angle bracket
    "\u27E9": "ket",  # right angle bracket -> CJK right angle bracket
    "\u27EA": "\u300A",  # left double angle bracket
    "\u27EB": "\u300B",  # right double angle bracket
}


_GREEK_LETTERS = [
    "ALPHA",
    "BETA",
    "GAMMA",
    "DELTA",
    "EPSILON",
    "ZETA",
    "ETA",
    "THETA",
    "IOTA",
    "KAPPA",
    "LAMBDA",
    "MU",
    "NU",
    "XI",
    "OMICRON",
    "PI",
    "RHO",
    "SIGMA",
    "TAU",
    "UPSILON",
    "PHI",
    "CHI",
    "PSI",
    "OMEGA",
]


_UNICODE_NAME_OVERRIDES = {
    **{f"GREEK SMALL LETTER {name}": name.lower() for name in _GREEK_LETTERS},
    **{f"GREEK CAPITAL LETTER {name}": name.capitalize() for name in _GREEK_LETTERS},
    "GREEK SMALL LETTER FINAL SIGMA": "sigma",
    "GREEK SMALL LETTER LAMDA": "lambda",
    "GREEK CAPITAL LETTER LAMDA": "Lambda",
}


def normalize_for_tts(text: str) -> str:
    """Simplify math styled characters so TTS skips font qualifiers."""
    if not text:
        return text

    normalized = unicodedata.normalize("NFKC", text)
    simplified = []

    for ch in normalized:
        if ch in _MATH_CHAR_REPLACEMENTS:
            simplified.append(_MATH_CHAR_REPLACEMENTS[ch])
            continue

        name = unicodedata.name(ch, "")
        if name in _UNICODE_NAME_OVERRIDES:
            simplified.append(_UNICODE_NAME_OVERRIDES[name])
            continue
        if name.startswith("MATHEMATICAL "):
            base_name = name[len("MATHEMATICAL "):].strip()
            try:
                replacement = unicodedata.lookup(base_name)
            except KeyError:
                replacement = None
            if replacement and replacement != ch:
                simplified.append(replacement)
                continue

        simplified.append(ch)

    return "".join(simplified)


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            print(f"[warn] Failed to extract text from page {i+1}: {e}", file=sys.stderr)
            t = ""
        texts.append(t)
    text = "\n".join(texts)
    # Simple whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return normalize_for_tts(text)


def naive_chapter_split(text: str) -> list[tuple[str, str]]:
    """
    Split text into chapters based on naive heading detection.
    Returns list of (title, text_chunk).
    """
    # Detect headings like "Introduction", "Main Result", etc.
    pattern = re.compile(
        r"(?im)^(?:\s*)([A-Z][A-Za-z0-9 \-\(\)\/]{3,40})(?:\s*[\.:])?\s*$"
    )

    chapters = []
    last_idx = 0
    last_title = "Chapter 1"

    matches = list(pattern.finditer(text))
    if not matches:
        return [(last_title, text)]

    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        if idx == 0:
            # preface text before first heading
            preface = text[:start].strip()
            if preface:
                chapters.append(("Preface", preface))
        else:
            # content between previous heading and this heading
            chapters.append((last_title, text[last_idx:start].strip()))
        last_idx = start
        last_title = title

    # tail
    chapters.append((last_title, text[last_idx:].strip()))
    # remove empties
    chapters = [(t, c) for (t, c) in chapters if c]
    # uniquify titles
    seen = {}
    uniq = []
    for t, c in chapters:
        n = seen.get(t, 0)
        if n:
            nt = f"{t} ({n+1})"
            seen[t] = n + 1
            uniq.append((nt, c))
        else:
            seen[t] = 1
            uniq.append((t, c))
    return uniq


def chunk_text(text: str, chunk_size: int = 4000) -> list[str]:
    """
    Chunk text into ~chunk_size characters, preferably at sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    # Split into sentences and accumulate
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    chunks = []
    buf = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 > chunk_size and buf:
            chunks.append(" ".join(buf).strip())
            buf = [s]
            cur_len = len(s)
        else:
            buf.append(s)
            cur_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


def tts_to_wav(text: str, wav_path: Path, rate_wpm: int = 180, voice: str | None = None):
    """
    Render text to a WAV file using offline pyttsx3.
    """
    engine = pyttsx3.init()
    # Set speaking rate
    try:
        engine.setProperty("rate", rate_wpm)
    except Exception:
        pass

    # Optional: choose a voice by substring match
    if voice:
        try:
            voices = engine.getProperty("voices")
            for v in voices:
                if voice.lower() in (v.id.lower(), getattr(v, "name", "").lower()):
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass

    # Render
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()


def concat_wavs_to_mp3(wav_paths: list[Path], mp3_path: Path, sample_rate=44100, bitrate="128k"):
    """
    Concatenate WAVs and export MP3 using libmp3lame for QuickTime compatibility.
    """
    combined = None
    for wp in wav_paths:
        # Use format auto-detection; pyttsx3 may write AIFF on macOS
        seg = AudioSegment.from_file(str(wp))
        seg = seg.set_frame_rate(sample_rate).set_channels(2)
        combined = seg if combined is None else combined + seg

    # Ensure parent exists
    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    combined.export(
        str(mp3_path),
        format="mp3",
        bitrate=bitrate,
        parameters=[],
        codec="libmp3lame",
    )


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]", "_", name).strip()
    return re.sub(r"\s+", "_", name)[:80] or "chapter"


def build_continuous(pdf_path: Path, out_mp3: Path, chunk_size: int, rate: int, voice: str | None):
    text = extract_text_from_pdf(pdf_path)
    parts = chunk_text(text, chunk_size=chunk_size)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wavs = []
        for i, part in enumerate(tqdm(parts, desc="TTS (chunks)")):
            w = tmp / f"part_{i:03d}.wav"
            tts_to_wav(part, w, rate_wpm=rate, voice=voice)
            wavs.append(w)
        concat_wavs_to_mp3(wavs, out_mp3)
    print(f"[ok] Wrote {out_mp3}")


def build_chapters(pdf_path: Path, outdir: Path, chunk_size: int, rate: int, voice: str | None):
    text = extract_text_from_pdf(pdf_path)
    chapters = naive_chapter_split(text)
    outdir.mkdir(parents=True, exist_ok=True)

    for title, body in chapters:
        title_safe = safe_filename(title)
        chapter_mp3 = outdir / f"{title_safe}.mp3"
        parts = chunk_text(body, chunk_size=chunk_size)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            wavs = []
            for i, part in enumerate(tqdm(parts, desc=f"TTS ({title_safe})", leave=False)):
                w = tmp / f"{title_safe}_{i:03d}.wav"
                tts_to_wav(part, w, rate_wpm=rate, voice=voice)
                wavs.append(w)
            concat_wavs_to_mp3(wavs, chapter_mp3)
        print(f"[ok] Chapter: {title} -> {chapter_mp3}")


def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        print(
            "[error] FFmpeg not found. Install it and ensure it's on your PATH.\n"
            "  • macOS: brew install ffmpeg\n"
            "  • Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  • Windows (choco): choco install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Convert a PDF to QuickTime-compatible MP3.")
    # Positional PDF is optional to allow --qip mode
    ap.add_argument("pdf", type=Path, nargs="?", default=None, help="Input PDF path")
    ap.add_argument("-o", "--output", type=Path, help="Output MP3 path (for continuous mode)")
    ap.add_argument("--outdir", type=Path, default=Path("./audio_output"), help="Output folder")
    ap.add_argument("--chapters", action="store_true", help="Export chapter-sized MP3s instead")
    ap.add_argument("--chunk-size", type=int, default=4000, help="Chunk size in characters (default: 4000)")
    ap.add_argument("--rate", type=int, default=180, help="TTS rate (words per minute, default: 180)")
    ap.add_argument("--voice", type=str, default='Alex', help="Voice match substring (optional)")
    args = ap.parse_args()

    if args.pdf is None:
        print("[error] Missing input PDF. Provide a PDF path or use --qip <N>.", file=sys.stderr)
        sys.exit(1)

    if not args.pdf.exists():
        print(f"[error] File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    ensure_ffmpeg()

    if args.chapters:
        build_chapters(args.pdf, args.outdir, args.chunk_size, args.rate, args.voice)
    else:
        if not args.output:
            # Default to same stem with .mp3 in outdir
            args.outdir.mkdir(parents=True, exist_ok=True)
            args.output = args.outdir / (args.pdf.stem + ".mp3")
        build_continuous(args.pdf, args.output, args.chunk_size, args.rate, args.voice)


if __name__ == "__main__":
    main()
