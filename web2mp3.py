#!/usr/bin/env python3
"""Convert web articles to MP3 using the pdf2mp3 synthesis pipeline."""

import argparse
import importlib
import re
import sys
import tempfile
from pathlib import Path

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None


_PDF2MP3_MODULE = None


def get_pdf2mp3():
    global _PDF2MP3_MODULE
    if _PDF2MP3_MODULE is None:
        try:
            _PDF2MP3_MODULE = importlib.import_module("pdf2mp3")
        except ModuleNotFoundError as exc:  # pragma: no cover - surfaced at runtime
            raise RuntimeError(
                "Failed to import pdf2mp3. Install its dependencies (pydub, pyttsx3, pypdf, tqdm)."
            ) from exc
    return _PDF2MP3_MODULE


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def clean_text(text: str) -> str:
    pdf2mp3_mod = get_pdf2mp3()
    text = pdf2mp3_mod.normalize_for_tts(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def soup_text_and_title(html: str) -> tuple[str, str | None]:
    if BeautifulSoup is None:
        raise RuntimeError("Missing dependency 'beautifulsoup4'. Install it to parse HTML.")

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content_root = soup.find("article") or soup.find("main") or soup.body or soup

    # Drop common non-content sections inside the chosen root.
    for tag in content_root.find_all(["nav", "header", "footer", "aside" ]):
        tag.decompose()

    text = content_root.get_text("\n")
    return text, title


def extract_text_from_html(html: str) -> tuple[str, str | None]:
    base_text, base_title = soup_text_and_title(html)
    best_text, best_title = base_text, base_title

    if Document is not None:
        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            summary_text, summary_title = soup_text_and_title(summary_html)
            # Prefer readability result when it is reasonably close in length to fallback.
            if summary_text and len(summary_text) >= 0.6 * len(base_text):
                best_text = summary_text
                best_title = doc.short_title() or summary_title or base_title
        except Exception:
            pass

    return best_text, best_title


def fetch_html(url: str) -> str:
    if requests is None:
        raise RuntimeError("Missing dependency 'requests'. Install it to download web pages.")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_article_text(url: str) -> tuple[str, str | None]:
    html = fetch_html(url)
    raw_text, detected_title = extract_text_from_html(html)
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise ValueError("No readable text extracted from the page")
    return cleaned, detected_title


def synthesize_text(
    text: str,
    out_mp3: Path,
    chunk_size: int,
    rate: int,
    voice: str | None,
):
    pdf2mp3_mod = get_pdf2mp3()
    parts = pdf2mp3_mod.chunk_text(text, chunk_size=chunk_size)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wavs = []
        for i, part in enumerate(parts):
            wav_path = tmp / f"part_{i:03d}.wav"
            pdf2mp3_mod.tts_to_wav(part, wav_path, rate_wpm=rate, voice=voice)
            wavs.append(wav_path)
        pdf2mp3_mod.concat_wavs_to_mp3(wavs, out_mp3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a web page to MP3 audio")
    parser.add_argument("url", help="Web page URL to narrate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MP3 path (default: derived from page title in ./audio_output)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./audio_output"),
        help="Output directory when --output is omitted",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Chunk size in characters (default: 4000)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=180,
        help="TTS rate in words per minute (default: 180)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Alex",
        help="Voice substring to select in pyttsx3 (default: Alex)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the detected article title (used for default output naming)",
    )

    args = parser.parse_args()

    try:
        pdf2mp3_mod = get_pdf2mp3()
        pdf2mp3_mod.ensure_ffmpeg()
        text, detected_title = fetch_article_text(args.url)

        output_path = args.output
        if output_path is None:
            args.outdir.mkdir(parents=True, exist_ok=True)
            title = args.title or detected_title or "article"
            safe_name = pdf2mp3_mod.safe_filename(title)
            output_path = args.outdir / f"{safe_name}.mp3"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        synthesize_text(
            text,
            output_path,
            args.chunk_size,
            args.rate,
            args.voice,
        )
        title_print = args.title or detected_title or "article"
        print(f"[ok] Wrote {output_path} ({title_print})")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
