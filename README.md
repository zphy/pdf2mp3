# pdf2mp3

Convert scientific PDFs into QuickTime-friendly MP3 audio using offline text-to-speech.

## Requirements
- Python 3.8 or newer (needed by recent `pypdf` releases)
- FFmpeg with `libmp3lame` on your PATH (used for MP3 export)
- Core Python packages: `pypdf`, `pyttsx3`, `pydub`, `tqdm`
- Extras for `web2mp3.py`: `requests`, `beautifulsoup4`, `readability-lxml` (optional but recommended for cleaner article parsing)

Install everything with:

```bash
python3 -m pip install pypdf pyttsx3 pydub tqdm requests beautifulsoup4 readability-lxml
```

On macOS you can install FFmpeg via Homebrew (`brew install ffmpeg`). On Linux use your package manager (`sudo apt install ffmpeg`), and on Windows `choco install ffmpeg` works well.

## Basic Usage

Render an entire PDF into a single MP3:

```bash
python pdf2mp3.py input.pdf -o output.mp3
```

If you omit `-o/--output`, the script writes the MP3 into `./audio_output/<pdf-stem>.mp3`.

## Chapter Mode

Split the narration into separate MP3 files using a simple heading detector:

```bash
python pdf2mp3.py input.pdf --chapters --outdir ./chapters
```

Each heading becomes a chapter file named after the detected title. The default output directory is `./audio_output`.

## Tuning Speech Output

- `--chunk-size`: adjust the number of characters per TTS request (default 4000)
- `--rate`: set the speech rate in words per minute (default 180)
- `--voice`: pick a pyttsx3 voice by substring (default `Alex` on macOS)

The script automatically normalizes mathematical symbols (e.g., bold/italic OMEGA) so pyttsx3 avoids reading font descriptors out loud.

## Websites → MP3

Use `web2mp3.py` to narrate an online article:

```bash
python web2mp3.py https://ewintang.com/blog/2025/04/22/open/ -o open.mp3
```

Omit `-o` to drop the MP3 in `./audio_output` named after the page title. Extra options mirror `pdf2mp3.py` (`--chunk-size`, `--rate`, `--voice`, `--title`), with `--voice` defaulting to `Alex`.
Install `readability-lxml` for cleaner article extraction; otherwise the script falls back to BeautifulSoup text scraping.

## Notes & Troubleshooting

- pyttsx3 renders to temporary WAV files; make sure the script can write to a temp directory.
- FFmpeg must support `libmp3lame`. If MP3 export fails, run `ffmpeg -encoders | grep lame` to verify support.
- If you hear “mathematical italic …” pronunciations, confirm you are running the latest version with Unicode normalization enabled.
