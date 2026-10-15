#!/usr/bin/env python3
"""
boox_to_anki.py — turn handwritten Boox notes (PDF or PNG) into Anki notes with Arabic audio.

Pipeline:
  0. Any PDFs are split into one PNG per page, with the empty margins cropped off.
  1. Each PNG is sent to Claude (vision) to read the English and Arabic text, and to locate any
     drawing (a bottle, a car...). The drawing is cropped out on its own for the Image field.
  2. Arabic audio is generated with Microsoft Edge's free neural voices (edge-tts).
  3. A note is added to Anki via AnkiConnect with English, Arabic, image and audio filled in.

Requirements
  - Anki desktop running, with the AnkiConnect add-on installed (code 2055492159).
  - pip install anthropic requests edge-tts pymupdf pillow
  - export ANTHROPIC_API_KEY=sk-ant-...

Usage
  python boox_to_anki.py notes.pdf                 # a single PDF export from the Boox
  python boox_to_anki.py ./exports                 # a folder of PDFs and/or PNGs
  python boox_to_anki.py ./exports --dry-run       # show what would be added, add nothing
  python boox_to_anki.py notes.pdf --pdf-only      # just convert PDF pages to PNGs, stop there
  python boox_to_anki.py ./exports --voice ar-EG-SalmaNeural
  python boox_to_anki.py ./exports --no-audio
  edge-tts --list-voices | grep ar-                # see all Arabic voices

PDF pages are saved next to the PDF in a folder named <pdf name>_pages/, as
<pdf name>_p01.png, <pdf name>_p02.png, ... so you can look at them afterwards.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import anthropic
import edge_tts
import pymupdf
import requests
from PIL import Image, ImageChops

# ---- Configure these to match your Anki setup --------------------------------
DECK_NAME = "Arabic Level 3"
NOTE_TYPE = "Sams Arabic Note Format"          # must already exist in Anki
FIELD_ENGLISH = "English"
FIELD_ARABIC = "Arabic"
FIELD_TRANSLIT = "Transliteration"    # "" to skip
TRANSLIT_STYLE = ("simple English-friendly spelling with long vowels doubled "
                  "(e.g. kitaab, zujaaja, maa'), and ' for hamza and 3 for ayn")
FIELD_IMAGE = "Image"                 # gets just the drawing, cropped out (empty if there is none)
FIELD_PAGE = ""                       # optional: field name for the full handwritten page, e.g. "Page"; "" to skip
FIELD_AUDIO = "Audio"
TAG = "boox-import"
DEFAULT_VOICE = "ar-SA-HamedNeural"   # Saudi, male. Try ar-SA-ZariyahNeural (female), ar-EG-SalmaNeural (Egyptian)
SPEECH_RATE = "-10%"                  # slightly slower than normal for learners; "+0%" for natural speed
MODEL = "claude-sonnet-5-5"
ANKICONNECT_URL = "http://127.0.0.1:8765"
PDF_DPI = 200                         # sharp enough for Anki without huge files
CROP_PADDING = 24                     # pixels of white space kept around the drawing
DRAWING_PADDING = 0.03                # widen the model's drawing box by 3% so edges aren't clipped
# ------------------------------------------------------------------------------

PROMPT = """This image is a handwritten flashcard with English and Arabic text.
Transcribe it exactly. Keep Arabic in Arabic script, including any harakat (vowel marks) that are written.
If there are several English/Arabic pairs, return each pair separately.

The page may also have a drawing that illustrates a word (for example a bottle or a car).
For each pair, if there is a drawing that belongs to it, give its bounding box as
[left, top, right, bottom], each a fraction of the image width or height from 0 to 1.
The box must contain only the drawing, not the handwritten words. If there is no drawing for that pair, use null.

Also give a transliteration of the Arabic in Latin letters, using this style: {translit_style}.
Base it on the standard pronunciation of the word, adding vowels even if they are not written.

Reply with JSON only, no other text:
{{"pairs": [{{"english": "...", "arabic": "...", "transliteration": "...", "drawing": [0.1, 0.2, 0.5, 0.6] or null}}]}}
If a side is unreadable, use an empty string for it."""

# Structured outputs: the API guarantees the reply matches this schema, so no fence-stripping needed.
PAIRS_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "english": {"type": "string"},
                    "arabic": {"type": "string"},
                    "transliteration": {"type": "string"},
                    "drawing": {"type": ["array", "null"], "items": {"type": "number"}},
                },
                "required": ["english", "arabic", "transliteration", "drawing"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


# ---- Step 0: PDF -> PNG ------------------------------------------------------

def crop_whitespace(img, padding=CROP_PADDING):
    """Trim the blank margins around the ink, keeping a little padding. Blank pages return None."""
    gray = img.convert("L")
    background = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, background).point(lambda p: 255 if p > 40 else 0)
    box = diff.getbbox()
    if not box:
        return None
    left, top, right, bottom = box
    return img.crop((
        max(left - padding, 0),
        max(top - padding, 0),
        min(right + padding, img.width),
        min(bottom + padding, img.height),
    ))


def pdf_to_pngs(pdf_path, dpi=PDF_DPI):
    """Render each page of the PDF to a cropped PNG. Returns the PNG paths in page order."""
    out_dir = pdf_path.with_name(f"{pdf_path.stem}_pages")
    out_dir.mkdir(exist_ok=True)
    pngs = []
    with pymupdf.open(pdf_path) as doc:
        width = len(str(doc.page_count))
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            cropped = crop_whitespace(img)
            if cropped is None:
                print(f"  [skip] {pdf_path.name} page {i}: blank")
                continue
            out = out_dir / f"{pdf_path.stem}_p{i:0{max(width, 2)}d}.png"
            cropped.save(out, optimize=True)
            pngs.append(out)
    print(f"{pdf_path.name}: {len(pngs)} page(s) -> {out_dir}")
    return pngs


def collect_pngs(source):
    """Accept a PDF, a PNG, or a folder of either; convert PDFs and return all PNGs to process."""
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".pdf", ".png"))
    else:
        sys.exit(f"Not found: {source}")

    pngs = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            pngs.extend(pdf_to_pngs(f))
        elif f.suffix.lower() == ".png":
            pngs.append(f)
    return pngs


# ---- Steps 1-3: read, speak, add to Anki -------------------------------------

def anki(action, **params):
    resp = requests.post(ANKICONNECT_URL, json={"action": action, "version": 6, "params": params})
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"AnkiConnect {action}: {data['error']}")
    return data["result"]


def extract_pairs(client, png_path):
    image_b64 = base64.standard_b64encode(png_path.read_bytes()).decode()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16000,                 # thinking counts toward this, so leave room
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": PROMPT.format(translit_style=TRANSLIT_STYLE)},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": PAIRS_SCHEMA}},
    )
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"model stopped early: {msg.stop_reason}")
    # The reply can start with a thinking block, so find the text block instead of using content[0].
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text).get("pairs", [])


def crop_drawing(png_path, box, index, out_dir):
    """Cut the drawing out of the page using the model's [left, top, right, bottom] fractions.
    Returns the PNG path, or None if the box is missing or invalid."""
    try:
        left, top, right, bottom = (min(max(float(v), 0.0), 1.0) for v in box)
    except (TypeError, ValueError):
        return None
    if right - left < 0.02 or bottom - top < 0.02:
        return None
    with Image.open(png_path) as img:
        w, h = img.size
        # The model's box is approximate, so widen it a little, then tighten onto the actual ink.
        pad = DRAWING_PADDING
        rough = img.crop((
            int(max(left - pad, 0) * w), int(max(top - pad, 0) * h),
            int(min(right + pad, 1) * w), int(min(bottom + pad, 1) * h),
        ))
        drawing = crop_whitespace(rough)
        if drawing is None:
            return None
        out = out_dir / f"{png_path.stem}_drawing{index + 1}.png"
        drawing.save(out, optimize=True)
        return out


def make_audio(arabic, voice, out_dir):
    """Generate an MP3 for the Arabic text; returns its path. Filename is a hash so re-runs don't clash."""
    digest = hashlib.sha1(f"{voice}|{SPEECH_RATE}|{arabic}".encode()).hexdigest()[:12]
    path = out_dir / f"boox_ar_{digest}.mp3"
    if not path.exists():
        asyncio.run(edge_tts.Communicate(arabic, voice, rate=SPEECH_RATE).save(str(path)))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="a PDF, a PNG, or a folder of PDFs/PNGs")
    ap.add_argument("--dry-run", action="store_true", help="read the images but add nothing to Anki")
    ap.add_argument("--pdf-only", action="store_true", help="only convert PDFs to PNGs, then stop")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    pngs = collect_pngs(args.source)
    if not pngs:
        sys.exit(f"No PDFs or PNGs found in {args.source}")
    if args.pdf_only:
        return

    if not args.dry_run:
        anki("createDeck", deck=DECK_NAME)  # no-op if it exists

    client = anthropic.Anthropic()
    audio_dir = Path(tempfile.mkdtemp(prefix="boox_audio_"))
    drawing_dir = Path(tempfile.mkdtemp(prefix="boox_drawings_"))
    added, skipped = 0, 0

    for png in pngs:
        try:
            pairs = extract_pairs(client, png)
        except Exception as e:
            print(f"[skip] {png.name}: could not read ({e})")
            skipped += 1
            continue

        for i, pair in enumerate(pairs):
            en, ar = pair.get("english", "").strip(), pair.get("arabic", "").strip()
            tr = (pair.get("transliteration") or "").strip()
            drawing = crop_drawing(png, pair.get("drawing"), i, drawing_dir) if pair.get("drawing") else None
            print(f"{png.name}  EN: {en!r}  AR: {ar!r}  TR: {tr!r}  drawing: {drawing.name if drawing else 'none'}")
            if args.dry_run:
                continue

            pictures = []
            if drawing:
                pictures.append({"path": str(drawing.resolve()), "filename": drawing.name, "fields": [FIELD_IMAGE]})
            if FIELD_PAGE:
                pictures.append({"path": str(png.resolve()), "filename": png.name, "fields": [FIELD_PAGE]})

            note = {
                "deckName": DECK_NAME,
                "modelName": NOTE_TYPE,
                "fields": {FIELD_ENGLISH: en, FIELD_ARABIC: ar},
                "tags": [TAG],
                "options": {"allowDuplicate": False},
            }
            if FIELD_TRANSLIT:
                note["fields"][FIELD_TRANSLIT] = tr
            if pictures:
                note["picture"] = pictures

            if ar and not args.no_audio:
                try:
                    mp3 = make_audio(ar, args.voice, audio_dir)
                    note["audio"] = [{
                        "path": str(mp3),
                        "filename": mp3.name,
                        "fields": [FIELD_AUDIO],
                    }]
                except Exception as e:
                    print(f"  [warn] audio failed, adding note without it: {e}")

            try:
                anki("addNote", note=note)
                added += 1
            except RuntimeError as e:
                print(f"  [skip] {e}")
                skipped += 1

    print(f"\nDone: {added} added, {skipped} skipped.")
    if added:
        anki("guiBrowse", query=f"tag:{TAG} added:1")  # show today's new cards for a quick check


if __name__ == "__main__":
    main()