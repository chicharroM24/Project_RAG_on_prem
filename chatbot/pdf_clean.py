# -*- coding: utf-8 -*-
"""
pdf_clean.py
Clean Portuguese (pt-PT) PDFs for RAG with auto OCR-per-page.
- Reads PDFs from /data/pdfs
- Extracts text via PyPDF; OCRs only pages that look empty/garbled
- Uses GPU for OCR if available
- Writes ONE cleaned TXT per PDF to /data/data_clean/<name>.txt
"""
import os
from pathlib import Path
from typing import List

import re
import unicodedata

# OCR stack (imports at top, as requested)
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr

from pypdf import PdfReader

# ---------------- Paths ---------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Proj_Chatbot_RAG
BASE_DATA = PROJECT_ROOT / "data"                    # força sempre ./data no host
IN_DIR = BASE_DATA / "pdfs"
OUT_DIR = BASE_DATA / "data_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[PATHS] IN_DIR={IN_DIR} | OUT_DIR={OUT_DIR}")  # debug

# ---------------- OCR render knobs ----------------
DPI = 300
CROP_TOP = 50
CROP_BOTTOM = 100
CROP_SIDE = 50

# ---------------- Cleaning knobs ----------------
HEADER_FOOTER_MAX_LINES = 0
MIN_LETTER_RATIO = 0.25
MIN_LINE_LEN = 5
# confidence threshold for EasyOCR results (0..1)
OCR_MIN_CONF = 0.40

# Section breakers to preserve spacing around
PRESERVE_BREAKERS = (
    r"\bCAP[IÍ]TULO\b",
    r"\bANEXO\b",
    r"\bNOTA PR[ÉE]VIA\b",
)

# Domain boilerplate to remove
BOILERPLATE_PATTERNS = [
    r"(?i)\bPDE\s*0-20-18\s*C[aã]es Militares\b",
    r"(?i)\bN[ÃA]O CLASSIFICADO\b",
    r"(?i)P[áa]gina intencionalmente em branco",
    r"(?i)\bCHEFE DO ESTADO\b",
    r"(?i)\bASSINATURA\b",
    r"(?i)\bQ6u\b",
    r"(?i)\bCRESPO\b",
    r"(?i)\bFONSECA\b",
    r"(?i)\bRyureyd\b",
]

# Lines to drop entirely (captions, junk)
DROP_WHOLE_LINE_PATTERNS = [
    r"(?im)^\s*(figura|fig\.|tabela|tab\.)\s+\S.*$",
]

# Noisy characters to strip
STRIP_CHARS_PATTERN = r"[|@{}\[\]<>#;=%*_~^€]"

# Optional: try to fix weird encodings without breaking accents
try:
    from ftfy import fix_text as _ftfy_fix
except Exception:
    _ftfy_fix = lambda x: x  # no-op if ftfy is not installed

# Private-Use Area bullets and odd glyphs often seen in PDFs
PUA_BULLETS = {
    "\uf0b7": "- ",  # 
    "\uf0a7": "- ",  # 
    "\uf0ad": "- ",  # 
    "\uf0d8": "- ",
    "\uf0d9": "- ",
    "\uf0da": "- ",
    "\uf06c": "- ",  # 
    "\uf0a8": "- ",  # 
}

def normalize_pua(text: str) -> str:
    for k, v in PUA_BULLETS.items():
        text = text.replace(k, v)
    return text

def reflow_paragraphs(text: str) -> str:
    """
    Reflow paragraphs by joining hard-wrapped lines while keeping lists and headings.
    Heuristics:
      - keep breaks before bullets/numbered lists/headings
      - if previous line doesn't end with sentence punctuation and next starts lowercase, join
      - also join letter-\n-letter (in-word breaks like 'a\npoio' -> 'apoio')
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = []
    for i, ln in enumerate(lines):
        if not ln:
            out.append(ln)
            continue
        # Look ahead
        nxt = lines[i+1] if i + 1 < len(lines) else ""
        # preserve explicit blank lines
        if nxt == "":
            out.append(ln)
            continue

        # signals to KEEP a break: bullets / numbered / headings / breakers
        keep_break = (
            re.match(r"^\s*([-•▪■–—]|\d+\)|\d+\.\d+|\(?[A-Z]\)|CAP[IÍ]TULO|ANEXO)\b", nxt, flags=re.IGNORECASE)
            or re.search(r"[.:;!?)]\s*$", ln)  # sentence-ish end
        )

        if keep_break:
            out.append(ln)
            continue

        # in-word newline: letter \n letter  -> glue without space
        if re.search(r"[A-Za-zÁÉÍÓÚáéíóúÃÕãõÇç]\s*$", ln) and re.match(r"^[a-záéíóúãõç]", nxt):
            # Join softly with a space, then fix 'a poio' -> 'apoio' when small split occurred
            joined = ln + " " + nxt
            joined = re.sub(r"(\b[a-záéíóúãõç])\s+([a-záéíóúãõç]{2,})\b", r"\1\2", joined, flags=re.IGNORECASE)
            out.append(joined)
            # skip nxt consumption by replacing next line with empty
            lines[i+1] = ""
        else:
            out.append(ln)

    # remove empty lines that were placeholders after join
    text = "\n".join([l for l in out if l is not None])
    # pass 2: remove single hard breaks inside paragraphs -> turn into space
    text = re.sub(r"(?<![.:;!?])\n(?=[a-záéíóúãõç])", " ", text)
    # collapse >2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def drop_annex_sections(text: str, aggressive: bool = True) -> str:
    """
    Drop annexes/appendices that are mostly forms/tables.
    Looks for 'ANEXO X' blocks until next chapter-like heading or end.
    """
    if not aggressive:
        return text
    pattern = r"(?is)\n\s*ANEXO\s+[A-Z0-9]+.*?(?=\n\s*(CAP[IÍ]TULO|APÊNDICE|REFERÊNCIAS|BIBLIOGRAFIA|FIM DO ANEXO)\b|$)"
    return re.sub(pattern, "\n", text)

def drop_repeated_headers_footers(pages: list[str], top_n: int = 2, bottom_n: int = 2) -> list[str]:
    """
    Identify lines that repeat across many pages (headers/footers) and remove them.
    More robust than per-page trimming.
    """
    from collections import Counter
    tops, bots = Counter(), Counter()
    cleaned_pages = []

    def norm(l):  # normalize numbers/dates so counters generalize
        l = re.sub(r"\b\d{1,4}[\-/\.]\d{1,4}[\-/\.]\d{1,4}\b", "", l)
        l = re.sub(r"\b\d+\b", "", l)
        return l.strip().lower()

    tmp = [p.split("\n") for p in pages]
    for ls in tmp:
        if ls:
            tops.update([norm(x) for x in ls[:top_n]])
            bots.update([norm(x) for x in ls[-bottom_n:]])

    # candidates repeated in >= 30% of pages
    thresh = max(2, int(0.3 * len(pages)))
    top_rep = {k for k, v in tops.items() if v >= thresh and len(k) >= 3}
    bot_rep = {k for k, v in bots.items() if v >= thresh and len(k) >= 3}

    for ls in tmp:
        head = [l for l in ls[:top_n] if norm(l) not in top_rep]
        mid = ls[top_n:len(ls)-bottom_n] if len(ls) > top_n + bottom_n else []
        tail = [l for l in ls[-bottom_n:] if norm(l) not in bot_rep]
        cleaned_pages.append("\n".join(head + mid + tail).strip())

    return cleaned_pages

# ------------- Helpers: quality heuristics -------------
def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return text.replace("\x00", " ")


def fix_common_ocr(text: str) -> str:
    text = _ftfy_fix(text)                    # fix mojibake if possible
    text = normalize_pua(text)                # map PUA bullets to "- "
    text = re.sub(r"-\s*\n\s*", "", text)     # "co-\nmunicação" -> "comunicação"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # collapse dot leaders
    text = re.sub(r"\.{3,}", " ", text)
    # bullets
    for ch in ["•", "▪", "■", ""]:
        text = text.replace(ch, "- ")
    # quotes/dashes
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("–", "-").replace("—", "-")
    # conservative 0->o inside words
    text = re.sub(r"(?<=[a-záéíóúç])0(?=[a-záéíóúç])", "o", text, flags=re.IGNORECASE)
    return text


def drop_headers_footers(page_text: str) -> str:
    lines = page_text.split("\n")
    if not lines:
        return page_text

    def is_breaker(s: str) -> bool:
        return any(re.search(p, s, flags=re.IGNORECASE) for p in PRESERVE_BREAKERS)

    # top
    removed = 0
    while removed < min(HEADER_FOOTER_MAX_LINES, len(lines)):
        if not lines or is_breaker(lines[0]):
            break
        lines.pop(0)
        removed += 1

    # bottom
    removed = 0
    while removed < min(HEADER_FOOTER_MAX_LINES, len(lines)):
        if not lines or is_breaker(lines[-1]):
            break
        lines.pop()
        removed += 1

    return "\n".join(lines)


def remove_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(STRIP_CHARS_PATTERN, "", text)
    text = re.sub(r"\b\d{1,2}-\d{1,2}\b", "", text)  # date-like "12-34"
    for pat in DROP_WHOLE_LINE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*[IVXLCDM]{1,4}\s*$", "", text)   
    return text


def preserve_structure(text: str) -> str:
    # ensure blank lines before breakers to help chunking
    for pat in PRESERVE_BREAKERS:
        text = re.sub(fr"(?={pat})", "\n\n", text, flags=re.IGNORECASE)
    # add a line break between numbered headings (e.g., "123. Title")
    text = re.sub(r"(\d{2,3}\.\s+[^\n]+)(?=\n\d{2,3}\.\s+)", r"\1\n", text)
    return text


def filter_low_signal_lines(text: str) -> str:
    clean_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) < MIN_LINE_LEN and not re.search(r"\w{3,}", line):
            continue
        letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÃÕãõÇç]", line)
        ratio = (len(letters) / len(line)) if line else 0.0
        if ratio < MIN_LETTER_RATIO:
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_ocr_text(text: str) -> str:
    text = normalize_unicode(text)
    text = fix_common_ocr(text)
    text = remove_boilerplate(text)
    text = preserve_structure(text)
    text = filter_low_signal_lines(text)
    return text


# ------------- Extraction PdfReader -------------
def extract_with_pypdf(pdf_path: Path) -> List[str]:
    pages = []
    reader = PdfReader(str(pdf_path))
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return pages



def page_needs_ocr(text: str) -> bool:
    s = (text or "").strip()
    # If almost empty, OCR
    if len(s) < 150:
        return True

    # Count words and letters
    words = len(re.findall(r"\w+", s, flags=re.UNICODE))
    letters = len(re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÃÕãõÇç]", s))

    # Heuristics tuned for hybrid PDFs (few overlay words + scanned body)
    if words < 40 or letters < 200 or len(s) < 800:
        return True

    # If the page looks like just 1-3 short lines (likely a header or ToC), OCR
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if len(lines) <= 3 and len(s) <= 180:
        return True

    # If over half of lines have < 10 letters, likely garbage; OCR
    if lines and sum(1 for ln in lines
                     if len(re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÃÕãõÇç]", ln)) < 10) / len(lines) > 0.5:
        return True

    return False


def ocr_page_image(img, reader) -> str:
    """Run EasyOCR on a single PIL image, crop & binarize, filter by confidence when available."""
    im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]
    y1, y2 = CROP_TOP, max(CROP_TOP, h - CROP_BOTTOM)
    x1, x2 = CROP_SIDE, max(CROP_SIDE, w - CROP_SIDE)
    im = im[y1:y2, x1:x2]

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    binimg = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Nota: com paragraph=True, EasyOCR pode devolver (bbox,text) *ou* (bbox,text,conf)
    results = reader.readtext(binimg, detail=1, paragraph=True)

    parts = []
    for item in results:
        if not item:
            continue
        if len(item) == 3:
            _, text, conf = item
            if text and (conf is None or conf >= OCR_MIN_CONF):
                parts.append(text.strip())
        elif len(item) == 2:
            _, text = item
            if text:
                # Sem confiança disponível; aceita o texto
                parts.append(text.strip())
        else:
            # formato inesperado — ignora
            continue

    return " ".join(parts).strip()


def ocr_selected_pages(pdf_path: Path, need_idx: List[int], use_gpu: bool) -> List[str]:
    """OCR only pages listed in need_idx. Returns list aligned with PDF length (others '')."""
    if not need_idx:
        return []
    images = convert_from_path(str(pdf_path), dpi=DPI)
    reader = easyocr.Reader(["pt"], gpu=use_gpu)
    texts = [""] * len(images)
    for i in need_idx:
        texts[i] = ocr_page_image(images[i], reader)
    return texts



def clean_pdf(pdf: Path) -> str:
    pypdf_pages = extract_with_pypdf(pdf)

    need_idx = [i for i, t in enumerate(pypdf_pages) if page_needs_ocr(t)]
    pages = pypdf_pages[:]

    if need_idx:
        use_gpu = True
        ocr_texts = ocr_selected_pages(pdf, need_idx, use_gpu=use_gpu)
        for i in need_idx:
            if i < len(ocr_texts) and ocr_texts[i]:
                pages[i] = ocr_texts[i]

    # --- Fallback: if text density is still too low, OCR the whole document ---
    avg_chars = sum(len(p) for p in pages) / max(1, len(pages))
    if avg_chars < 400:
        print(f"[FALLBACK] Low text density (avg {avg_chars:.1f} chars/page). Full-document OCR.")
        all_idx = list(range(len(pages)))
        ocr_all = ocr_selected_pages(pdf, all_idx, use_gpu=True)
        for i, t in enumerate(ocr_all):
            if t:
                pages[i] = t

    full_text = clean_ocr_text("\n\n".join(pages))
    return full_text

def is_up_to_date(pdf_path: Path, txt_path: Path) -> bool:
    """Return True if output TXT exists and is newer (or same mtime) than the PDF and non-empty."""
    if not txt_path.exists():
        return False
    try:
        if txt_path.stat().st_size == 0:
            return False
        return txt_path.stat().st_mtime >= pdf_path.stat().st_mtime
    except Exception:
        return False

# ------------- Main -------------
def main():
    pdfs = sorted(IN_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No PDFs found in {IN_DIR}")
        return

    # Optional override: set FORCE_ALL=1 to rebuild everything
    force_all = os.getenv("FORCE_ALL", "").strip().lower() in {"1", "true", "yes"}

    total = len(pdfs)
    done = 0
    skipped = 0

    for pdf in pdfs:
        out = OUT_DIR / f"{pdf.stem}.txt"

        if not force_all and is_up_to_date(pdf, out):
            print(f"[SKIP] {pdf.name} -> up-to-date ({out.name})")
            skipped += 1
            continue

        print(f"[CLEAN] {pdf.name} (auto OCR per page)")
        try:
            full_text = clean_pdf(pdf)
            out.write_text(full_text, encoding="utf-8")
            print(f"  -> {out}")
            done += 1
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print(f"Done. {done} processed, {skipped} skipped. Clean TXT saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
