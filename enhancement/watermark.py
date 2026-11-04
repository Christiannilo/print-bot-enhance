"""Estampa de fecha naranja retro (DSEG14) — aprobada por Christian 04-nov-2026.

Formato DD-MM-AAAA, esquina derecha-abajo, tamaño ≈2.2% del ancho.
Fuente: /usr/share/fonts/truetype/dseg/DSEG14Classic-BoldItalic.ttf (OFL).
"""
import re
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DSEG_FONT = "/usr/share/fonts/truetype/dseg/DSEG14Classic-BoldItalic.ttf"
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
NARANJA = (255, 120, 30)
PCT = 0.022  # % del ancho de la foto

DATE_RE = re.compile(r"^(\d{1,2})[-/.\s](\d{1,2})[-/.\s](\d{4})$")


def normalize_date(text: str) -> str | None:
    """'4/11/2026', '04.11.2026' → '04-11-2026'. None si no parsea."""
    m = DATE_RE.match(text.strip())
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    if not (1 <= int(dd) <= 31 and 1 <= int(mm) <= 12):
        return None
    return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(DSEG_FONT, size)
    except OSError:
        return ImageFont.truetype(FALLBACK_FONT, size)


def stamp(src: str, date_str: str, dest: str | None = None) -> str:
    """Estampa date_str (DD-MM-AAAA) y guarda en dest (default: junto al src)."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    fs = max(10, round(w * PCT))
    f = _font(fs)
    draw = ImageDraw.Draw(img)
    margin = round(w * PCT / 2)
    draw.text((w - margin, h - margin), date_str, font=f, fill=NARANJA, anchor="rs")
    dest = dest or str(Path(src).with_name(Path(src).stem + "_stamped.jpg"))
    img.save(dest, quality=92)
    return dest


def has_exif_date(path: str) -> str | None:
    """Fecha EXIF como DD-MM-AAAA o None."""
    try:
        out = subprocess.run(
            ["identify", "-quiet", "-format", "%[EXIF:DateTime]", path],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if not out:
            return None
        m = re.match(r"(\d{4}):(\d{2}):(\d{2})", out)
        if m:
            yyyy, mm, dd = m.groups()
            return f"{dd}-{mm}-{yyyy}"
    except Exception:
        pass
    return None