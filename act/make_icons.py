"""Build the RVEStressStats icons from the MatViz3D logo with an "ACT" badge.

Usage:
    python act/make_icons.py [path/to/MatViz3D.png]

Writes act/RVEStressStats/images/:
    rve_ext.png  128x128  extension icon (Extension Manager)
    rve.png       16x16   toolbar buttons and tree objects
    rve32.png     32x32   high-DPI variant / preview
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "RVEStressStats" / "images"
DEFAULT_SRC = Path(r"D:\temp\1\MatViz3D\img\MatViz3D.png")
BADGE = (214, 64, 40, 255)      # orange-red, contrasts with the teal cube
TEXT = (255, 255, 255, 255)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def badge(base: Image.Image, size: int, band: float, font_frac: float,
          shrink: bool = False) -> Image.Image:
    """Resize ``base`` to ``size`` and draw an "ACT" band at the bottom.

    shrink: scale the logo into the area above the band instead of letting the
    band cover its lower part (keeps the "MV" letters visible at large sizes).
    """
    h = max(5, round(size * band))
    y0 = size - h
    if shrink:
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        side = y0 + round(size * 0.04)          # logo slightly tucked under the band
        logo = base.convert("RGBA").resize((side, side), Image.LANCZOS)
        im.paste(logo, ((size - side) // 2, 0), logo)
    else:
        im = base.convert("RGBA").resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    r = max(1, round(size * 0.06))
    d.rounded_rectangle([0, y0, size - 1, size - 1], radius=r, fill=BADGE)
    f = _font(max(5, round(size * font_frac)))
    l, t, rgt, b = d.textbbox((0, 0), "ACT", font=f)
    d.text(((size - (rgt - l)) / 2 - l, y0 + (h - (b - t)) / 2 - t), "ACT", font=f, fill=TEXT)
    return im


def main(src: Path) -> None:
    base = Image.open(src)
    OUT.mkdir(parents=True, exist_ok=True)
    badge(base, 128, band=0.26, font_frac=0.22, shrink=True).save(OUT / "rve_ext.png")
    badge(base, 32, band=0.38, font_frac=0.34).save(OUT / "rve32.png")
    # 16 px: draw the band at 64 px and downsample — text stays legible
    small = badge(base, 64, band=0.44, font_frac=0.42).resize((16, 16), Image.LANCZOS)
    small.save(OUT / "rve.png")
    # Mechanical tree objects use 16x16 24-bit BMP (as the shipped LSDYNA
    # extension does): light theme on white, dark theme in images/dark on (33,33,33)
    (OUT / "dark").mkdir(exist_ok=True)
    for folder, bg in ((OUT, (255, 255, 255)), (OUT / "dark", (33, 33, 33))):
        flat = Image.new("RGB", small.size, bg)
        flat.paste(small, (0, 0), small)
        flat.save(folder / "rve.bmp")
    small.save(OUT / "dark" / "rve.png")
    print("icons written to", OUT)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC)
