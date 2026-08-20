"""Generate the coordinate-frame diagram used in essay0818_revised.tex.

The PDF is sized for one IEEE conference column (3.5 in wide) and keeps all
geometry and text as vector content.
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "coordinate_frame.pdf"

PAGE_W = 252.0  # 3.5 in: IEEE single-column width
PAGE_H = 194.0

INK = HexColor("#17212B")
MUTED = HexColor("#66727D")
GRID = HexColor("#CBD3DA")
COMMAND = HexColor("#D97706")
POSITION = HexColor("#0F7C86")
VELOCITY = HexColor("#6D4CC3")
BALL_FILL = HexColor("#DDEFF4")
PELVIS_FILL = HexColor("#EEF1F4")


def register_fonts() -> None:
    fonts = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("TimesNewRoman", str(fonts / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRomanBold", str(fonts / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRomanItalic", str(fonts / "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRomanBoldItalic", str(fonts / "timesbi.ttf")))
    pdfmetrics.registerFontFamily(
        "TimesNewRoman",
        normal="TimesNewRoman",
        bold="TimesNewRomanBold",
        italic="TimesNewRomanItalic",
        boldItalic="TimesNewRomanBoldItalic",
    )


def arrow(
    c: canvas.Canvas,
    start: tuple[float, float],
    end: tuple[float, float],
    color,
    width: float = 1.5,
    head: float = 5.0,
    dash: tuple[float, float] | None = None,
) -> None:
    x0, y0 = start
    x1, y1 = end
    angle = math.atan2(y1 - y0, x1 - x0)
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dash:
        c.setDash(*dash)
    c.line(x0, y0, x1, y1)
    c.setDash()
    left = (
        x1 - head * math.cos(angle) + 0.55 * head * math.sin(angle),
        y1 - head * math.sin(angle) - 0.55 * head * math.cos(angle),
    )
    right = (
        x1 - head * math.cos(angle) - 0.55 * head * math.sin(angle),
        y1 - head * math.sin(angle) + 0.55 * head * math.cos(angle),
    )
    path = c.beginPath()
    path.moveTo(x1, y1)
    path.lineTo(*left)
    path.lineTo(*right)
    path.close()
    c.drawPath(path, stroke=0, fill=1)
    c.restoreState()


def arc_arrow(
    c: canvas.Canvas,
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    color,
    width: float = 1.2,
) -> None:
    cx, cy = center
    points = []
    count = max(8, int(abs(end_deg - start_deg) / 3))
    for i in range(count + 1):
        angle = math.radians(start_deg + (end_deg - start_deg) * i / count)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()

    arrow(c, points[-2], points[-1], color, width=width, head=3.8)


def label(
    c: canvas.Canvas,
    html: str,
    x: float,
    y: float,
    width: float,
    *,
    size: float = 7.6,
    color=INK,
    align: int = TA_LEFT,
    leading: float | None = None,
) -> None:
    style = ParagraphStyle(
        "figure-label",
        fontName="TimesNewRoman",
        fontSize=size,
        leading=leading or size * 1.12,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        spaceBefore=0,
    )
    paragraph = Paragraph(html, style)
    _, height = paragraph.wrap(width, 40)
    paragraph.drawOn(c, x, y - height)


def math_label(
    c: canvas.Canvas,
    base: str,
    x: float,
    y: float,
    *,
    superscript: str | None = None,
    subscript: str | None = None,
    size: float = 9.0,
    color=INK,
    center: bool = False,
    background: bool = False,
) -> None:
    """Draw a compact math-style label without paragraph line wrapping."""
    script_size = size * 0.57
    base_width = pdfmetrics.stringWidth(base, "TimesNewRomanItalic", size)
    script_width = max(
        pdfmetrics.stringWidth(superscript or "", "TimesNewRomanItalic", script_size),
        pdfmetrics.stringWidth(subscript or "", "TimesNewRomanItalic", script_size),
    )
    total_width = base_width + (1.0 + script_width if script_width else 0.0)
    if center:
        x -= total_width / 2

    c.saveState()
    if background:
        c.setFillColor(white)
        c.roundRect(
            x - 1.5,
            y - size * 0.44,
            total_width + 3.0,
            size * 1.52,
            1.2,
            stroke=0,
            fill=1,
        )
    c.setFillColor(color)
    c.setFont("TimesNewRomanItalic", size)
    c.drawString(x, y, base)
    script_x = x + base_width + 0.8
    if superscript:
        c.setFont("TimesNewRomanItalic", script_size)
        c.drawString(script_x, y + size * 0.47, superscript)
    if subscript:
        c.setFont("TimesNewRomanItalic", script_size)
        c.drawString(script_x, y - size * 0.31, subscript)
    c.restoreState()


def point(origin: tuple[float, float], length: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    return origin[0] + length * math.cos(theta), origin[1] + length * math.sin(theta)


def generate() -> Path:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("World-parallel task frame and command-relative ball velocity")
    c.setAuthor("Footmimic project")

    origin = (43.0, 38.0)
    psi = 26.0
    beta = 52.0
    chi = 72.0

    # Quiet frame annotation.
    c.setFillColor(HexColor("#F6F8FA"))
    c.roundRect(11, 178, 230, 12, 3.5, stroke=0, fill=1)
    label(
        c,
        "World-parallel task axes; the origin follows the pelvis",
        17,
        187.0,
        218,
        size=7.0,
        color=MUTED,
        align=TA_CENTER,
    )

    # Task axes.
    arrow(c, origin, (235, origin[1]), INK, width=1.25, head=5.2)
    arrow(c, origin, (origin[0], 161), INK, width=1.25, head=5.2)
    label(c, "task +<i>X</i> (field-forward)", 158, 34.0, 76, size=7.0, color=INK)
    label(c, "task +<i>Y</i><br/>(lateral)", 48, 160.0, 50, size=7.0, color=INK)

    # Pelvis origin marker.
    c.setFillColor(PELVIS_FILL)
    c.setStrokeColor(INK)
    c.setLineWidth(1.1)
    c.roundRect(origin[0] - 8, origin[1] - 5, 16, 10, 3, stroke=1, fill=1)
    c.setLineWidth(0.8)
    c.line(origin[0] - 4, origin[1], origin[0] + 4, origin[1])
    c.line(origin[0], origin[1] - 3, origin[0], origin[1] + 3)
    label(c, "pelvis origin", 10, 27.5, 64, size=7.0, color=INK, align=TA_CENTER)

    # Command direction from the pelvis and its angle from task +X.
    command_end = point(origin, 121, psi)
    arrow(c, origin, command_end, COMMAND, width=2.0, head=6.0)
    c.setFillColor(COMMAND)
    c.setFont("TimesNewRoman", 7.2)
    c.drawCentredString(188, 102, "command direction")
    math_label(c, "d", 188, 90.5, superscript="cmd", subscript="t", size=9.0, color=COMMAND, center=True)
    arc_arrow(c, origin, 24, 0, psi, COMMAND)
    math_label(
        c,
        "ψ",
        69,
        46.5,
        superscript="cmd",
        subscript="t",
        size=8.7,
        color=COMMAND,
        center=True,
        background=True,
    )

    # Pelvis-to-ball position: angle beta is measured from the same task +X.
    ball = point(origin, 113, beta)
    arrow(c, origin, ball, POSITION, width=2.15, head=6.2)
    arc_arrow(c, origin, 43, 0, beta, POSITION)
    math_label(c, "β", 81, 63.5, superscript="ball", subscript="t", size=8.7, color=POSITION, center=True)
    math_label(c, "d", 83, 101, superscript="ball", subscript="t", size=9.0, color=POSITION, center=True)

    # Ball marker.
    c.setFillColor(BALL_FILL)
    c.setStrokeColor(POSITION)
    c.setLineWidth(1.5)
    c.circle(ball[0], ball[1], 6.3, stroke=1, fill=1)
    c.setFillColor(POSITION)
    c.circle(ball[0], ball[1], 1.4, stroke=0, fill=1)
    c.setFillColor(POSITION)
    c.setFont("TimesNewRoman", 7.1)
    c.drawRightString(ball[0] - 10, ball[1] + 12, "physical ball")

    # Translate the command direction to the ball so Delta chi is visually
    # unambiguous: it is not measured from task +X.
    translated_command_end = point(ball, 58, psi)
    arrow(c, ball, translated_command_end, COMMAND, width=1.0, head=4.2, dash=(3, 2))
    c.setFillColor(COMMAND)
    c.setFont("TimesNewRoman", 6.7)
    c.drawString(169, 139.5, "parallel to")
    math_label(c, "d", 207, 138.4, superscript="cmd", subscript="t", size=8.0, color=COMMAND)

    velocity_end = point(ball, 41, chi)
    arrow(c, ball, velocity_end, VELOCITY, width=2.15, head=6.0)
    c.setFillColor(VELOCITY)
    c.setFont("TimesNewRoman", 7.1)
    c.drawString(144, 166, "ball velocity")
    math_label(c, "v", 175, 154, superscript="ball", subscript="t,xy", size=9.0, color=VELOCITY, center=True)
    arc_arrow(c, ball, 22, psi, chi, VELOCITY)
    math_label(c, "Δχ", 143, 126.5, superscript="ball", subscript="t", size=9.2, color=VELOCITY, center=True)

    c.showPage()
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(generate())
