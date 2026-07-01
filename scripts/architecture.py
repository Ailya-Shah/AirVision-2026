"""
architecture.py — regenerates architecture.png (AirVision-2026 pipeline diagram).

Run:  python architecture.py   ->   writes architecture.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----- palette -------------------------------------------------------------
BLUE   = dict(fc="#dfe7f3", ec="#3f6fa8")   # satellite / reanalysis sources
PURPLE = dict(fc="#e9e3f6", ec="#6b51b5")   # Earth Engine
TAN    = dict(fc="#ece7dd", ec="#8f7a5f")   # on-disk data stages
GREEN  = dict(fc="#deefdd", ec="#4f9d5b")   # notebook processing stages
RED    = dict(fc="#f8dedd", ec="#c23a2b")   # dashboard
TITLE_C = "#1a1a1a"
SUB_C   = "#555555"
ARROW_C = "#555555"

fig, ax = plt.subplots(figsize=(12.8, 9.9), dpi=170)
ax.set_xlim(0, 100)
ax.set_ylim(-8, 100)
ax.axis("off")


def box(cx, cy, w, h, title, sub, style):
    """Rounded box centred at (cx, cy) with a bold title and grey subtitle."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=2, facecolor=style["fc"], edgecolor=style["ec"],
        mutation_aspect=0.5))
    ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color=TITLE_C)
    ax.text(cx, cy - h * 0.24, sub, ha="center", va="center",
            fontsize=11, color=SUB_C)
    return dict(cx=cx, cy=cy, w=w, h=h)


def arrow(a, b, a_side="bottom", b_side="top"):
    """Draw an arrow from box a to box b along the given sides."""
    def point(bx, side):
        if side == "bottom": return (bx["cx"], bx["cy"] - bx["h"] / 2)
        if side == "top":    return (bx["cx"], bx["cy"] + bx["h"] / 2)
    ax.add_patch(FancyArrowPatch(
        point(a, a_side), point(b, b_side),
        arrowstyle="-|>", mutation_scale=18, lw=2,
        color=ARROW_C, shrinkA=2, shrinkB=4))


# ----- title ---------------------------------------------------------------
ax.text(50, 97, "AirVision-2026  —  Pipeline Architecture",
        ha="center", va="center", fontsize=20, fontweight="bold", color=TITLE_C)

# ----- nodes ---------------------------------------------------------------
s1 = box(17, 87, 28, 11, "Sentinel-5P  (NO₂)", "satellite · combustion proxy", BLUE)
s2 = box(50, 87, 28, 11, "CAMS  (PM2.5)",       "model · µg/m³",              BLUE)
s3 = box(83, 87, 28, 11, "ERA5-Land  (weather)", "reanalysis · temp/wind/rain", BLUE)

gee = box(50, 71, 40, 11, "Google Earth Engine",
          "extraction · 15 cities · per-year exports", PURPLE)
raw = box(50, 56, 34, 10, "data/raw/",
          "24 yearly CSVs (3 vars × 8 years)", TAN)
clean = box(50, 42, 44, 10, "Cleaning + merge  ·  notebook Part A",
            "NaN-safe · negatives kept · join on (city, date)", GREEN)
proc = box(50, 28, 38, 10, "data/processed/",
           "master_daily · weekly · monthly", TAN)

partB = box(24, 14, 40, 11, "Analysis  ·  Part B",
            "rankings · seasonality · trends · COVID · weather", GREEN)
partC = box(76, 14, 40, 11, "ML model  ·  Part C",
            "PM2.5 ~ NO₂ + weather + season · grouped CV", GREEN)

dash = box(50, -1, 42, 11, "Streamlit dashboard  ·  app.py",
           "explore layers · live PM2.5 predictor", RED)

# ----- edges ---------------------------------------------------------------
for s in (s1, s2, s3):
    arrow(s, gee)
arrow(gee, raw)
arrow(raw, clean)
arrow(clean, proc)
arrow(proc, partB)
arrow(proc, partC)
arrow(partB, dash)
arrow(partC, dash)

plt.tight_layout()
fig.savefig("architecture.png", bbox_inches="tight", facecolor="white")
print("wrote architecture.png")