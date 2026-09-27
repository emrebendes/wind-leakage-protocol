# -*- coding: utf-8 -*-
"""
Ortak çizim stili.

Applied Energy sanat eseri kuralları:
  vektör çizimler EPS veya PDF, gömülü font; raster en az 300 dpi.
  Tek sütun genişliği ~90 mm, tam sayfa ~190 mm.
  Renk körlüğüne erişilebilir olmalı — bu yüzden paletteki her seri
  aynı zamanda çizgi tipi veya işaretle de ayrışıyor, yalnız renkle değil.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
TEK_SUTUN = 90 * MM          # 90 mm
TAM_SAYFA = 190 * MM         # 190 mm

# Okada/Wong tabanlı, renk körlüğüne güvenli palet
LEAKY = "#D55E00"     # kiremit  — tam seri
PART  = "#0072B2"     # mavi     — bölüm bazlı
CAUS  = "#009E73"     # yeşil    — origin bazlı
NONE  = "#666666"
VURGU = "#CC79A7"

ARM_RENK = {"leaky": LEAKY, "partition": PART, "causal": CAUS, "none": NONE}
ARM_AD = {"leaky": "Whole-series", "partition": "Partition-level",
          "causal": "Per-origin", "none": "Undecomposed"}


def hazirla():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1, "figure.dpi": 120,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # Elsevier: vektörde font gömülü olsun (TrueType, Type 3 değil)
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def kaydet(fig, ad, klasor="."):
    """PDF (yayın için, vektör, font gömülü) + PNG 500 dpi (Elsevier: çizgi+yarım ton karışımı için en az 500 dpi)."""
    import os
    for ext, kw in (("pdf", {}), ("png", {"dpi": 500})):
        fig.savefig(os.path.join(klasor, f"{ad}.{ext}"), **kw)
    plt.close(fig)
    print(f"  {ad}.pdf / {ad}.png")
