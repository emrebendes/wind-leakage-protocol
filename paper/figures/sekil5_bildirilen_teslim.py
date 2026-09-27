# -*- coding: utf-8 -*-
"""
Şekil 5 — Bildirilen, teslim edilen ve kalıcılık; iki rejim için.

Makalenin operasyonel iddiasını taşıyan şekil, ve başlıktaki iddianın en
güçlü hâli. Her hücre için üç sayı: modelin kendi kolunda raporladığı RMSE,
aynı modelin origin bazlı özniteliklerle teslim ettiği RMSE, ve kalıcılık.

Kalıcılık yatay çizgi. Çizginin ÜSTÜNDE kalan her nokta "hiçbir şey
yapmamaktan kötü" demektir. İki panelde de 35/35 nokta çizginin üstünde —
bölüm bazlı çare operasyonel geçerlilik bakımından hiçbir şey satın almıyor.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stil import hazirla, kaydet, LEAKY, PART, CAUS, TAM_SAYFA
import veri

KAYNAK = [("deployment_gap.json", "trained under whole-series", LEAKY),
          ("deployment_gap_partition.json", "trained under partition-level", PART)]


def main():
    yok = [a for a, _, _ in KAYNAK if not veri.var(a)]
    if yok:
        print(f"  ATLANDI — yok: {', '.join(yok)}")
        return 1
    hazirla()
    fig, axes = plt.subplots(1, 2, figsize=(TAM_SAYFA, 2.6), sharey=True)
    for ax, (dosya, baslik, renk) in zip(axes, KAYNAK):
        rows = veri.teslim(dosya)
        x = np.arange(len(rows))
        rep = [r[1] for r in rows]
        dep = [r[2] for r in rows]
        pers = float(np.mean([r[3] for r in rows]))
        ax.vlines(x, rep, dep, color="#D0D0D0", lw=1.0, zorder=1)
        ax.scatter(x, rep, s=13, color=CAUS, marker="o", zorder=3,
                   label="reported by the arm")
        ax.scatter(x, dep, s=15, color=renk, marker="^", zorder=3,
                   label="delivered on causal inputs")
        ax.axhline(pers, color="black", lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=4.8)
        ax.set_title(baslik, loc="left", pad=5)
        ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=1)
        ust = sum(1 for v in dep if v >= pers)
        ax.text(0.985, 0.04, f"{len(rows) - ust} of {len(rows)} beat persistence",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
                fontweight="bold")
    axes[0].set_ylabel("RMSE at $h$ = 1 (m/s)")
    # Etiket eksenin ICINDE: disarida y ekseni sayilarinin ustune biniyordu.
    axes[0].text(0.985, 0.63, "persistence = 0.7008", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=6.5)
    fig.subplots_adjust(wspace=0.06, bottom=0.34)
    kaydet(fig, "sekil5_bildirilen_teslim", os.path.dirname(os.path.abspath(__file__)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
