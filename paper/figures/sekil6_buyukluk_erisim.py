# -*- coding: utf-8 -*-
"""
Şekil 6 — Büyüklük ve ufka erişim.

İki eksenli dağılım: yatayda h=1'deki şişme (BÜYÜKLÜK), dikeyde 24 saatte
kalan pay h24/h1 (ERİŞİM), yani bir saatlik şişmenin ne kadarının en uzak
ufka taşındığı. İddia: büyüklük ayrıştırmanın küreselliğine göre, erişim ise
zamansal desteğine göre sıralanıyor.

DWT sonlu destekli: şişmesi orta (+49) ama 24 saatte %2'si bile kalmıyor.
VMD küresel: hem şişmesi en büyük (+93) hem 24 saatte üçte ikisi duruyor.
EMD ailesi arada (~0,13). Bu iki eksen aynı sayı değil ve şekil onları ayırıyor.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stil import hazirla, kaydet, LEAKY, TEK_SUTUN
import veri

# (dx, dy, hizalama) — nokta yogunlugu elle cozuluyor, otomatik yerlestirme
# bu kadar az noktada daha kotu sonuc veriyor.
SAPMA = {"dwt": (0, 9, "center"), "vmd": (-10, -9, "right"),
         "emd": (0, 9, "center"), "eemd": (-8, -4, "right"),
         "ceemdan": (9, -4, "left")}

DESTEK = {"dwt": "finite support", "vmd": "global", "emd": "global",
          "eemd": "global", "ceemdan": "global"}


def main():
    if not veri.var("leakage_inflation.json"):
        print("  ATLANDI — leakage_inflation.json yok.")
        return 1
    hazirla()
    M = veri.sisme()["leaky"]
    fig, ax = plt.subplots(figsize=(TEK_SUTUN * 1.7, 2.3))
    for i, m in enumerate(veri.YONTEM):
        h1, h24 = M[i, 0], M[i, 3]
        if np.isnan(h1) or np.isnan(h24) or h1 == 0:
            continue
        oran = h24 / h1
        yerel = DESTEK[m] == "finite support"
        ax.scatter(h1, oran, s=52, marker="D" if yerel else "o",
                   color=LEAKY, edgecolor="#333333", lw=0.6, zorder=3)
        # EEMD (61,5) ile CEEMDAN (64,6) yatayda 3 puan arayla duruyor;
        # ikisini de isaretin ustune koyunca etiketler ust uste bindi.
        # Yakin duran ciftler dikeyde ayriliyor.
        dx, dy, ha = SAPMA.get(m, (0, 9, "center"))
        ax.annotate(m.upper(), (h1, oran), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7)
    ax.axhline(1.0, color="#AAAAAA", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(20, 106)
    ax.set_ylim(0.008, 2.2)
    # Aciklama kesikli cizginin ALTINDA. Ustte solda EMD isaretini, sagda
    # CEEMDAN ve VMD etiketlerini kesiyordu; y ekseninde noktanin olmadigi
    # tek serit cizginin altidir.
    ax.text(21, 1.08, "1: the one-hour inflation carries undiminished to $h$ = 24",
            fontsize=6.3, va="bottom", ha="left", color="#777777")
    ax.set_xlabel("inflation at $h$ = 1 (%)   —   magnitude")
    ax.set_ylabel("retained at $h$ = 24   —   reach")
    hs = [plt.Line2D([], [], ls="", marker="D", color=LEAKY, ms=5,
                     markeredgecolor="#333333"),
          plt.Line2D([], [], ls="", marker="o", color=LEAKY, ms=5,
                     markeredgecolor="#333333")]
    ax.legend(hs, ["finite support (local)", "global"], frameon=False,
              fontsize=6.6, loc="lower right")
    kaydet(fig, "sekil6_buyukluk_erisim", os.path.dirname(os.path.abspath(__file__)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
