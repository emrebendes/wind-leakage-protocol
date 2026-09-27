# -*- coding: utf-8 -*-
"""
Şekil 1 — Üç ayrıştırma rejimi.

MAKALENİN TEK EN ÖNEMLİ ŞEKLİ. Metnin bütün iddiası burada görülmeli:
bir test orijini t için üç rejimin ayrıştırıcıya verdiği blok B(t), ve
o bloğun t'den SONRAKİ kısmı.

Kritik görsel nokta: tam-seri ve bölüm-bazlı panellerinde t'nin sağındaki
kırmızı bölge AYNI. Okur bunu tabloyu okumadan görmeli — Kanıt 0 budur.

Veri gerekmez; sayılar gerçek bölme sınırlarından (n = 50.718).
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stil import hazirla, kaydet, LEAKY, PART, CAUS, TAM_SAYFA

N = 50718
TRAIN_END, VAL_END = 35502, 43110
ISINMA = 167
T = 46110                      # örnek test orijini
W = 512

GELECEK = "#E8ADA5"            # t'den sonrası — sızıntı bölgesi
BLOK = "#DCE7F0"               # ayrıştırıcının gördüğü blok


def panel(ax, ad, lo, hi, renk, aciklama):
    ax.set_xlim(-1200, N + 1200)
    ax.set_ylim(0, 1)
    # Ticks off, not just spines off. The first version hid the spines and left
    # the tick labels, so each of the three panels printed its own identical
    # x-axis and the figure carried the same scale three times.
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

    # tüm kayıt, soluk zemin
    ax.add_patch(Rectangle((0, 0.32), N, 0.36, fc="#F2F2F2", ec="none"))

    # B(t): ayrıştırıcının gördüğü blok
    ax.add_patch(Rectangle((lo, 0.32), hi - lo, 0.36, fc=BLOK,
                           ec=renk, lw=1.2))
    # bloğun t'den sonraki kısmı — görünen gelecek
    if hi > T:
        ax.add_patch(Rectangle((T, 0.32), hi - T, 0.36, fc=GELECEK,
                               ec="none", hatch="///", lw=0))

    # bölme sınırları
    for x in (TRAIN_END, VAL_END):
        ax.plot([x, x], [0.28, 0.72], color="#999999", lw=0.6, ls=(0, (2, 2)))

    # tahmin orijini
    ax.plot([T, T], [0.20, 0.80], color="black", lw=1.4)
    ax.plot(T, 0.80, marker="v", ms=4, color="black")

    ax.text(-900, 0.50, ad, ha="right", va="center", fontsize=8,
            fontweight="bold", color=renk)
    ax.text(N + 900, 0.50, aciklama, ha="left", va="center", fontsize=6.8,
            color="#444444")


def main():
    hazirla()
    fig, axes = plt.subplots(3, 1, figsize=(TAM_SAYFA, 2.5))
    fig.subplots_adjust(left=0.16, right=0.72, hspace=0.62, top=0.84,
                        bottom=0.26)

    panel(axes[0], "Whole-series\n(leaky)", 0, N, LEAKY,
          "B(t) = [0, n)\nfuture seen: (t, n)")
    panel(axes[1], "Partition-level", VAL_END - ISINMA, N, PART,
          "B(t) = [42943, n)\nfuture seen: (t, n)  — identical")
    panel(axes[2], "Per-origin\n(proposed)", T - W + 1, T, CAUS,
          "B(t) = [t-W+1, t]\nfuture seen: none")

    # Etiketler üç yere dağıtıldı, çünkü hepsi alt satırdayken üst üste
    # biniyordu: origin işareti en üste, bölüm adları alt eksenin altına,
    # W notu en alta. Sınır sayıları (35 502 / 43 110) şekilden çıkarıldı —
    # metinde zaten var ve burada birbirine giriyorlardı.
    axes[0].annotate("forecast origin $t$", xy=(T, 1.30),
                     xycoords=("data", "axes fraction"), ha="center",
                     va="bottom", fontsize=7.5, annotation_clip=False)

    ax = axes[2]
    ax.annotate("", xy=(0, -0.16), xytext=(N, -0.16),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#BBBBBB"),
                annotation_clip=False)
    # Uc noktalar bolum adlarinin ALTINDAKI satirda: test bolumu genisligin
    # yalnizca %15'i ve merkezi saga yakin, dolayisiyla ayni satirda "n = ..."
    # ile ust uste biniyordu.
    for x, lbl, ha in ((0, "0", "left"),
                       (N, f"n = {N:,}".replace(",", " "), "right")):
        ax.annotate(lbl, xy=(x, -0.52), xycoords=("data", "axes fraction"),
                    ha=ha, va="top", fontsize=6.5, color="#AAAAAA",
                    annotation_clip=False)
    for x, lbl in ((TRAIN_END / 2, "train (70%)"),
                   ((TRAIN_END + VAL_END) / 2, "val (15%)"),
                   ((VAL_END + N) / 2, "test (15%)")):
        ax.annotate(lbl, xy=(x, -0.30), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=6.8, color="#666666",
                    annotation_clip=False)
    for x in (TRAIN_END, VAL_END):
        ax.annotate("", xy=(x, -0.10), xytext=(x, -0.22),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#BBBBBB"),
                    annotation_clip=False)

    # W = 512, kaydın binde üçü. Bu ölçekte kıl kadar görünmesi dürüst olan
    # resim ve şeklin bütün meselesi, ama sözle söylenmesi gerekiyor.
    ax.annotate(f"$W = {W}$ samples — 0.3% of the record",
                xy=(T - W, 0.32), xytext=(TRAIN_END * 0.60, -1.05),
                xycoords="data", textcoords=("data", "axes fraction"),
                fontsize=6.6, color=CAUS, ha="center", va="center",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=CAUS,
                                connectionstyle="arc3,rad=-0.15",
                                shrinkA=2, shrinkB=1))

    # açıklayıcı gösterge
    hs = [Rectangle((0, 0), 1, 1, fc=BLOK, ec="#888888", lw=0.8),
          Rectangle((0, 0), 1, 1, fc=GELECEK, ec="none", hatch="///")]
    fig.legend(hs, ["block given to the decomposition, $B(t)$",
                    "part of that block after $t$ — look-ahead"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.44, -0.20))

    kaydet(fig, "sekil1_uc_rejim", os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
