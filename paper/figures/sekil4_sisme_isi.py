# -*- coding: utf-8 -*-
"""
Şekil 4 — Şişme ısı haritası: yöntem × ufuk × rejim.

Makalenin ana tablosunun görsel hâli. Her hücre, `none` kontrolüne göre
kazancın yedi mimari üzerindeki medyanı.

Okunacak şey soldan sağa: tam-seri ve bölüm bazlı panelleri birbirine
benziyor, origin bazlı panel sıfır civarında ve çoğu yerde NEGATİF.
Yani kazanç ayrıştırmadan değil, ayrıştırmanın ne zaman hesaplandığından
geliyor.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stil import hazirla, kaydet, ARM_AD, TAM_SAYFA
import veri


def main():
    if not veri.var("leakage_inflation.json"):
        print("  ATLANDI — leakage_inflation.json yok.  Once: python run.py analyze")
        return 1
    hazirla()
    M = veri.sisme()
    kollar = [k for k in veri.KOL if k in M]
    fig, axes = plt.subplots(1, len(kollar), figsize=(TAM_SAYFA, 2.3),
                             sharey=True)
    vmax = float(np.nanmax([np.nanmax(np.abs(v)) for v in M.values()]))
    for ax, kol in zip(np.atleast_1d(axes), kollar):
        im = ax.imshow(M[kol], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(veri.UFUK)))
        ax.set_xticklabels([f"h={h}" for h in veri.UFUK])
        ax.set_title(ARM_AD[kol], loc="left", pad=5)
        for i in range(len(veri.YONTEM)):
            for j in range(len(veri.UFUK)):
                v = M[kol][i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                            fontsize=6.6,
                            color="white" if abs(v) > vmax * 0.55 else "black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    np.atleast_1d(axes)[0].set_yticks(range(len(veri.YONTEM)))
    np.atleast_1d(axes)[0].set_yticklabels([m.upper() for m in veri.YONTEM])
    # Dikey sömürgeç etiketi tam sayfa genişliğinde kesiliyordu; yatay ve
    # panellerin altında.
    cb = fig.colorbar(im, ax=list(np.atleast_1d(axes)), orientation="horizontal",
                      pad=0.22, fraction=0.055, aspect=45)
    cb.set_label("median gain over the undecomposed control (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    kaydet(fig, "sekil4_sisme_isi", os.path.dirname(os.path.abspath(__file__)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
