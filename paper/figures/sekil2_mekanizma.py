# -*- coding: utf-8 -*-
"""
Şekil 2 — Ölçülen farkın mekanizması: bilgi mi, hizalama artefaktı mı?

İki panel, çünkü iki ayrı iddia var ve ayrı ayrı gösterilmeleri gerekiyor:

(a) KAYDIRMA SONDASI. Blok uzunluğu sabit, başlangıcı birer örnek geriye
    kayıyor, aynı zaman damgasındaki öznitelik okunuyor. DWT'de desen tam
    periyodik (periyot 4 = 2^level) ve dördün katlarında fark BİREBİR SIFIR.
    VMD'de böyle bir yapı yok, gürültü tabanı düz.

(b) İKİ EKSEN SONDASI. Bölüm bazlı rejim bulaşmayı tam sıfırlıyor, ileri
    bakışı sıfırlamıyor. Makalenin tek cümlelik iddiası bu paneldir.

Sayılar `rejim_farki.py` ve nedensellik sondalarının kayıtlı çıktısından;
kaydırma sondası istasyon 0, t = 46.902, W = 512, blok 4.096; Tablo 4 t = 46.110.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stil import hazirla, kaydet, LEAKY, PART, CAUS, TAM_SAYFA

VERI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "veriler", "hizalama_sondasi.csv")


def sonda_oku():
    """
    Kaydirma sondasinin OLCULEN degerleri: veriler/hizalama_sondasi.csv,
    rejim_farki.py alignment_probe() ciktisi (istasyon 0, W=512, blok 4096,
    origin 46902, 40 kaydirma; oturum_kaydi.md Tur 219). Ilk surumde 5-16
    arasi desenden turetilmisti; simdi 0-39 tamami olculmus degerdir.
    """
    import csv
    kayma, dwt, vmd = [], [], []
    with open(VERI, encoding="utf-8") as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            kayma.append(int(row["kayma"]))
            dwt.append(float(row["dwt"]) if row["dwt"] else None)
            vmd.append(float(row["vmd"]) if row["vmd"] else None)
    return kayma, dwt, vmd


LEVEL = 2
# Tablo 4 — tablo4_sondasi.py, istasyon 0, t = 46 110, W = 512, ayarlanmış parametreler
# (DWT db4 J=2, üç deneme; VMD K=8 alpha=500, tek deneme). 19 Eyl 2026, Emre'nin PC'si.
KOL = ["DWT\nwhole-series", "DWT\npartition-level", "DWT\nper-origin",
       "VMD\nwhole-series", "VMD\npartition-level", "VMD\nper-origin"]
GELECEK = [8.94e-1, 3.65e-1, 0.0, 5.67e-1, 8.40e-1, 0.0]   # geleceği permüte et
EGITIM = [0.0, 0.0, 0.0, 8.22e-1, 0.0, 0.0]                 # eğitim bölümünü permüte et
TABAN = 1e-4                          # sıfırın log eksende görünmesi için


def main():
    """İki ayrı dosya: sekilA1_kaydirma (Ek A, kaydırma sondası) ve
    sekil2_permutasyon (ana metin, Tablo 4). Eski birleşik şekil kaldırıldı."""
    hazirla()
    from stil import TEK_SUTUN
    figA, a = plt.subplots(figsize=(TEK_SUTUN * 1.15, 2.6))
    figB, b = plt.subplots(figsize=(TEK_SUTUN * 1.15, 2.7))

    # ---- (a) kaydırma sondası -------------------------------------------
    kayma, dwt, vmd = sonda_oku()
    kayma, dwt, vmd = kayma[:17], dwt[:17], vmd[:17]   # sekil 0-16 gosterir
    kd = [(k, v) for k, v in zip(kayma, dwt) if v is not None]
    kv = [(k, v) for k, v in zip(kayma, vmd) if v is not None]
    # bitisik olmayan noktalar (varsa) ayri parcalar halinde cizilir.
    def parcalar(pts):
        out, cur = [], [pts[0]]
        for a_, b_ in zip(pts, pts[1:]):
            if b_[0] - a_[0] == 1:
                cur.append(b_)
            else:
                out.append(cur); cur = [b_]
        out.append(cur)
        return out

    for i, seg in enumerate(parcalar(kd)):
        xs = [p[0] for p in seg]; ys = [max(p[1], TABAN) for p in seg]
        a.plot(xs, ys, "o-" if len(seg) > 1 else "o", color=LEAKY, ms=3.4,
               label="DWT (finite support, shift-variant)" if i == 0 else None)
    for i, seg in enumerate(parcalar(kv)):
        xs = [p[0] for p in seg]; ys = [max(p[1], TABAN) for p in seg]
        a.plot(xs, ys, "s--" if len(seg) > 1 else "s", color=PART, ms=3.0,
               label="VMD (global)" if i == 0 else None)
    katlar = [k for k, v in kd if k and k % 2 ** LEVEL == 0]
    for k in katlar:
        a.axvline(k, color="#CCCCCC", lw=0.5, zorder=0)
    a.scatter(katlar, [TABAN] * len(katlar), marker="v", s=22, color=LEAKY,
              zorder=5, clip_on=False)
    a.set_yscale("log")
    a.set_ylim(TABAN * 0.75, 0.4)
    a.set_xlim(-0.4, 16.6)
    a.set_xticks([0, 1, 2, 3, 4, 8, 12, 16])
    a.set_xlabel("block start shifted back (samples)")
    a.set_ylabel("relative difference vs. shift 0")
    a.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    a.set_yticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    a.legend(frameon=False, loc="lower right", handlelength=2.2)
    a.set_title("Shift probe: the difference is periodic with $2^{J} = 4$",
                loc="left", pad=6)
    a.annotate("exactly zero at every multiple of 4", xy=(12, TABAN),
               xytext=(5.2, 4.5e-3), fontsize=6.6, color=LEAKY, ha="left",
               arrowprops=dict(arrowstyle="->", lw=0.6, color=LEAKY))

    # ---- (b) iki eksen sondası ------------------------------------------
    y = np.arange(len(KOL))[::-1]
    h = 0.34
    b.barh(y + h / 2, np.maximum(GELECEK, TABAN), height=h, color="#E8ADA5",
           edgecolor="#B5544A", lw=0.6, label="permute the future")
    b.barh(y - h / 2, np.maximum(EGITIM, TABAN), height=h, color="#BFD4E4",
           edgecolor=PART, lw=0.6, label="permute the training partition")
    b.set_xscale("log")
    b.set_xlim(TABAN * 0.75, 3.0)
    b.set_yticks(y)
    b.set_yticklabels(KOL)
    b.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    b.set_xticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
    b.set_xlabel("largest absolute change in the features (m/s)")
    b.tick_params(axis="y", labelsize=6.3)
    b.axhline(2.5, color="#BBBBBB", lw=0.5)
    b.legend(frameon=False, loc="lower right", fontsize=6.4, bbox_to_anchor=(1.0, -0.02))
    b.set_title("Permutation probe: which leakage each regime removes", loc="left", pad=6)
    for yy, v in zip(y + h / 2, GELECEK):
        if v == 0:
            b.text(TABAN * 1.15, yy, "0", va="center", ha="left", fontsize=6.5)
    for yy, v in zip(y - h / 2, EGITIM):
        if v == 0:
            b.text(TABAN * 1.15, yy, "0", va="center", ha="left", fontsize=6.5)
    # Not, isaret ettigi satirin ALTINA konuyor. Ilk yerlesimde Whole-series
    # satirinda duruyor ve oku asagi iniyordu, dolayisiyla yanlis satiri
    # etiketliyormus gibi okunuyordu.
    b.annotate("contamination gone,\nlook-ahead intact",
               xy=(TABAN * 1.6, y[4] - h / 2), xytext=(4e-3, y[2] - 0.05),
               fontsize=6.4, color=PART, ha="left", va="center",
               arrowprops=dict(arrowstyle="->", lw=0.6, color=PART,
                               connectionstyle="arc3,rad=-0.25"))

    here = os.path.dirname(os.path.abspath(__file__))
    kaydet(figA, "sekilA1_kaydirma", here)
    kaydet(figB, "sekil2_permutasyon", here)


if __name__ == "__main__":
    main()
