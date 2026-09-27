# -*- coding: utf-8 -*-
"""
Analiz JSON'larını okuyan tek yer.

ŞEMAYI VARSAYMA, PROBE ET
    Şekil betiklerinin ilk sürümü `{"arms": {kol: {yontem: {...}}}}` varsaydı.
    Gerçek şema başka: kök doğrudan kol adları, ve hücre anahtarı `yontem`
    değil `yontem|model`. Üç betik de sessizce boş grafik üretecekti — veri
    yok diye değil, yanlış yerde aradıkları için. Okuma tek yerde toplandı ki
    şema bir kez doğrulansın.

    leakage_inflation.json : {kol: {"yontem|model": {"H1": {"gain_pct", ...}}}}
    deployment_gap*.json   : {"yontem|model": {"H1": {reported, deployed,
                                                      persistence, ratio}}}
"""
import json, os
import numpy as np

KOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "figures", "analysis")
YONTEM = ["dwt", "vmd", "emd", "eemd", "ceemdan"]
UFUK = [1, 8, 16, 24]
KOL = ["leaky", "partition", "causal"]


def yol(ad):
    return os.path.normpath(os.path.join(KOK, ad))


def var(ad):
    return os.path.exists(yol(ad))


def sisme():
    """(kol, yontem, ufuk) -> modeller uzerinden medyan kazanc yuzdesi."""
    d = json.load(open(yol("leakage_inflation.json"), encoding="utf-8"))
    out = {}
    for kol in KOL:
        if kol not in d:
            continue
        M = np.full((len(YONTEM), len(UFUK)), np.nan)
        for i, m in enumerate(YONTEM):
            for j, h in enumerate(UFUK):
                v = [c[f"H{h}"]["gain_pct"] for k, c in d[kol].items()
                     if k.split("|")[0] == m and f"H{h}" in c
                     and c[f"H{h}"].get("gain_pct") is not None]
                if v:
                    M[i, j] = float(np.median(v))
        out[kol] = M
    return out


def teslim(ad):
    """[(etiket, bildirilen, teslim, kalicilik)] — h = 1."""
    d = json.load(open(yol(ad), encoding="utf-8"))
    out = []
    for k, c in sorted(d.items()):
        g = c.get("H1")
        if g:
            out.append((k, g["reported"], g["deployed"], g["persistence"]))
    return out
