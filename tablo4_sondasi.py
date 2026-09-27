# -*- coding: utf-8 -*-
"""Tablo 4 — permütasyon sondası (yeniden üretim).

Kullanım:  python tablo4_sondasi.py dwt vmd [--t 46110 --station 0 --W 512 --trials 3]

Ne ölçülür: t orijininde üç kolun (tam-seri / bölüm bazlı / origin bazlı) öznitelik
matrisi F(t) (K × D) hesaplanır. Sonra seri iki biçimde bozulur ve F(t) yeniden
hesaplanır:
  (a) geleceği permüte et : x[t+1 : n] değerleri kendi aralarında rastgele sıralanır
  (b) eğitimi permüte et  : x[0 : train_end] değerleri kendi aralarında rastgele sıralanır
Rapor edilen sayı max |F'(t) − F(t)|, `trials` deneme üzerinden en büyüğü. Aynı permütasyon
üç kola da uygulanır (deneme başına sabit tohum), böylece kollar karşılaştırılabilir.
Parametreler decomp_params/W{W}/{method}.json'dan (ayarlanmış değerler).
VMD tam seri ayrıştırması ~6,5 GB RAM ister; TRUBA'da koşun.
"""
import sys, json, os, argparse, time
import numpy as np
BURASI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BURASI, "src"))
from config import load_series, PARAMS_DIR
from decompositions import get_decomposer
from causal_features import get_builder, split_bounds

ap = argparse.ArgumentParser()
ap.add_argument("methods", nargs="+")
ap.add_argument("--t", type=int, default=46110)
ap.add_argument("--station", type=int, default=0)
ap.add_argument("--W", type=int, default=512)
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()

x = load_series()[a.station].astype(float)
n = len(x); t = a.t; W = a.W
tr_end, va_end = split_bounds(n)
print(f"n={n} train_end={tr_end} val_end={va_end} t={t} station={a.station} W={W} trials={a.trials}")
# deneme başına aynı permütasyonlar (bütün kollar ve yöntemler için)
perms = []
for k in range(a.trials):
    rng = np.random.RandomState(1000 + k)
    perms.append((rng.permutation(n - (t + 1)), rng.permutation(tr_end)))

for method in a.methods:
    adaylar = [os.path.join(PARAMS_DIR, f"W{W}", f"{method}.json"),
               os.path.join(BURASI, "..", "decomp_params", f"W{W}", f"{method}.json")]
    path = next((p for p in adaylar if os.path.exists(p)), None)
    params = json.load(open(path)) if path else None
    print(f"\n== {method}  params={params}  ({'ayarlanmış' if params else 'VARSAYILAN'})")
    for arm in ("leaky", "partition", "causal"):
        t0 = time.time()
        def feat(series):
            b = get_builder(arm, get_decomposer(method, params), W=W)
            return np.asarray(b.at_origin(series, t), dtype=float)
        base = feat(x)
        fut = trn = 0.0
        for pf, pt in perms:
            x2 = x.copy(); x2[t + 1:] = x[t + 1:][pf]
            fut = max(fut, float(np.max(np.abs(feat(x2) - base))))
            x3 = x.copy(); x3[:tr_end] = x[:tr_end][pt]
            trn = max(trn, float(np.max(np.abs(feat(x3) - base))))
        print(f"  {arm:10s} geleceği permüte et {fut:.2e}   eğitimi permüte et {trn:.2e}   "
              f"(K×D={base.shape}, {time.time() - t0:.0f} s)")
