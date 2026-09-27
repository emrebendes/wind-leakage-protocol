#!/usr/bin/env python
"""Bütün şekilleri üret. Verisi olmayanlar atlanır ve niçin atlandığını yazar."""
import importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODUL = ["sekil1_uc_rejim", "sekil2_mekanizma", "sekil3_calisma_alani",
         "sekil4_sisme_isi", "sekil5_bildirilen_teslim",
         "sekil6_buyukluk_erisim"]
hazir = atlanan = 0
for m in MODUL:
    print(f"\n{m}")
    rc = importlib.import_module(m).main()
    if rc:
        atlanan += 1
    else:
        hazir += 1
print(f"\n{hazir} sekil uretildi, {atlanan} sekil veri bekliyor.")
