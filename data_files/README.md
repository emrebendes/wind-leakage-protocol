# data_files

This directory is intentionally empty in the repository.

The pipeline reads one file, `data_files/all_station_data.npy`, through `src/config.py::load_series()`. It is a NumPy array of shape `(8, n_hours)` with hourly mean wind speed in m/s, one row per station in the order used in the paper (Acıgöl, Avanos, Derinkuyu, Gülşehir, Hacıbektaş, Kozaklı, Nevşehir, Ürgüp). `config.SERIES_END = 50718` truncates every row to the first 50,718 hours (1 January 2020 – 14 October 2025). The array was saved with `allow_pickle=True` and requires NumPy 2.

The records are third-party data of the Turkish State Meteorological Service (MGM, https://www.mgm.gov.tr) and cannot be redistributed by the authors. They can be requested from MGM. Any hourly record with the same layout can be used to run the pipeline on other stations.
