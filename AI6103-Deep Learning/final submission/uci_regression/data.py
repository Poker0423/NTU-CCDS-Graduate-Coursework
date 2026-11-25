"""UCI regression datasets for the HCM experiments.

Sources (checked 2026-11-17):
  wine      OpenML 287    wine_quality (red + white, 6497 rows), target `quality`
  concrete  OpenML 4353   Concrete_Data (1030 rows), target = last column
  energy    OpenML 44960  energy_efficiency (768 rows), target `heating_load`
            (OpenML 1472 has the same inputs but binned, categorical targets: do not use it)
  kin8nm    OpenML 189    kin8nm (8192 rows), target `y`  -- the ID the HCM paper cites
  yacht     OpenML 42370  yacht_hydrodynamics (308 rows), target = last column
  power     UCI 294       Combined Cycle Power Plant (9568 rows), target `PE`
            (not found on OpenML by name; taken from the UCI archive instead)
  energy2   UCI 242       Energy Efficiency, original file with BOTH targets `Y1`, `Y2`
            (extra dataset, not in the paper's six: a true 2-D target, no (y, y) duplication)

Every dataset is cached as data_cache/<name>.npz after the first load.
"""
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")

OPENML = {
    "wine": (287, ["quality"]),
    "concrete": (4353, None),      # None -> last column
    "energy": (44960, ["heating_load"]),
    "kin8nm": (189, ["y"]),
    "yacht": (42370, None),
}
POWER_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
ENERGY2_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
DATASETS = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]   # the paper's six
ALL = DATASETS + ["energy2"]


def _to_float(frame):
    return frame.apply(lambda c: pd.to_numeric(c.astype(str), errors="raise")).astype(np.float64)


def _fetch_openml(name):
    from sklearn.datasets import fetch_openml
    did, targets = OPENML[name]
    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto",
                         data_home=os.path.join(CACHE, "openml"))
    frame = _to_float(bunch.frame)
    if targets is None:
        targets = [frame.columns[-1]]
    features = [c for c in frame.columns if c not in targets]
    return frame[features].values, frame[targets].values, features, targets


def _fetch_power():
    raw = os.path.join(CACHE, "raw_power")
    os.makedirs(raw, exist_ok=True)
    zpath = os.path.join(raw, "ccpp.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(POWER_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        with z.open("CCPP/Folds5x2_pp.xlsx") as f:
            frame = pd.read_excel(f, sheet_name=0, engine="openpyxl")
    frame = _to_float(frame)
    targets = ["PE"]
    features = [c for c in frame.columns if c not in targets]
    return frame[features].values, frame[targets].values, features, targets


def _fetch_energy2():
    raw = os.path.join(CACHE, "raw_energy2")
    os.makedirs(raw, exist_ok=True)
    zpath = os.path.join(raw, "energy.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(ENERGY2_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        with z.open("ENB2012_data.xlsx") as f:
            frame = pd.read_excel(f, sheet_name=0, engine="openpyxl")
    frame = _to_float(frame)
    targets = ["Y1", "Y2"]
    features = [c for c in frame.columns if c not in targets]
    return frame[features].values, frame[targets].values, features, targets


def load(name):
    """Return X (n, p) and Y (n, k) as float64 arrays, in original units."""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return d["X"], d["Y"]
    os.makedirs(CACHE, exist_ok=True)
    fetch = {"power": _fetch_power, "energy2": _fetch_energy2}.get(name, lambda: _fetch_openml(name))
    X, Y, features, targets = fetch()
    assert np.isfinite(X).all() and np.isfinite(Y).all(), name
    np.savez(path, X=X, Y=Y, features=np.array(features), targets=np.array(targets))
    return X, Y


def split(n, seed):
    """80/10/10 train/val/test split, as in the HCM paper (appendix A.1)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr, n_va = int(round(0.8 * n)), int(round(0.1 * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


if __name__ == "__main__":
    for name in ALL:
        X, Y = load(name)
        print("%-9s X=%-11s Y=%-9s  y min/mean/max = %s / %s / %s" % (
            name, X.shape, Y.shape, np.round(Y.min(0), 3), np.round(Y.mean(0), 3), np.round(Y.max(0), 3)))
