"""Print mean and std over seeds for the metrics of one experiment folder.

  python summarize.py step1 rmse_test_s0
  python summarize.py step2 rmse_test_s0 spearman_clean auroc_s1 auroc_s5 u_mean_s0 u_mean_s5
"""
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(exp):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", exp, "*.json")))]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    exp, metrics = sys.argv[1], sys.argv[2:] or ["rmse_test_s0"]
    df = load(exp)
    pd.set_option("display.width", 250)
    order = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]
    for metric in metrics:
        g = df.groupby(["tag", "dataset"])[metric].agg(["mean", "std"])
        cell = g.apply(lambda r: "%.4g ± %.2g" % (r["mean"], r["std"]), axis=1)
        table = cell.unstack("dataset")
        print("\n== %s  (mean ± std over %d seeds)" % (metric, df["seed"].nunique()))
        print(table[[c for c in order if c in table.columns]].to_string())
