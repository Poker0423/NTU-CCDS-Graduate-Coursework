"""HCM OOD detection from exported outputs (no GPU needed).

    python eval_hcm_ood.py --runs runs/hcm*            # needs export_outputs.py first

For every HCM run and every uncertainty formula in hcm/scores.py, computes AUROC,
AUPR-in/out and FPR@95TPR (OpenOOD convention, see hcm/metrics.py) for each OOD set,
plus near / far / overall averages. Writes runs/<run>/ood_hcm.json and prints a table.
New score formulas (report Exp C) can be added to SCORES in hcm/scores.py; then re-run this file.
"""
import argparse
import glob
import json
import os

import numpy as np

from hcm.constants import ALL_OOD, FAR_OOD, NEAR_OOD
from hcm.metrics import ood_metrics
from hcm.scores import SCORES


def eval_run(run, score_names):
    out_dir = os.path.join(run, "outputs")
    test = os.path.join(out_dir, "test.npz")
    if not os.path.exists(test):
        return None
    idd = np.load(test)
    if "d" not in idd:
        return None                                  # CE run
    results = {}
    for sname in score_names:
        f = SCORES[sname]
        id_u = f(idd["d"], idd["R"])
        per = {}
        for o in ALL_OOD:
            path = os.path.join(out_dir, f"{o}.npz")
            if os.path.exists(path):
                z = np.load(path)
                per[o] = ood_metrics(id_u, f(z["d"], z["R"]))
        for grp, names in (("near", NEAR_OOD), ("far", FAR_OOD), ("avg", ALL_OOD)):
            have = [per[n] for n in names if n in per]
            if have:
                per[f"_{grp}"] = {m: float(np.mean([h[m] for h in have])) for m in have[0]}
                per[f"_{grp}"]["n_sets"] = len(have)
        results[sname] = per
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--scores", nargs="+", default=list(SCORES))
    a = p.parse_args()
    for run in sorted({r for pat in a.runs for r in glob.glob(pat)}):
        res = eval_run(run, a.scores)
        if res is None:
            continue
        with open(os.path.join(run, "ood_hcm.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n== {os.path.basename(run)} ==   AUROC% / FPR@95% (OpenOOD convention)")
        cols = [c for c in ALL_OOD + ["_near", "_far", "_avg"] if c in res[a.scores[0]]]
        print(f"{'score':16s}" + "".join(f"{c.strip('_'):>16s}" for c in cols))
        for s, per in res.items():
            print(f"{s:16s}" + "".join(f"{100*per[c]['auroc']:8.2f}/{100*per[c]['fpr95']:6.2f} " for c in cols))


if __name__ == "__main__":
    main()
