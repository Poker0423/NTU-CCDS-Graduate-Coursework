"""Run a grid of training runs in parallel (one CPU thread per run).

Example:
  python run_grid.py --exp step1 --methods mlp --target_modes standardize --lrs 1e-4 1e-3
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import data
from train import make_cfg, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["hcm"])
    ap.add_argument("--target_modes", nargs="+", default=["raw"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0])
    ap.add_argument("--warmups", nargs="+", type=int, default=[0])
    ap.add_argument("--branches", nargs="+", default=["shared"],
                    help="only used by hcmp and rhcm: shared | detach | own")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip runs whose .json already exists")
    a = ap.parse_args()

    grid = itertools.product(a.datasets, a.seeds, a.methods, a.target_modes, a.lrs, a.lams,
                             a.warmups, a.branches)
    cfgs = [make_cfg(exp=a.exp, dataset=d, seed=s, method=m, target_mode=t, lr=lr, lam=lam,
                     warmup_epochs=w, epochs=a.epochs, batch=a.batch, branch=b)
            for d, s, m, t, lr, lam, w, b in grid
            if m in ("hcmp", "rhcm") or b == a.branches[0]]
    if a.resume:
        here = os.path.dirname(os.path.abspath(__file__))
        cfgs = [c for c in cfgs if not os.path.exists(os.path.join(
            here, "results", c["exp"], "%s_%s_s%d.json" % (c["dataset"], c["tag"], c["seed"])))]
    print("%d runs, %d workers" % (len(cfgs), a.jobs))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run, c) for c in cfgs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            print("[%d/%d] %-9s %-28s seed %d  rmse_test %.4g  (%.0fs)" % (
                i, len(cfgs), r["dataset"], r["tag"], r["seed"], r["rmse_test_s0"], r["seconds"]))
