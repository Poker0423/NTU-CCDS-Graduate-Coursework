"""Run a trained model once over ID / OOD data and save everything other roles need.

    python export_outputs.py --runs runs/hcm_s0 runs/ce_s0 [--oods CIFAR100 SVHN ...]
    python export_outputs.py --runs runs/*            # all finished runs

For each run writes runs/<run>/outputs/<split>.npz with
    labels  (N,)      int, -1 for OOD
    logits  (N, C)    CE: fc output; HCM: R * d
    feat    (N, 512)  penultimate features (for MDS / KNN / ViM ...)
    HCM only: d (N, C), R (N,)
Splits: "train" (CIFAR-10 train set, test-time transform, no augmentation - needed by
MDS class means / KNN feature bank), "test" (CIFAR-10 test = ID), and one file per OOD
set (CIFAR100, TIN, MNIST, SVHN, DTD, PLACES365). OOD sets that cannot be found are
skipped with a warning. Existing files are skipped unless --overwrite.
"""
import argparse
import glob
import os

import numpy as np
import torch

from hcm.data import ALL_OOD, cifar10_datasets, eval_loader, ood_dataset
from hcm.models import load_model


@torch.no_grad()
def run_split(model, loader, device, feat_dtype):
    keys = ["logits", "feat"] + (["d", "R"] if model.method == "hcm" else [])
    buf = {k: [] for k in keys}
    labels = []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        for k in keys:
            buf[k].append(out[k].float().cpu())
        labels.append(torch.as_tensor(y))
    res = {k: torch.cat(v).numpy() for k, v in buf.items()}
    res["feat"] = res["feat"].astype(feat_dtype)
    res["labels"] = torch.cat(labels).numpy().astype(np.int64)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (globs ok)")
    p.add_argument("--splits", nargs="+", default=["train", "test"] + ALL_OOD)
    p.add_argument("--oods", nargs="+", default=None, help="shortcut: test + these OOD sets (+train)")
    p.add_argument("--no_train", action="store_true", help="skip the 50k train-feature export")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--tin_dir", default=None)
    p.add_argument("--places_dir", default=None)
    p.add_argument("--feat_dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fake_data", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    splits = (["train", "test"] + a.oods) if a.oods else a.splits
    if a.no_train:
        splits = [s for s in splits if s != "train"]
    runs = sorted({r for pat in a.runs for r in glob.glob(pat)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_cache = {}

    def get_ds(split):
        if split not in datasets_cache:
            if split in ("train", "test"):
                tr, te = cifar10_datasets(a.data_root, a.fake_data, train_aug=False)
                datasets_cache["train"], datasets_cache["test"] = tr, te
            else:
                datasets_cache[split] = ood_dataset(split, a.data_root, a.tin_dir, a.places_dir, a.fake_data)
        return datasets_cache[split]

    for run in runs:
        ck = os.path.join(run, "final.pt")
        if not os.path.exists(ck):
            print(f"[skip] {run}: no final.pt")
            continue
        model, cfg = load_model(ck, device)
        model.to(device)
        out_dir = os.path.join(run, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            path = os.path.join(out_dir, f"{split}.npz")
            if os.path.exists(path) and not a.overwrite:
                continue
            try:
                ds = get_ds(split)
            except Exception as e:                      # missing manual datasets etc.
                print(f"[warn] {split}: {e}")
                continue
            res = run_split(model, eval_loader(ds, a.batch, a.workers), device, a.feat_dtype)
            np.savez_compressed(path, **res)
            msg = f"{run} {split}: {len(res['labels'])} samples"
            if split == "test":
                msg += f", acc {100 * (res['logits'].argmax(1) == res['labels']).mean():.2f}%"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
