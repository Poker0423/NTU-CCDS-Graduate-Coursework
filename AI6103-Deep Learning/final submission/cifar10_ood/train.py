"""Train ResNet-18 on CIFAR-10 with cross-entropy (CE) or HCM, with or without MixUp.

Defaults reproduce the paper / official script setting (Appendix A.2):
    SGD(momentum 0.9, wd 1e-4), lr 0.1, x0.1 at epochs 30/60/90, 100 epochs, batch 64,
    RandomCrop(32, pad 4) + HorizontalFlip, loss x0.1 during the first 2 epochs
    (the undocumented warm-up in HCM_CIFAR10.py; applied to CE too so both methods are
    trained under identical conditions).

Examples
    python train.py --method ce  --seed 0
    python train.py --method hcm --seed 0
    python train.py --method ce  --mix_k 2 --mix_alpha 1.0 --seed 0
    python train.py --method hcm --mix_k 2 --mix_alpha 1.0 --seed 0
    python train.py --method hcm --mix_k 20 --mix_alpha 0.5 --seed 0     # Dirichlet MixUp
    python train.py --method hcm --fake_data --epochs 2 --max_steps 5    # smoke test

Outputs (runs/<run_name>/):
    config.json     all arguments + derived fields (group, run_name)
    log.csv         one row per epoch (lr, train loss, test acc, HCM stats ...)
    ckpt_last.pt    full training state, rewritten every epoch -> automatic resume
    final.pt        {"model": state_dict, "config": ...} after the last epoch
    metrics.json    final test accuracy etc. (read by aggregate_results.py)
A run whose final.pt exists is skipped, so re-submitting a job after the 6-hour limit
simply continues where it stopped.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

from hcm.data import cifar10_loaders
from hcm.losses import hcm_loss, soft_cross_entropy
from hcm.mixup import mix_batch
from hcm.models import build_model
from hcm.utils import CSVLogger, rng_state, save_json, set_rng_state, set_seed


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("method")
    g.add_argument("--method", choices=["ce", "hcm"], required=True)
    g.add_argument("--hcm_loss", choices=["official", "paper"], default="official",
                   help="official = HCM_CIFAR10.py loss; paper = Eq. 3")
    g.add_argument("--loss_power", type=float, default=None,
                   help="p in the loss family (default 0.75 official, 2 paper)")
    g.add_argument("--lambda_norm", type=float, default=0.0, help="weight of the unit-norm term")
    g.add_argument("--mix_target", choices=["soft", "decomp"], default="soft",
                   help="how a MixUp label is split into (R*, d*) for HCM, see hcm/losses.py")
    g = p.add_argument_group("mixup (off when --mix_k 0)")
    g.add_argument("--mix_k", type=int, default=0, help="0 = no MixUp, 2 = classic, >2 = Dirichlet")
    g.add_argument("--mix_alpha", type=float, default=1.0)
    g.add_argument("--mix_prob", type=float, default=1.0, help="fraction of batches that are mixed")
    g.add_argument("--mix_per_sample", action="store_true")
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--milestones", type=int, nargs="*", default=[30, 60, 90])
    g.add_argument("--gamma", type=float, default=0.1)
    g.add_argument("--warmup_mode", choices=["loss_scale", "linear_lr", "none"], default="loss_scale",
                   help="loss_scale = official (loss * factor in the first N epochs)")
    g.add_argument("--warmup_epochs", type=int, default=2)
    g.add_argument("--warmup_factor", type=float, default=0.1)
    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--data_root", default="./data")
    g.add_argument("--out_dir", default="./runs")
    g.add_argument("--run_name", default=None, help="default: built from the settings")
    g.add_argument("--tag", default="", help="suffix appended to the auto run name")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--amp", action="store_true", help="bf16 autocast (faster; off = paper setting)")
    g.add_argument("--deterministic", action="store_true", help="cudnn deterministic (slower)")
    g.add_argument("--overwrite", action="store_true", help="ignore existing final.pt/ckpt")
    g.add_argument("--fake_data", action="store_true", help="random tensors, for smoke tests")
    g.add_argument("--max_steps", type=int, default=0, help=">0: cap steps per epoch (smoke tests)")
    g.add_argument("--print_every", type=int, default=200)
    return p.parse_args(argv)


def fmt_num(x):
    return f"{x:g}"


def make_group_name(a):
    """Seed-independent config name; aggregate_results.py groups runs by it."""
    name = a.method
    if a.method == "hcm":
        if a.hcm_loss != "official":
            name += f"-{a.hcm_loss}"
        if a.loss_power is not None:
            name += f"-p{fmt_num(a.loss_power)}"
        if a.lambda_norm > 0:
            name += f"-lam{fmt_num(a.lambda_norm)}"
    if a.mix_k >= 2:
        name += f"_mix-k{a.mix_k}-a{fmt_num(a.mix_alpha)}"
        if a.mix_prob < 1:
            name += f"-pr{fmt_num(a.mix_prob)}"
        if a.mix_per_sample:
            name += "-ps"
        if a.method == "hcm" and a.mix_target != "soft":
            name += f"-{a.mix_target}"
    if a.lr != 0.1:
        name += f"_lr{fmt_num(a.lr)}"
    if (a.warmup_mode, a.warmup_epochs, a.warmup_factor) != ("loss_scale", 2, 0.1):
        name += f"_wu-{a.warmup_mode}{a.warmup_epochs}" + (
            f"x{fmt_num(a.warmup_factor)}" if a.warmup_mode != "none" else "")
    if a.epochs != 100:
        name += f"_ep{a.epochs}"
    if a.batch != 64:
        name += f"_bs{a.batch}"
    if a.tag:
        name += f"_{a.tag}"
    return name


def lr_at(a, epoch, it, iters_per_epoch):
    """epoch is 1-based. Multi-step decay (+ optional linear LR warm-up)."""
    lr = a.lr * (a.gamma ** sum(epoch > m for m in a.milestones))
    if a.warmup_mode == "linear_lr" and epoch <= a.warmup_epochs:
        frac = ((epoch - 1) * iters_per_epoch + it + 1) / (a.warmup_epochs * iters_per_epoch)
        lr *= a.warmup_factor + (1 - a.warmup_factor) * frac
    return lr


def loss_scale_at(a, epoch):
    return a.warmup_factor if (a.warmup_mode == "loss_scale" and epoch <= a.warmup_epochs) else 1.0


@torch.no_grad()
def evaluate(model, loader, device, method, amp=False):
    model.eval()
    n = correct = correct_dir = 0
    nll = dn_sum = R_sum = u_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(x)
        logits = out["logits"].float()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
        if method == "ce":
            nll += F.cross_entropy(logits, y, reduction="sum").item()
        else:
            d, R = out["d"].float(), out["R"].float()
            dn = d.norm(dim=1)
            correct_dir += (d.argmax(1) == y).sum().item()
            dn_sum += dn.sum().item()
            R_sum += R.sum().item()
            u_sum += (R * (dn - 1).abs()).sum().item()
    res = {"test_acc": 100.0 * correct / n}
    if method == "ce":
        res["test_nll"] = nll / n
    else:
        res.update({"test_acc_dir": 100.0 * correct_dir / n, "mean_dnorm": dn_sum / n,
                    "mean_R": R_sum / n, "mean_u": u_sum / n})
    return res


def main(argv=None):
    a = get_args(argv)
    a.group = make_group_name(a)
    a.run_name = a.run_name or f"{a.group}_s{a.seed}"
    run_dir = os.path.join(a.out_dir, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    final_path, ckpt_path = os.path.join(run_dir, "final.pt"), os.path.join(run_dir, "ckpt_last.pt")
    done = os.path.exists(final_path) or os.path.exists(os.path.join(run_dir, "metrics.json"))
    if done and not a.overwrite:
        print(f"[skip] {a.run_name}: already finished (final.pt / metrics.json exists)")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(a.seed, a.deterministic)
    cfg = dict(vars(a), num_classes=10, device=str(device), torch=torch.__version__,
               command=" ".join(sys.argv))
    save_json(cfg, os.path.join(run_dir, "config.json"))
    print(f"== {a.run_name} on {device} ==")

    train_loader, test_loader = cifar10_loaders(a.data_root, a.batch, a.workers, a.fake_data)
    model = build_model(a.method).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    ipe = min(len(train_loader), a.max_steps) if a.max_steps > 0 else len(train_loader)

    start_epoch, elapsed, resumed = 1, 0.0, False
    if os.path.exists(ckpt_path) and not a.overwrite:
        # load on CPU: RNG states must stay CPU ByteTensors; params are copied to device below
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        set_rng_state(ck["rng"])
        start_epoch, elapsed, resumed = ck["epoch"] + 1, ck["elapsed"], True
        print(f"[resume] from epoch {ck['epoch']}")
    fields = ["epoch", "lr", "train_loss", "loss_d", "loss_r", "loss_norm", "test_acc", "test_nll",
              "test_acc_dir", "mean_dnorm", "mean_R", "mean_u", "epoch_time_s"]
    log = CSVLogger(os.path.join(run_dir, "log.csv"), fields, append=resumed)

    diverged, res = False, {}
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        t0 = time.time()
        tot = cnt = 0.0
        comp_sum = {"loss_d": 0.0, "loss_r": 0.0, "loss_norm": 0.0}
        scale = loss_scale_at(a, epoch)
        for it, (x, y) in enumerate(train_loader):
            if it >= ipe:
                break
            lr = lr_at(a, epoch, it, ipe)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_soft = F.one_hot(y, 10).float()
            if a.mix_k >= 2:
                x, y_soft = mix_batch(x, y_soft, a.mix_k, a.mix_alpha, a.mix_per_sample, a.mix_prob)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda"):
                out = model(x)
            if a.method == "ce":
                loss = soft_cross_entropy(out["logits"].float(), y_soft)
                comps = {}
            else:
                loss, comps = hcm_loss(out["d"].float(), out["R"].float(), y_soft, a.hcm_loss,
                                       a.mix_target, a.loss_power, a.lambda_norm)
            if not torch.isfinite(loss):
                diverged = True
                print(f"[diverged] non-finite loss at epoch {epoch} iter {it}")
                break
            opt.zero_grad(set_to_none=True)
            (scale * loss).backward()
            opt.step()
            tot += loss.item()
            cnt += 1
            for k, v in comps.items():
                comp_sum[k] += v
            if a.print_every and it % a.print_every == 0:
                print(f"ep {epoch:3d} it {it:4d}/{ipe} lr {lr:.4g} loss {loss.item():.4f}", flush=True)
        if diverged:
            break
        res = evaluate(model, test_loader, device, a.method, a.amp)
        dt = time.time() - t0
        elapsed += dt
        row = {"epoch": epoch, "lr": lr, "train_loss": tot / max(cnt, 1), "epoch_time_s": round(dt, 1), **res}
        if a.method == "hcm":
            row.update({k: v / max(cnt, 1) for k, v in comp_sum.items()})
        log.log(row)
        print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "elapsed": elapsed, "rng": rng_state(), "config": cfg}, ckpt_path)
    log.close()
    if not diverged and not res:      # resumed from a checkpoint of the last epoch
        res = evaluate(model, test_loader, device, a.method, a.amp)
        epoch = a.epochs

    metrics ={"run_name": a.run_name, "group": a.group, "method": a.method, "seed": a.seed,
               "mixup": a.mix_k >= 2, "mix_k": a.mix_k, "mix_alpha": a.mix_alpha,
               "diverged": diverged, "train_time_min": round(elapsed / 60, 2)}
    if diverged:
        metrics.update({"epochs_completed": epoch - 1, "test_acc": float("nan")})
    else:
        metrics.update({"epochs_completed": a.epochs, **res})
        torch.save({"model": model.state_dict(), "config": cfg}, final_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)          # keep the run folder small; final.pt is enough
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    print("[done]", metrics)


if __name__ == "__main__":
    main()
