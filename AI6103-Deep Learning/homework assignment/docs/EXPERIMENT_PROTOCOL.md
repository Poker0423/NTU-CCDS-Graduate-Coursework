# Experiment and evaluation protocol

The updated PDF and instructor correction are authoritative. Configuration lives in
`scripts/config.py`; train/val loaders, training and selection have one maintained implementation.

| Section | Conditions | LR | Epochs | Schedule | WD | Mixup |
|---|---|---|---:|---|---|---|
| 3 | sec3_lr0.5 / sec3_lr0.05 / sec3_lr0.001 | 0.5 / 0.05 / 0.001 | 15 | Constant | 0 | Off |
| 4 | sec4_constant / sec4_cosine | Section 3 winner | 300 each | Constant / cosine | 0 | Off |
| 5 | sec5_wd5e-4 / sec5_wd1e-4 | Section 3 winner | 300 each | Cosine | 5e-4 / 1e-4 | Off |
| 6 | sec6_mixup | Matching chosen baseline | 300 | Baseline's schedule | Baseline's WD | alpha=0.2 |

All runs: random EfficientNetV2-S (`weights=None`), 120 outputs, SGD, momentum 0.9,
batch 128, 100x100 input, zero-padded random crop and flip p=0.5, no pretrained features.
No AMP, gradient accumulation, or altered batch sizes are introduced. Workers=0 and
separate checkpointed DataLoader generators favor correct epoch-boundary restoration.
All conditions start fresh with matched seed 6103; each loader is rebuilt per run.

## Fixed selection rules

Section 3 requires all three 15-epoch records to be complete, compatible and comparable.
Minimize final training CE only. Exact ties use candidate order [0.5, 0.05, 0.001].
The report separately identifies winners for training loss, training accuracy,
validation loss and validation accuracy. Test metrics never influence selection.

Section 4 must finish both constant and cosine experiments before Section 5 starts;
Section 5 always uses cosine regardless of Section 4's validation winner.
After Sections 4 and 5, choose the Section 6 no-Mixup baseline among all four 300-epoch
conditions by lowest final validation CE. Exact ties follow registry order: constant,
cosine without WD, cosine WD=5e-4, cosine WD=1e-4. This resolves “best setup discovered
so far” using an explicit rule; it does not assume a WD or cosine winner before observation.
Section 6 inherits the winner's LR/schedule/WD and changes only Mixup. If constant or
WD=0 wins, the corresponding Section 4 run is the matching baseline, without redundant retraining.

The prescribed Mixup model after epoch 300 is the final model. This choice is declared
before hold-out evaluation; baseline performance does not justify inspecting two test accuracies.
No best-validation-epoch checkpoint replaces the required last-epoch model.

## Metric and scheduler definitions

Batch mean CE is multiplied by batch size before summation; accuracy counts samples,
including the last partial batch. Standard train metrics use augmented training examples.
Mixup train CE is lambda*CE(y_a)+(1-lambda)*CE(y_b), and train accuracy uses the same
soft-label weighting. Validation metrics are conventional CE/top-1 accuracy on clean
validation transforms. Mixup and standard train metrics are not directly interchangeable.
CE histories exclude the weight-decay penalty; PyTorch SGD applies decay through its
momentum update. Validation provides the primary regularization comparison.

For cosine, eta_t=eta_0*(1+cos(pi*t/300))/2, eta_min=0. Epoch e uses eta_(e-1).
Step the scheduler once after training/validation, so eta_300=0 after epoch 300.
The used and next LRs are recorded and checked against this timing convention.

## Artifacts, verification and resume

Each experiment has `results/<experiment_id>/record.json` and two alternating
`checkpoint_0.pt` / `checkpoint_1.pt` generations. Each epoch commits a full checkpoint
atomically, hashes it, then atomically commits the lightweight record. At least the previous
committed checkpoint survives a crash while writing the next generation. Epoch 0 is saved
before the first training batch. A mid-epoch interrupt restores the last completed epoch
and repeats the unfinished epoch; it does not claim minibatch-exact continuation.

Checkpoints contain model, optimizer, scheduler, history, last epoch, configuration,
fingerprint, Python/NumPy/torch and accelerator RNG state, and loader generator state.
Configuration fingerprints include dataset contents, normalization manifest and maintained
scientific source code. Exact recorded environment matching is required for resume;
all conditions must have matching recorded environments for controlled selection.
Moving identical files to another path is allowed; changing the experiment data/code/settings
is rejected. Configuration mismatches and orphan files are preserved for review, not overwritten.

An exclusive per-run lock prevents concurrent writes. After an abrupt termination, a stale
lock can remain. Confirm that the originating process is stopped, then remove only that
experiment's `.run.lock` and use `--resume`. Never clear a lock for a live run. Completed
compatible runs are skipped only after history, configuration, checkpoint checksum and
checkpoint contents are verified. Separate output roots preserve incompatible research.

Checkpoints contain Python/NumPy RNG objects and are loaded with weights_only=False;
use only locally generated/trusted project checkpoints. Do not import arbitrary .pt files.
Atomic replace/fsync improves local durability; cloud-sync/Drive consistency is a separate
limitation. Keep a separate checkpoint archive for recovery if necessary.

## Final hold-out guard

Training has no test loader parameter and validation rejects loaders marked `test`.
After all eight conditions are verified, `freeze` writes an immutable manifest binding
dataset/code/stats and hashes of every completed record. Further training under that
output root is blocked. `evaluate-final --authorize-final-test` additionally verifies
the frozen bundle and makes a durable started marker BEFORE opening the test loader.
An existing started marker blocks repeats, even if evaluation was interrupted. In that
case inspect the failure and request a deliberate recovery decision; automatic retry is
not safe for one-time evaluation. Phase A does not invoke this operation on real data.

Dataset image/label/hash inspection and analytical Beta plotting do not evaluate test
performance. Supplied cross-split duplicates are recorded rather than removed, to retain
the assignment's exact splits. They must be acknowledged when interpreting final accuracy.

PyTorch documents that reproducibility is not guaranteed across platforms/releases:
https://docs.pytorch.org/docs/stable/notes/randomness.html
Scheduler reference: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
