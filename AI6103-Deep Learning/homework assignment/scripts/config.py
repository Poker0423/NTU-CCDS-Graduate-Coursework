"""Official assignment settings. Paths are runtime arguments."""
from dataclasses import asdict, dataclass
from pathlib import Path
try:
    from .artifacts import fingerprint
except ImportError:
    from artifacts import fingerprint

ROOT = Path(__file__).resolve().parents[1]
SECTION3_LRS = [0.5, 0.05, 0.001]
EXPECTED_SPLITS = {"train": 7155, "val": 1533, "test": 1534}
DEPENDENCIES = ["torch", "torchvision", "numpy", "Pillow", "scipy", "matplotlib", "nbformat", "pytest"]


@dataclass(frozen=True)
class ExperimentConfig:
    section: int
    experiment_id: str
    lr: float
    epochs: int
    dataset_fingerprint: str
    stats_fingerprint: str
    implementation_fingerprint: str
    batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: str = "constant"
    mixup: bool = False
    mixup_alpha: float = 0.2
    seed: int = 6103
    num_classes: int = 120
    model: str = "efficientnet_v2_s"
    weights: None = None
    optimizer: str = "SGD"
    image_size: int = 100
    flip_probability: float = 0.5
    crop_padding: int = 12
    num_workers: int = 0
    deterministic: bool = True
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return fingerprint(self.to_dict())

    def validate(self):
        checks = [self.section in (3, 4, 5, 6), self.batch_size == 128 and self.momentum == 0.9,
                  self.num_classes == 120 and self.num_workers == 0,
                  self.model == "efficientnet_v2_s" and self.weights is None,
                  self.optimizer == "SGD" and self.image_size == 100,
                  self.flip_probability == 0.5 and self.crop_padding == 12,
                  self.scheduler in ("constant", "cosine"), self.lr in SECTION3_LRS,
                  self.epochs == (15 if self.section == 3 else 300), self.deterministic]
        if self.section == 3:
            checks += [self.weight_decay == 0 and self.scheduler == "constant" and not self.mixup]
        if self.section == 4:
            checks += [self.weight_decay == 0 and not self.mixup]
        if self.section == 5:
            checks += [self.weight_decay in (5e-4, 1e-4) and self.scheduler == "cosine" and not self.mixup]
        if self.section == 6:
            checks += [self.mixup and self.mixup_alpha == 0.2 and self.weight_decay in (0, 5e-4, 1e-4)]
        if not all(checks):
            raise ValueError("Configuration violates the official assignment protocol")
        return self


def section3(context):
    return [ExperimentConfig(3, f"sec3_lr{lr:g}", lr, 15, **context).validate()
            for lr in SECTION3_LRS]


def section4(context, lr):
    return [ExperimentConfig(4, f"sec4_{s}", lr, 300, scheduler=s, **context).validate()
            for s in ("constant", "cosine")]


def section5(context, lr):
    return [ExperimentConfig(5, name, lr, 300, scheduler="cosine", weight_decay=wd,
                             **context).validate()
            for name, wd in (("sec5_wd5e-4", 5e-4), ("sec5_wd1e-4", 1e-4))]


def section6(context, baseline):
    return ExperimentConfig(6, "sec6_mixup", baseline.lr, 300,
                            scheduler=baseline.scheduler, weight_decay=baseline.weight_decay,
                            mixup=True, **context).validate()
