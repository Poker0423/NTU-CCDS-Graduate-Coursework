"""CIFAR-10 (ID) and OOD test-set loaders.

Transforms and normalisation constants are copied from the official HCM repo so that our
numbers are comparable with theirs. All roles should load data through this module so
that every method sees exactly the same images.

OOD sets (names used everywhere in this package):
    near-OOD : CIFAR100, TIN (TinyImageNet)
    far-OOD  : MNIST, SVHN, DTD (Textures), PLACES365
CIFAR100 / SVHN / MNIST / DTD are downloaded by torchvision. TIN and Places365 must be
downloaded manually (see README) and pointed to with --tin_dir / --places_dir; any folder
of images works (searched recursively), e.g. tiny-imagenet-200/test/images or
places365 val_256.
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)          # as in the official script

from hcm.constants import ALL_OOD, FAR_OOD, NEAR_OOD  # noqa: F401
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def tf_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def tf_test():
    return transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def tf_test_gray():
    return transforms.Compose([
        transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


class ImageDirDataset(Dataset):
    """All images under `root` (recursive), label -1."""

    def __init__(self, root, transform=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"OOD image folder not found: {root}")
        self.files = sorted(
            os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
            if f.lower().endswith(IMG_EXT))
        if not self.files:
            raise FileNotFoundError(f"No images under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return (self.transform(img) if self.transform else img), -1


class _FakeCIFAR(Dataset):
    """Random tensors for smoke tests (no download needed)."""

    def __init__(self, n, train, seed=0):
        g = torch.Generator().manual_seed(seed + (0 if train else 1))
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _loader(ds, batch, shuffle, workers, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=workers > 0)


def cifar10_datasets(root="./data", fake=False, train_aug=True):
    if fake:
        return _FakeCIFAR(512, True), _FakeCIFAR(256, False)
    tr = datasets.CIFAR10(root, train=True, download=True,
                          transform=tf_train() if train_aug else tf_test())
    te = datasets.CIFAR10(root, train=False, download=True, transform=tf_test())
    return tr, te


def cifar10_loaders(root="./data", batch=64, workers=4, fake=False):
    tr, te = cifar10_datasets(root, fake)
    return _loader(tr, batch, True, workers), _loader(te, 256, False, workers)


def ood_dataset(name, root="./data", tin_dir=None, places_dir=None, fake=False):
    name = name.upper()
    if fake:
        return _FakeCIFAR(128, False, seed=hash(name) % 1000)
    tf = tf_test()
    if name == "CIFAR100":
        return datasets.CIFAR100(root, train=False, download=True, transform=tf)
    if name == "SVHN":
        return datasets.SVHN(os.path.join(root, "svhn"), split="test", download=True, transform=tf)
    if name in ("DTD", "TEXTURE", "TEXTURES"):
        return datasets.DTD(root, split="test", download=True, transform=tf)
    if name == "MNIST":
        return datasets.MNIST(root, train=False, download=True, transform=tf_test_gray())
    if name in ("TIN", "TINYIMAGENET"):
        return ImageDirDataset(tin_dir or os.path.join(root, "tiny-imagenet-200/test/images"), tf)
    if name in ("PLACES365", "PLACE", "PLACES"):
        return ImageDirDataset(places_dir or os.path.join(root, "places365"), tf)
    raise ValueError(f"Unknown OOD dataset {name}")


def eval_loader(ds, batch=256, workers=4, max_n=None):
    if max_n is not None and len(ds) > max_n:
        ds = Subset(ds, list(range(max_n)))
    return _loader(ds, batch, False, workers)
