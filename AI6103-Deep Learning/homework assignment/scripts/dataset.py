"""CSV labels; ordinary loaders deliberately exclude the test split."""
import csv
from pathlib import Path
try:
    from .gpu_guard import bootstrap
except ImportError:
    from gpu_guard import bootstrap
bootstrap()
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    from .config import ROOT, EXPECTED_SPLITS
    from .artifacts import file_hash, fingerprint
except ImportError:
    from config import ROOT, EXPECTED_SPLITS
    from artifacts import file_hash, fingerprint

DATA_ROOT = ROOT / "data"


def label_rows(root, split):
    with (Path(root) / f"{split}_labels.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "breed"]:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        rows = list(reader)
    if any(not r["breed"] or not r["id"] or Path(r["id"]).name != r["id"]
           or "/" in r["id"] or "\\" in r["id"] for r in rows):
        raise ValueError("Invalid CSV ID or label")
    return rows


def class_mapping(root):
    return {name: i for i, name in enumerate(sorted({r["breed"] for r in label_rows(root, "train")}))}


def inspect_dataset(root=DATA_ROOT, expected=EXPECTED_SPLITS, expected_classes=120,
                    readable_per_split=5):
    """Check every file hash, IDs, labels and duplicates; decode a small sample.

    Test metadata/image integrity checks perform no prediction or accuracy evaluation.
    """
    root = Path(root)
    mapping = class_mapping(root)
    if len(mapping) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(mapping)}")
    seen_ids, seen_bytes, inventory, splits, duplicates = set(), {}, [], {}, []
    for split, count in expected.items():
        rows = label_rows(root, split)
        ids = [r["id"] for r in rows]
        if len(rows) != count or len(set(ids)) != count:
            raise ValueError(f"Wrong size or duplicate IDs in {split}: {len(rows)}")
        if seen_ids.intersection(ids):
            raise ValueError("Cross-split duplicate ID")
        seen_ids.update(ids)
        unknown = {r["breed"] for r in rows} - mapping.keys()
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")
        image_dir = root / split / "images"
        actual = {p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"}
        if actual != {f"{i}.jpg" for i in ids}:
            raise ValueError(f"Image/CSV mismatch in {split}")
        for index, row in enumerate(rows):
            path = image_dir / (row["id"] + ".jpg")
            digest = file_hash(path)
            if digest in seen_bytes:
                duplicates.append({"first": seen_bytes[digest], "second": path.relative_to(root).as_posix(),
                                   "cross_split": seen_bytes[digest].split("/")[0] != split})
            else:
                seen_bytes[digest] = path.relative_to(root).as_posix()
            inventory.append([path.relative_to(root).as_posix(), row["breed"], digest])
            if index < readable_per_split:
                with Image.open(path) as image:
                    image.verify()
        splits[split] = {"count": count, "classes": len({r["breed"] for r in rows}),
                         "csv_sha256": file_hash(root / f"{split}_labels.csv"),
                         "readable_sample": min(count, readable_per_split)}
    return {"dataset_fingerprint": fingerprint({"images": inventory, "splits": splits}),
            "splits": splits, "class_mapping": mapping,
            "content_hashes_checked": len(inventory), "no_duplicate_ids": True,
            "duplicate_contents": duplicates,
            "cross_split_duplicate_contents": sum(d["cross_split"] for d in duplicates)}


class DogBreedDataset(Dataset):
    def __init__(self, split, transform=None, label2idx=None, root=DATA_ROOT):
        self.split = split
        self.root = Path(root)
        rows = label_rows(root, split)
        self.label2idx = class_mapping(root) if label2idx is None else label2idx
        self.paths = [self.root / split / "images" / (r["id"] + ".jpg") for r in rows]
        self.labels = [self.label2idx[r["breed"]] for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert("RGB")
            return (self.transform(image) if self.transform else image), self.labels[idx]


def make_transforms(mean, std):
    norm = transforms.Normalize(mean, std)
    train = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomCrop(100, padding=12, fill=0, padding_mode="constant"),
                                transforms.ToTensor(), norm])
    evaluation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor(), norm])
    return train, evaluation


def get_train_val_loaders(mean, std, root=DATA_ROOT, batch_size=128, seed=6103):
    tr_tf, ev_tf = make_transforms(mean, std)
    train = DogBreedDataset("train", tr_tf, root=root)
    val = DogBreedDataset("val", ev_tf, train.label2idx, root=root)
    # Separate generators are checkpointed. Workers=0 makes augmentation RNG resumable.
    tr_gen = torch.Generator().manual_seed(seed)
    va_gen = torch.Generator().manual_seed(seed + 1)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=0, generator=tr_gen),
            DataLoader(val, batch_size, shuffle=False, num_workers=0, generator=va_gen),
            len(train.label2idx))


def get_final_test_loader(mean, std, root=DATA_ROOT, batch_size=128):
    _, ev_tf = make_transforms(mean, std)
    ds = DogBreedDataset("test", ev_tf, class_mapping(root), root=root)
    return DataLoader(ds, batch_size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(6105))


def compute_mean_std(root=DATA_ROOT):
    """Population std over all RGB pixels of resized TRAIN images, in float64."""
    tf = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    ds = DogBreedDataset("train", tf, root=root)
    loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)
    sums, squares, n = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
    for images, _ in loader:
        images = images.double()
        sums += images.sum((0, 2, 3))
        squares += images.square().sum((0, 2, 3))
        n += images.shape[0] * images.shape[2] * images.shape[3]
    mean = sums / n
    std = (squares / n - mean.square()).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()
