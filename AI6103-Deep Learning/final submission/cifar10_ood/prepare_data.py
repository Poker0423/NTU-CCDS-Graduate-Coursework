"""Download every dataset torchvision can fetch (run once, on a node with internet).

    python prepare_data.py --data_root ./data

TinyImageNet and Places365 need a manual download - see README "Data".
"""
import argparse
import os

from torchvision import datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    a = p.parse_args()
    r = a.data_root
    datasets.CIFAR10(r, train=True, download=True)
    datasets.CIFAR10(r, train=False, download=True)
    datasets.CIFAR100(r, train=False, download=True)
    datasets.SVHN(os.path.join(r, "svhn"), split="test", download=True)
    datasets.MNIST(r, train=False, download=True)
    datasets.DTD(r, split="test", download=True)
    for name, path in (("TinyImageNet", "tiny-imagenet-200/test/images"), ("Places365", "places365")):
        full = os.path.join(r, path)
        print(f"{name:12s}: {'found' if os.path.isdir(full) else 'MISSING (manual download)'} -> {full}")
    print("done")


if __name__ == "__main__":
    main()
