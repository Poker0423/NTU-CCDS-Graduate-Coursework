"""Torch-free constants so analysis scripts run on machines without PyTorch."""
NEAR_OOD = ["CIFAR100", "TIN"]
FAR_OOD = ["MNIST", "SVHN", "DTD", "PLACES365"]
ALL_OOD = NEAR_OOD + FAR_OOD
