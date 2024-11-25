from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Dict, Any

def Transform() -> transforms.Compose:
    """
    Create a transformation for datasets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform



def load_cifar10() -> Tuple[Dataset, Dataset]:
    """
    Load the CIFAR-10 dataset.
    """
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR10(
        "./datasets/cifar-10/", download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR10(
        "./datasets/cifar-10/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_cifar100() -> Tuple[Dataset, Dataset]:
    """
    Load the CIFAR-100 dataset.
    """
    transform = Transform()
    # Download and load the training data
    trainset = datasets.CIFAR100(
        "./datasets/cifar-100/", download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR100(
        "./datasets/cifar-100/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_mnist() -> Tuple[Dataset, Dataset]:
    """
    Load the MNIST dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # Download and load the training dat
    trainset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=True, transform=transform
    )
    testset = datasets.MNIST(
        "./datasets/MNIST/", download=True, train=False, transform=transform
    )
    return trainset, testset


def load_dataset(dataset_name: str) -> Dataset:
    """
    Load the dataset specified by the dataset_name.

    Args:
        dataset_name (str): The name of the dataset to load
    """
    if dataset_name == "cifar10":
        return load_cifar10()
    elif dataset_name == "cifar100":
        return load_cifar100()
    elif dataset_name == "mnist":
        return load_mnist()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_cifar10(32, 0.2)
    print("Train size: ", len(train_loader) * 32)
    print("Val size: ", len(val_loader) * 32)
    print("Test size: ", len(test_loader) * 32)
