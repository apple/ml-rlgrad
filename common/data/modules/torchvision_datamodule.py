
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch.utils.data
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms

from common.data.cifar100.cifar100_coarse import CIFAR100Coarse
from .datamodule import DataModule


class TorchvisionDataModule(DataModule):
    SUPPORTED_DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "cifar100coarse", "svhn"]
    DATASET_TO_PER_CHANNEL_MEAN = {
        "mnist": [0.1307],
        "fmnist": [0.286],
        "cifar10": [0.491, 0.482, 0.446],
        "cifar100": [0.507, 0.486, 0.44],
        "cifar100coarse": [0.507, 0.486, 0.44],
        "svhn": [0.437, 0.443, 0.472]
    }
    DATASET_TO_PER_CHANNEL_STD = {
        "mnist": [0.3081],
        "fmnist": [0.353],
        "cifar10": [0.247, 0.243, 0.261],
        "cifar100": [0.267, 0.256, 0.276],
        "cifar100coarse": [0.267, 0.256, 0.276],
        "svhn": [0.198, 0.201, 0.197]
    }

    def __init__(self, dataset_name: str, data_dir: str = './data', num_train_samples: int = -1, batch_size: int = 32, num_workers: int = 0,
                 split_random_state: int = -1):
        super().__init__()
        self.dataset_name = dataset_name
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: '{self.dataset_name}'")

        self.data_dir = data_dir
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_random_state = split_random_state

    def setup(self):
        self.train_dataset = self.__get_dataset(train=True)
        self.test_dataset = self.__get_dataset(train=False)
        self.val_dataset = self.test_dataset

        self.input_dims = tuple(self.train_dataset[0][0].shape)
        self.num_classes = len(self.train_dataset.classes)

        if 0 < self.num_train_samples < len(self.train_dataset):
            self.train_dataset = self.__subsample_dataset(self.train_dataset, self.num_train_samples)

    def __get_dataset(self, train: bool):
        transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=self.DATASET_TO_PER_CHANNEL_MEAN[self.dataset_name],
                                                                         std=self.DATASET_TO_PER_CHANNEL_STD[self.dataset_name])])
        if self.dataset_name == "mnist":
            return torchvision.datasets.MNIST(self.data_dir, train=train, download=True, transform=transform)
        elif self.dataset_name == "fmnist":
            return torchvision.datasets.FashionMNIST(self.data_dir, train=train, download=True, transform=transform)
        elif self.dataset_name == "cifar10":
            return torchvision.datasets.CIFAR10(self.data_dir, train=train, download=True, transform=transform)
        elif self.dataset_name == "cifar100":
            return CIFAR100Coarse(self.data_dir, train=train, download=True, transform=transform, use_coarse_labels=False)
        elif self.dataset_name == "cifar100coarse":
            return CIFAR100Coarse(self.data_dir, train=train, download=True, transform=transform, use_coarse_labels=True)
        elif self.dataset_name == "svhn":
            return torchvision.datasets.SVHN(self.data_dir, split="train" if train else "test", download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: '{self.dataset_name}'")

    def __subsample_dataset(self, dataset: torch.utils.data.Dataset, num_samples: int):
        train_indices, _ = train_test_split(torch.arange(len(dataset)), train_size=num_samples, stratify=self.train_dataset.targets,
                                            random_state=self.split_random_state if self.split_random_state > 0 else None)
        return torch.utils.data.Subset(dataset, indices=train_indices)

    def train_dataloader(self, shuffle: bool = True):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
