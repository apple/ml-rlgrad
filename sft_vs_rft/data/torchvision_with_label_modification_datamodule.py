
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch.utils.data
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms

from common.data.cifar100.cifar100_coarse import CIFAR100Coarse
from common.data.datasets.wrappers import TensorsDatasetWrapper
from common.data.modules import DataModule


class TorchvisionWithLabelModificationDataModule(DataModule):
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

    def __init__(self, dataset_name: str, data_dir: str = './data', num_train_samples: int = -1, num_test_samples=-1,
                 batch_size: int = 256, num_labels_per_sample: int = 1, use_multiple_labels_per_sample: bool = False,
                 frac_train_samples_modify_label: float = 0, num_workers: int = 0, split_random_state: int = -1):
        super().__init__()
        self.dataset_name = dataset_name
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: '{self.dataset_name}'")

        self.data_dir = data_dir
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        self.num_labels_per_sample = num_labels_per_sample
        self.use_multiple_labels_per_sample = use_multiple_labels_per_sample
        self.frac_train_samples_modify_label = frac_train_samples_modify_label
        self.num_workers = num_workers
        self.split_random_state = split_random_state

        if use_multiple_labels_per_sample and num_labels_per_sample <= 1:
            raise ValueError("'num_labels_per_sample' must be greater than 1 when 'use_multiple_labels_per_sample' is True")

    def setup(self):
        self.train_dataset = self.__get_dataset(train=True)
        self.train_dataset.targets = torch.tensor(self.train_dataset.targets, dtype=torch.long)
        self.test_dataset = self.__get_dataset(train=False)
        self.test_dataset.targets = torch.tensor(self.test_dataset.targets, dtype=torch.long)

        self.original_num_train_samples = len(self.train_dataset)
        self.input_dims = tuple(self.train_dataset[0][0].shape)
        self.num_classes = len(self.train_dataset.classes)

        self.used_train_indices = torch.arange(len(self.train_dataset))
        self.used_test_indices = torch.arange(len(self.test_dataset))

        if 0 < self.num_train_samples < len(self.train_dataset):
            self.train_dataset, self.used_train_indices, self.remaining_train_indices = self.__subsample_dataset(self.train_dataset, self.num_train_samples)

        if 0 < self.num_test_samples < len(self.test_dataset):
            self.test_dataset, self.used_test_indices, _ = self.__subsample_dataset(self.test_dataset, self.num_test_samples)

        self.train_indices = torch.arange(0, len(self.train_dataset))
        self.test_indices = torch.arange(0, len(self.test_dataset))
        self.train_per_sample_multiple_labels, self.per_class_labels = (None, None) if self.num_labels_per_sample <= 1 \
            else self.__create_per_sample_multiple_labels(self.train_dataset)
        self.test_per_sample_multiple_labels, _ = (None, None) if self.num_labels_per_sample <= 1 else self.__create_per_sample_multiple_labels(
            self.test_dataset,
            self.per_class_labels)
        self.indices_of_modified_label = self.__modify_train_labels()

        self.train_modified_indicator = torch.zeros_like(self.train_indices, dtype=torch.bool)
        self.test_modified_indicator = torch.zeros_like(self.test_indices, dtype=torch.bool)

        if self.indices_of_modified_label is not None:
            self.train_modified_indicator[self.indices_of_modified_label] = 1

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
        to_use_indices, remaining_indices = train_test_split(torch.arange(len(dataset)), train_size=num_samples, stratify=dataset.targets,
                                                             random_state=self.split_random_state if self.split_random_state > 0 else None)
        return torch.utils.data.Subset(dataset, indices=to_use_indices), to_use_indices, remaining_indices

    def __create_per_sample_multiple_labels(self, dataset, per_class_labels: torch.Tensor = None):
        targets = dataset.targets if not isinstance(dataset, torch.utils.data.Subset) else dataset.dataset.targets

        if per_class_labels is None:
            per_class_label_options = torch.arange(0, self.num_classes, dtype=torch.long).unsqueeze(dim=0).expand(self.num_classes, -1)
            per_class_label_options_not_equal_to_existing = per_class_label_options[
                per_class_label_options != torch.arange(self.num_classes).unsqueeze(dim=1)].view(self.num_classes, -1)

            per_class_new_label_indices = torch.argsort(torch.rand_like(per_class_label_options_not_equal_to_existing, dtype=torch.float), dim=-1)
            per_class_new_labels = torch.gather(per_class_label_options_not_equal_to_existing, dim=-1,
                                                index=per_class_new_label_indices)[:, :self.num_labels_per_sample - 1]
            return (torch.cat([targets.unsqueeze(dim=1), per_class_new_labels[targets]], dim=1),
                    torch.cat([torch.arange(self.num_classes).unsqueeze(dim=1), per_class_new_labels], dim=1))
        else:
            return per_class_labels[targets], per_class_labels

    def __modify_train_labels(self):
        if self.frac_train_samples_modify_label <= 0:
            return None

        targets = self.train_dataset.targets if not isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset.dataset.targets
        num_samples_to_modify_labels_for = int(self.used_train_indices.shape[0] * self.frac_train_samples_modify_label)
        modify_labels_of_indices = torch.randperm(self.used_train_indices.shape[0])[:num_samples_to_modify_labels_for]
        label_options = torch.arange(0, self.num_classes, dtype=torch.long).unsqueeze(dim=0).expand(num_samples_to_modify_labels_for, -1)

        if self.train_per_sample_multiple_labels is None:
            targets_to_modify = targets[self.used_train_indices][modify_labels_of_indices]
            label_options_not_equal_to_existing = label_options[label_options != targets_to_modify.unsqueeze(dim=1)].view(num_samples_to_modify_labels_for, -1)
        else:
            targets_to_modify = self.train_per_sample_multiple_labels[self.used_train_indices][modify_labels_of_indices]
            label_options_not_equal_to_existing = label_options
            for i in range(targets_to_modify.shape[1]):
                label_options_not_equal_to_existing = label_options_not_equal_to_existing[
                    label_options_not_equal_to_existing != targets_to_modify[:, i: i + 1]].view(
                    num_samples_to_modify_labels_for, -1)

        new_label_indices = torch.randint(0, label_options_not_equal_to_existing.shape[1], size=(label_options_not_equal_to_existing.shape[0],))
        new_labels = label_options_not_equal_to_existing[torch.arange(0, label_options_not_equal_to_existing.shape[0]), new_label_indices]
        targets[self.used_train_indices[modify_labels_of_indices]] = new_labels
        return modify_labels_of_indices

    def train_dataloader(self, shuffle: bool = True):
        if not self.use_multiple_labels_per_sample:
            wrapped_train_dataset = TensorsDatasetWrapper(self.train_dataset, self.train_indices, self.train_modified_indicator)
        else:
            wrapped_train_dataset = TensorsDatasetWrapper(self.train_dataset,
                                                          self.train_per_sample_multiple_labels[self.used_train_indices],
                                                          self.train_indices,
                                                          self.train_modified_indicator,
                                                          take_only_first_dataset_element=True)

        return torch.utils.data.DataLoader(wrapped_train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        if not self.use_multiple_labels_per_sample:
            wrapped_test_dataset = TensorsDatasetWrapper(self.test_dataset, self.test_indices, self.test_modified_indicator)
        else:
            wrapped_test_dataset = TensorsDatasetWrapper(self.test_dataset,
                                                         self.test_per_sample_multiple_labels[self.used_test_indices],
                                                         self.test_indices,
                                                         self.test_modified_indicator,
                                                         take_only_first_dataset_element=True)

        return torch.utils.data.DataLoader(wrapped_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
