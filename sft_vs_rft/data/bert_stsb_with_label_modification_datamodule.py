
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import datasets
import datasets as huggingface_datasets
import torch.utils.data
from datasets import Features, Value
from sklearn.model_selection import train_test_split
from transformers.models.bert import BertTokenizer

from common.data.datasets.wrappers import TensorsDatasetWrapper
from common.data.modules import DataModule


class HuggingfaceDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.__dict__.update(dataset.__dict__)
        self.dataset = dataset

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        label = sample.pop("label")
        return sample, label

    def __len__(self):
        return len(self.dataset)


class BertSTSBWithLabelModificationDataModule(DataModule):

    def __init__(self, tokenizer: BertTokenizer, batch_size: int, cache_dir: str = "./data", num_train_samples: int = -1,
                 num_labels_per_sample: int = 1, use_multiple_labels_per_sample: bool = False, frac_train_samples_modify_label: float = 0,
                 split_random_state: int = -1, num_workers: int = 0):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.num_train_samples = num_train_samples
        self.num_labels_per_sample = num_labels_per_sample
        self.use_multiple_labels_per_sample = use_multiple_labels_per_sample
        self.frac_train_samples_modify_label = frac_train_samples_modify_label
        self.split_random_state = split_random_state
        self.num_workers = num_workers

        if use_multiple_labels_per_sample and num_labels_per_sample <= 1:
            raise ValueError("'num_labels_per_sample' must be greater than 1 when 'use_multiple_labels_per_sample' is True")

    def setup(self):
        self.dataset = huggingface_datasets.load_dataset(path="glue", name="stsb", cache_dir=self.cache_dir)
        self.train_dataset = self.__tokenize_dataset(self.dataset["train"])
        self.val_dataset = self.__tokenize_dataset(self.dataset["validation"])
        self.test_dataset = self.val_dataset

        self.original_num_train_samples = len(self.train_dataset)
        self.num_classes = 6
        self.used_train_indices = torch.arange(len(self.train_dataset))

        if 0 < self.num_train_samples < len(self.train_dataset):
            self.train_dataset, self.used_train_indices, self.remaining_train_indices = self.__subsample_dataset(self.train_dataset, self.num_train_samples)

        self.train_indices = torch.arange(0, len(self.train_dataset))
        self.val_indices = torch.arange(0, len(self.val_dataset))
        self.test_indices = self.val_indices

        self.train_per_sample_multiple_labels, self.per_class_labels = (None, None) if self.num_labels_per_sample <= 1 \
            else self.__create_per_sample_multiple_labels(self.train_dataset)
        self.val_per_sample_multiple_labels, _ = (None, None) if self.num_labels_per_sample <= 1 \
            else self.__create_per_sample_multiple_labels(self.val_dataset, self.per_class_labels)

        self.indices_with_modified_label = self.__modify_train_labels()

        self.train_modified_indicator = torch.zeros(len(self.train_dataset), dtype=torch.bool)
        self.val_modified_indicator = torch.zeros(len(self.val_dataset), dtype=torch.bool)

        if self.indices_with_modified_label is not None:
            self.train_modified_indicator[self.indices_with_modified_label] = 1

    def __tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        dataset = dataset.map(lambda ex: {"label": round(ex["label"])})
        dataset = dataset.cast(Features({"sentence1": Value("string"), "sentence2": Value("string"), "idx": Value("int32"), "label": Value("int64")}))
        dataset = dataset.map(lambda ex: self.tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, padding="max_length"), batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        return dataset

    def __subsample_dataset(self, dataset, num_samples: int):
        to_use_indices, remaining_indices = train_test_split(torch.arange(len(dataset)), train_size=num_samples, stratify=dataset["label"],
                                                             random_state=self.split_random_state if self.split_random_state > 0 else None)
        return torch.utils.data.Subset(dataset, indices=to_use_indices), to_use_indices, remaining_indices

    def __create_per_sample_multiple_labels(self, dataset, per_class_labels: torch.Tensor = None):
        labels = dataset["label"] if not isinstance(dataset, torch.utils.data.Subset) else dataset.dataset["label"]

        if per_class_labels is None:
            per_class_label_options = torch.arange(0, self.num_classes, dtype=torch.long).unsqueeze(dim=0).expand(self.num_classes, -1)
            per_class_label_options_not_equal_to_existing = per_class_label_options[
                per_class_label_options != torch.arange(self.num_classes).unsqueeze(dim=1)].view(self.num_classes, -1)

            per_class_new_label_indices = torch.argsort(torch.rand_like(per_class_label_options_not_equal_to_existing, dtype=torch.float), dim=-1)
            per_class_new_labels = torch.gather(per_class_label_options_not_equal_to_existing, dim=-1,
                                                index=per_class_new_label_indices)[:, :self.num_labels_per_sample - 1]
            return (torch.cat([labels.unsqueeze(dim=1), per_class_new_labels[labels]], dim=1),
                    torch.cat([torch.arange(self.num_classes).unsqueeze(dim=1), per_class_new_labels], dim=1))
        else:
            return per_class_labels[labels], per_class_labels

    def __modify_train_labels(self):
        if self.frac_train_samples_modify_label <= 0:
            return None

        train_dataset = self.train_dataset if not isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset.dataset
        train_labels = train_dataset["label"]

        num_samples_to_modify_labels_for = int(self.used_train_indices.shape[0] * self.frac_train_samples_modify_label)
        modify_labels_of_indices = torch.randperm(self.used_train_indices.shape[0])[:num_samples_to_modify_labels_for]
        label_options = torch.arange(0, self.num_classes, dtype=torch.long).unsqueeze(dim=0).expand(num_samples_to_modify_labels_for, -1)

        if self.train_per_sample_multiple_labels is None:
            labels_to_modify = train_labels[self.used_train_indices][modify_labels_of_indices]
            label_options_not_equal_to_existing = label_options[label_options != labels_to_modify.unsqueeze(dim=1)].view(num_samples_to_modify_labels_for, -1)
        else:
            labels_to_modify = self.train_per_sample_multiple_labels[self.used_train_indices][modify_labels_of_indices]
            label_options_not_equal_to_existing = label_options
            for i in range(labels_to_modify.shape[1]):
                label_options_not_equal_to_existing = label_options_not_equal_to_existing[
                    label_options_not_equal_to_existing != labels_to_modify[:, i: i + 1]].view(
                    num_samples_to_modify_labels_for, -1)

        new_label_indices = torch.randint(0, label_options_not_equal_to_existing.shape[1], size=(label_options_not_equal_to_existing.shape[0],))
        new_labels = label_options_not_equal_to_existing[torch.arange(0, label_options_not_equal_to_existing.shape[0]), new_label_indices]

        train_labels[self.used_train_indices[modify_labels_of_indices]] = new_labels
        train_dataset = train_dataset.map(lambda ex: {"label": train_labels[ex["idx"]]})
        if isinstance(self.train_dataset, torch.utils.data.Subset):
            self.train_dataset.dataset = train_dataset
        else:
            self.train_dataset = train_dataset

        return modify_labels_of_indices

    def train_dataloader(self, shuffle: bool = True):
        if not self.use_multiple_labels_per_sample:
            wrapped_train_dataset = TensorsDatasetWrapper(HuggingfaceDatasetWrapper(self.train_dataset), self.train_indices, self.train_modified_indicator)
        else:
            wrapped_train_dataset = TensorsDatasetWrapper(HuggingfaceDatasetWrapper(self.train_dataset),
                                                          self.train_per_sample_multiple_labels[self.used_train_indices],
                                                          self.train_indices,
                                                          self.train_modified_indicator,
                                                          take_only_first_dataset_element=True)

        return torch.utils.data.DataLoader(wrapped_train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        if not self.use_multiple_labels_per_sample:
            wrapped_test_dataset = TensorsDatasetWrapper(HuggingfaceDatasetWrapper(self.val_dataset), self.val_indices, self.val_modified_indicator)
        else:
            wrapped_test_dataset = TensorsDatasetWrapper(HuggingfaceDatasetWrapper(self.val_dataset),
                                                         self.val_per_sample_multiple_labels,
                                                         self.val_indices,
                                                         self.val_modified_indicator,
                                                         take_only_first_dataset_element=True)

        return torch.utils.data.DataLoader(wrapped_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
