
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch.utils.data

from .image.field_mapper import FieldMapper


class OnlineTripletDataset(torch.utils.data.Dataset):
    """
    Dataset for getting query, positive and negative triplets with online sampling support.
    """

    def __init__(self, dataset, positive_sampler, negative_sampler):
        self.dataset = dataset
        self.positive_sampler = positive_sampler
        self.negative_sampler = negative_sampler

    def __getitem__(self, index):
        query = self.dataset[index]
        positive = self.dataset[self.positive_sampler(index)]
        negative = self.dataset[self.negative_sampler(index)]
        return query, positive, negative

    def __len__(self):
        return len(self.dataset)


class OfflineTripletDataset(torch.utils.data.Dataset):
    """
    Dataset for getting query, positive and negative triplets from predefined triplets.
    """

    def __init__(self, dataset, triplet_indices):
        self.dataset = dataset
        self.triplet_indices = triplet_indices

    def __getitem__(self, index):
        query_index, positive_index, negative_index = self.triplet_indices[index]
        return self.dataset[query_index], self.dataset[positive_index], self.dataset[negative_index]

    def __len__(self):
        return len(self.triplet_indices)


class SameIdPositiveSampler:

    def __init__(self, image_dataset, id_field_name="id", category_field_name="category"):
        self.image_dataset = image_dataset
        self.id_options_mapper = FieldMapper(image_dataset.images_metadata, field_name=id_field_name)
        self.id_to_same_id_indices_set = {value: set(indices) for value, indices in self.id_options_mapper.field_value_to_indices.items()}

        self.category_options_mapper = FieldMapper(image_dataset.images_metadata, field_name=category_field_name)
        self.category_to_same_category_indices_set = {value: set(indices) for value, indices in
                                                      self.category_options_mapper.field_value_to_indices.items()}

    def __call__(self, index):
        id = self.id_options_mapper.get_field_value(index)
        positive_options_indices = self.id_to_same_id_indices_set[id]

        if index in positive_options_indices:
            positive_options_indices.remove(index)
            positive_options_indices = list(positive_options_indices)

        if len(positive_options_indices) == 0:
            positive_options_indices = self.__get_no_same_id_options_fallback(index)

        return np.random.choice(positive_options_indices)

    def __get_no_same_id_options_fallback(self, index):
        category = self.category_options_mapper.get_field_value(index)
        # Get a positive candidate from same category
        positive_options_indices = set(self.category_to_same_category_indices_set[category])
        if index in positive_options_indices:
            positive_options_indices.remove(index)

        # If doesn't exist then just return self as only positive option
        if len(positive_options_indices) == 0:
            return [index]

        return list(positive_options_indices)


class SameIdNegativeSampler:

    def __init__(self, image_dataset, id_field_name="id", category_field_name="category",
                 use_in_category_frac_sampling=False, in_category_negative_frac=0.3):
        self.image_dataset = image_dataset
        self.id_options_mapper = FieldMapper(image_dataset.images_metadata, field_name=id_field_name)
        self.id_to_same_id_indices_set = {value: set(indices) for value, indices in self.id_options_mapper.field_value_to_indices.items()}

        self.use_in_category_frac_sampling = use_in_category_frac_sampling
        self.in_category_negative_frac = in_category_negative_frac
        self.category_options_mapper = FieldMapper(image_dataset.images_metadata, field_name=category_field_name)
        self.category_to_same_category_indices_set = {value: set(indices) for value, indices in
                                                      self.category_options_mapper.field_value_to_indices.items()}

    def __call__(self, index):
        id = self.id_options_mapper.get_field_value(index)
        category = self.category_options_mapper.get_field_value(index)
        same_id_indices = self.id_to_same_id_indices_set[id]
        same_category_indices = self.category_to_same_category_indices_set[category]

        if not self.use_in_category_frac_sampling:
            options = [i for i in range(len(self.image_dataset)) if i not in same_id_indices]
        else:
            options = self.__get_with_in_category_negative_frac_sampling_options(same_id_indices, same_category_indices)

        return np.random.choice(options)

    def __get_with_in_category_negative_frac_sampling_options(self, same_id_indices, same_category_indices):
        in_category = np.random.rand() < self.in_category_negative_frac
        if in_category:
            return [i for i in same_category_indices if i not in same_id_indices]
        else:
            return [i for i in range(len(self.image_dataset)) if i not in same_category_indices]
