
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import bisect
import json
import os
from collections import OrderedDict

import torch.utils.data
from PIL import Image

from ..image.field_mapper import FieldMapper


class ImageDataset(torch.utils.data.Dataset):
    """
    Image dataset that is based on a JSON metadata file. The metadata file is an array where each entry includes the details
    of an image in the dataset. The image details should include the relative path to the image from the dataset directory
    and any other wanted fields.
    Example dataset_metadata.json:
    {
        images: [
            {
                "path": "path/to/image1.jpg"
                ...
            },
            {
                "path": "path/to/image2.jpg"
                ...
            }
        ]
    }
    """

    IMAGES_FIELD_NAME = "images"
    PATH_FIELD_NAME = "path"

    def __init__(self, dir, dataset_metadata_file_name="dataset_metadata.json", with_metadata_transform=None, transform=None, metadata_transform=None,
                 include_image_metadata=True, image_mode="RGB"):
        """
        :param dir: directory of the dataset (where the dataset metadata json is located).
        :param dataset_metadata_file_name: name of the dataset metadata json file.
        :param with_metadata_transform: transform for retrieved images that receives as inputs the image and its metadata.
        :param transform: transform for retrieved images that receives only image as input. This transform runs after the with_metadata_transform.
        :param metadata_transform: transform for the metadata of an image being retrieved.
        :param include_image_metadata: flag whether to include the image metadata when accessing an item in the dataset.
        :param image_mode: mode to load images in. Defaults to RGB.
        """
        self.dir = dir
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.metadata_file_path = os.path.join(dir, dataset_metadata_file_name)
        self.with_metadata_transform = with_metadata_transform if with_metadata_transform else lambda x, metadata: x
        self.transform = transform if transform else lambda x: x
        self.metadata_transform = metadata_transform if metadata_transform else lambda x: x
        self.include_image_metadata = include_image_metadata
        self.image_mode = image_mode

        if not os.path.exists(self.metadata_file_path):
            self.metadata = {self.IMAGES_FIELD_NAME: []}
            self.save_metadata_json()
        else:
            with open(self.metadata_file_path) as f:
                self.metadata = json.load(f)

        self.images_metadata = self.metadata[self.IMAGES_FIELD_NAME]

    def __preprocess_image_transform(self, image, metadata):
        image = self.with_metadata_transform(image, metadata)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_metadata = self.images_metadata[index]
        path = image_metadata[self.PATH_FIELD_NAME]
        image = Image.open(os.path.join(self.dir, path)).convert(self.image_mode)

        if self.include_image_metadata:
            return self.__preprocess_image_transform(image, image_metadata), self.metadata_transform(image_metadata)
        return self.__preprocess_image_transform(image, image_metadata)

    def __len__(self):
        return len(self.images_metadata)

    def get_image_metadata(self, index, use_transform=False):
        image_metadata = self.images_metadata[index]
        if use_transform:
            return self.metadata_transform(image_metadata)

        return image_metadata

    def add_image_metadata(self, image_metadata, update_metadata_file=True):
        if self.PATH_FIELD_NAME not in image_metadata:
            raise ValueError(f"Image metadata must contain image path field with name: {self.PATH_FIELD_NAME}")
        self.images_metadata.append(image_metadata)

        if update_metadata_file:
            self.save_metadata_json()

    def add_image(self, image, image_metadata, update_metadata_file=True):
        self.add_image_metadata(image_metadata)
        image_path = image_metadata[self.PATH_FIELD_NAME]
        image.save(os.path.join(self.dir, image_path))

        if update_metadata_file:
            self.save_metadata_json()

    def save_metadata_json(self):
        with open(self.metadata_file_path, "w") as f:
            json.dump(self.metadata, f, indent=2)


class DataAugmentedImageDataset(torch.utils.data.Dataset):
    """
    Data augmentation wrapper for ImageDataset. Creates for each image multiple augmented variants according to the given transforms. The variants
    are created once upon dataset creation and are used as extra images in the dataset. The augmented images are stored in memory, so this should
    be used with caution in terms of memory usage.
    """

    AUGMENTED_METADATA_FLAG = "augmented"

    def __init__(self, image_dataset, augmentation_transforms, transform=None, metadata_transform=None, include_image_metadata=True):
        """
        :param image_dataset: ImageDataset to wrap.
        :param augmentation_transforms: sequence of transforms.
        :param transform: transform for retrieved images that receives only image as input. This transform runs after the with_metadata_transform.
        :param metadata_transform: transform for the metadata of an image being retrieved.
        :param include_image_metadata: flag whether to include the image metadata when accessing an item in the dataset. Must match the given
        ImageDataset include_image_metadata.
        """
        self.image_dataset = image_dataset
        self.augmented_data = []

        self.include_image_metadata = include_image_metadata
        self.augmentation_transforms = augmentation_transforms
        self.transform = transform if transform else lambda x: x
        self.metadata_transform = metadata_transform if metadata_transform else lambda x: x

        self.images_metadata = self.image_dataset.images_metadata.copy()
        self.__populate_augmented_data()

    def __populate_augmented_data(self):
        for i in range(len(self.image_dataset)):
            image = self.image_dataset[i]
            if self.include_image_metadata:
                image = image[0]  # In this case image would be a tuple of image, metadata
            metadata = self.image_dataset.images_metadata[i]

            for augmentation_transform in self.augmentation_transforms:
                augmented_image = augmentation_transform(image)
                augmented_image_metadata = metadata.copy()
                augmented_image_metadata[self.AUGMENTED_METADATA_FLAG] = True

                self.augmented_data.append(augmented_image)
                self.images_metadata.append(augmented_image_metadata)

    def __getitem__(self, index):
        if index < len(self.image_dataset):
            return self.__get_from_original_image_dataset(index)

        return self.__get_from_augmented_data(index)

    def __get_from_original_image_dataset(self, index):
        if self.include_image_metadata:
            image, metadata = self.image_dataset[index]
            return self.transform(image), self.metadata_transform(metadata)

        return self.transform(self.image_dataset[index])

    def __get_from_augmented_data(self, index):
        image = self.augmented_data[index - len(self.image_dataset)]
        if self.include_image_metadata:
            return self.transform(image), self.metadata_transform(self.images_metadata[index])

        return self.transform(image)

    def __len__(self):
        return len(self.images_metadata)


class SubsetImageDataset(torch.utils.data.Dataset):
    """
    Subset wrapper for ImageDataset. Uses only subset of the dataset with the given indices.
    """

    def __init__(self, image_dataset, indices, transform=None):
        self.image_dataset = image_dataset
        self.original_indices = indices
        self.transform = transform if transform else lambda x: x
        self.images_metadata = [self.image_dataset.images_metadata[i] for i in indices]

    def __getitem__(self, index):
        return self.transform(self.image_dataset[self.original_indices[index]])

    def __len__(self):
        return len(self.images_metadata)


class FilteredImageDataset(SubsetImageDataset):
    """
    Filter wrapper for ImageDataset. Filters such the dataset contains only matching images (match is done according to their metadata).
    """

    def __init__(self, image_dataset, filter, transform=None):
        self.image_dataset = image_dataset
        self.filter = filter
        relevant_indices = self.__create_relevant_images_indices()
        super().__init__(image_dataset, relevant_indices, transform if transform else lambda x: x)

    def __create_relevant_images_indices(self):
        relevant_images_indices = []
        for index, image_metadata in enumerate(self.image_dataset.images_metadata):
            if self.filter(image_metadata):
                relevant_images_indices.append(index)

        return relevant_images_indices


class ConcatImageDataset(torch.utils.data.Dataset):
    """
    Concatenate multiple ImageDatasets together.
    Arguments:
        datasets (sequence): List of ImageDatasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.images_metadata = []
        for dataset in self.datasets:
            self.images_metadata.extend(dataset.images_metadata)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def create_field_filtered_image_dataset(image_dataset, field_name, field_values):
    """
    Creates an image dataset that is a subset of the given dataset with only the images that have a matching value for the given field name and
    values.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter by.
    :param field_values: sequence of matching field values. An item will be in the filtered dataset if its field value is in this sequence.
    :return: FilteredImageDataset with only the images with a field value that is in the given field values for the field name.
    """

    def filter_func(image_metadata):
        return image_metadata[field_name] in field_values

    return FilteredImageDataset(image_dataset, filter_func)


def create_label_mapper(image_dataset, field_name):
    """
    Creates a dictionary mapping between a label to its index.
    :param image_dataset: image dataset.
    :param field_name: field name to create the mapper for.
    :return: dictionary of field value (label) to id.
    """
    labels_set = {metadata[field_name] for metadata in image_dataset.images_metadata if field_name in metadata}
    sorted_labels = sorted(list(labels_set))
    return {label: i for i, label in enumerate(sorted_labels)}


def create_extract_label_id_metadata_transform(image_dataset, field_name, no_label_id=-100):
    """
    Creates a metadata transform function that extracts the label id for a field name from the given metadata.
    :param image_dataset: image dataset.
    :param field_name: field name to extract label from.
    :param no_label_id: default id the transform will return to images without the given field name.
    :return: metadata transform function to extract label id.
    """
    label_mapper = create_label_mapper(image_dataset, field_name)

    def metadata_to_label_id(metadata):
        if field_name not in metadata:
            return no_label_id

        field_value = metadata[field_name]
        if field_value not in label_mapper:
            return no_label_id

        return label_mapper[field_value]

    return metadata_to_label_id


def create_dict_of_label_mappers(image_dataset, by_task_field_name):
    """
    Creates dictionary of mappers that map field value to label id. Each key in field_names_dict is in the returned dictionary and the value is the
    mapper relevant to that field.
    :param image_dataset: image dataset.
    :param by_task_field_name: dictionary of key name to a field name to extract label id for.
    :return: dictionary of task name to label mapper.
    """
    label_mappers = OrderedDict()
    for task_name, field_name in by_task_field_name.items():
        label_mappers[task_name] = create_label_mapper(image_dataset, field_name)

    return label_mappers


def create_multitask_extract_label_ids_metadata_transform(image_dataset, by_task_field_name, no_label_id=-100):
    """
    Creates a metadata transform function that extracts a dictionary of labels for the given ids. The keys of the field_names_dict will be those
    of the returned dictionary from the transform and the values will be the label ids.
    :param image_dataset: image dataset.
    :param by_task_field_name: dictionary of key name to a field name to extract label id for.
    :param no_label_id: default id the transform will return to images without the given field name.
    :return: metadata transform to extract multitask label ids.
    """
    label_mappers = create_dict_of_label_mappers(image_dataset, by_task_field_name)

    def metadata_to_label_ids(metadata):
        label_ids_dict = OrderedDict()
        for task_name, field_name in by_task_field_name.items():
            label_mapper = label_mappers[task_name]

            if field_name not in metadata:
                label_ids_dict[task_name] = no_label_id
                continue

            field_value = metadata[field_name]
            if field_value not in label_mapper:
                label_ids_dict[task_name] = no_label_id
                continue

            label_ids_dict[task_name] = label_mapper[field_value]

        return label_ids_dict

    return metadata_to_label_ids


def create_frequent_values_filtered_image_dataset(image_dataset, field_name, freq_threshold=2):
    """
    Creates a FilteredImageDataset, filtering out images that have infrequent values for a certain field. The resulting dataset will contain only
    images that their value for the field has frequency that is greater than (or equals) to freq_threshold.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter according to its frequency.
    :param freq_threshold: threshold frequency to keep images with values that their frequency is above (or equal) to the threshold.
    :return: FilteredImageDataset with images with frequent values for the given field.
    """
    field_indices_mapper = FieldMapper(image_dataset.images_metadata, field_name)
    indices_to_remove = set()
    for field_value, indices in field_indices_mapper.field_value_to_indices.items():
        if len(indices) < freq_threshold:
            indices_to_remove.update(indices)

    indices_to_keep = [i for i in range(len(image_dataset)) if i not in indices_to_remove]
    return SubsetImageDataset(image_dataset, indices_to_keep)
