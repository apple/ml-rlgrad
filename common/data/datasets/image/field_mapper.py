
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import json
from typing import Sequence, List, Dict


class FieldMapper:
    """
    Maps a given index to the indices of all items with the same field value.
    """

    def __init__(self, metadatas: List[Dict], field_name: str):
        self.metadatas = metadatas
        self.field_name = field_name
        self.field_value_to_indices = {}

        for index, metadata in enumerate(self.metadatas):
            field_value = metadata[self.field_name]
            if field_value not in self.field_value_to_indices:
                self.field_value_to_indices[field_value] = []

            self.field_value_to_indices[field_value].append(index)

    def __getitem__(self, index: int) -> List[int]:
        field_value = self.get_field_value(index)
        return self.field_value_to_indices[field_value]

    def by_field_indices(self) -> Sequence[List[int]]:
        return list(self.field_value_to_indices.values())

    def get_field_value(self, index: int) -> object:
        return self.metadatas[index][self.field_name]

    def get_identifier(self, index: int) -> object:
        return self.get_field_value(index)

    def get_identifier_to_indices_dict(self) -> Dict[object, List[int]]:
        return self.field_value_to_indices


class MultiFieldMapper:
    """
    Maps a given index to the indices of all items with the same multiple fields values.
    """

    FIELDS_SEPARATOR = '.'

    def __init__(self, metadatas: List[Dict], fields_names: List[str], allow_missing_fields: bool = False):
        self.metadatas = metadatas
        self.fields_names = fields_names
        self.fields_values_to_indices = {}
        self.allow_missing_fields = allow_missing_fields

        for index, metadata in enumerate(self.metadatas):
            fields_values = self.__get_metadata_field_values(metadata)
            fields_values_identifier = self.__create_fields_identifier(fields_values)
            if fields_values_identifier not in self.fields_values_to_indices:
                self.fields_values_to_indices[fields_values_identifier] = []

            self.fields_values_to_indices[fields_values_identifier].append(index)

    def __get_metadata_field_values(self, metadata: dict) -> List[str]:
        field_values = []
        for field_name in self.fields_names:
            if field_name in metadata:
                field_values.append(metadata[field_name])
            elif self.allow_missing_fields:
                field_values.append("")
            else:
                raise ValueError(f"Missing field value for field '{field_name}' in metadata: {json.dumps(metadata, indent=2)}")

        return field_values

    def __getitem__(self, index: int) -> List[int]:
        fields_values = self.get_fields_values(index)
        fields_values_identifier = self.__create_fields_identifier(fields_values)
        return self.fields_values_to_indices[fields_values_identifier]

    def by_fields_indices(self) -> Sequence[List[int]]:
        return list(self.fields_values_to_indices.values())

    def get_fields_values(self, index: int) -> List[object]:
        metadata = self.metadatas[index]
        return self.__get_metadata_field_values(metadata)

    def get_identifier(self, index: int) -> object:
        return self.__create_fields_identifier(self.get_fields_values(index))

    def get_identifier_to_indices_dict(self) -> Dict[object, List[int]]:
        return self.fields_values_to_indices

    def __create_fields_identifier(self, fields_values):
        return self.FIELDS_SEPARATOR.join(fields_values)


class DummyMapper:
    """
    Maps a given index to the same indices.
    """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __getitem__(self, index: int):
        return self.indices
