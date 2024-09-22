import os
from typing import Any, Dict, List, Optional, Union

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from .configs import DataArguments

def mix_datasets(
    dataset_mixer: Dict[str, float],
    configs: Optional[List[Optional[str]]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    **kwargs,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Loads all available splits from each dataset.

    Args:
        dataset_mixer (Dict[str, float]):
            Dictionary containing the dataset names and their fractions. The fractions should be between 0 and 1.
        configs (Optional[List[Optional[str]]]):
            List of dataset config names. Must be the same length as 'dataset_mixer'. If None, defaults to None for all datasets.
        columns_to_keep (Optional[List[str]]):
            List of column names to keep in the datasets. All other columns will be removed.
        shuffle (bool):
            Whether to shuffle the concatenated datasets.
        **kwargs:
            Additional keyword arguments to pass to the dataset loading functions.

    Returns:
        DatasetDict: A dictionary with keys as split names and values as concatenated datasets.

    Raises:
        ValueError: If dataset fractions are not between 0 and 1.
        ValueError: If no datasets were loaded.
        ValueError: If the number of configs does not match the number of datasets.
    """
    columns_to_keep = columns_to_keep or []

    if configs is None:
        configs = [None] * len(dataset_mixer)
    elif len(configs) != len(dataset_mixer):
        raise ValueError("The number of configs must match the number of datasets in 'dataset_mixer'.")

    datasets_per_split = {}

    for (dataset_name, frac), config_name in zip(dataset_mixer.items(), configs):
        if not (0 <= frac <= 1):
            raise ValueError(f"Dataset fraction for '{dataset_name}' must be between 0 and 1.")

        # Load all splits
        try:
            dataset_dict = load_dataset(dataset_name, config_name, **kwargs)
        except Exception:
            try:
                # Attempt to load from disk if not found on the hub
                dataset_dict = load_from_disk(dataset_name)
            except Exception as e:
                raise ValueError(f"Could not load dataset '{dataset_name}'.") from e

        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError(f"Dataset '{dataset_name}' is not a DatasetDict with multiple splits.")

        for split, dataset in dataset_dict.items():
            # Remove unnecessary columns
            columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
            dataset = dataset.remove_columns(columns_to_remove)

            # Subsample the dataset
            subset_size = int(frac * len(dataset))
            dataset = dataset.select(range(subset_size))

            # Add dataset to the split
            datasets_per_split.setdefault(split, []).append(dataset)

    # Concatenate datasets for each split
    mixed_datasets = DatasetDict()
    for split, datasets_list in datasets_per_split.items():
        if datasets_list:
            concatenated_dataset = concatenate_datasets(datasets_list)
            if shuffle:
                concatenated_dataset = concatenated_dataset.shuffle(seed=42)
            mixed_datasets[split] = concatenated_dataset

    if not mixed_datasets:
        raise ValueError("No datasets were loaded. Please check your dataset names and configurations.")

    return mixed_datasets


def get_datasets(
    data_config: Union['DataArguments', Dict[str, float]],
    *,
    configs: Optional[List[Optional[str]]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    **kwargs,
) -> DatasetDict:
    """
    Loads and mixes datasets according to the provided data configuration.

    Args:
        data_config (DataArguments or Dict[str, float]):
            Dataset configuration containing dataset names and their corresponding fractions.
        configs (Optional[List[Optional[str]]]):
            List of dataset config names. Must be the same length as the number of datasets in 'data_config'.
        columns_to_keep (Optional[List[str]]):
            List of column names to keep in the datasets.
        shuffle (bool):
            Whether to shuffle the concatenated datasets.
        **kwargs:
            Additional keyword arguments to pass to the dataset loading functions.

    Returns:
        DatasetDict: A dictionary with keys as split names and values as concatenated datasets.

    Raises:
        ValueError: If 'data_config' is not a recognized type.
    """
    if isinstance(data_config, DataArguments):
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config of type {type(data_config)} is not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer=dataset_mixer,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
        **kwargs,
    )
    return raw_datasets