from typing import Optional, Union, Sequence, Mapping, Dict, Any
from datasets import Dataset, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict, DownloadConfig, DownloadMode, VerificationMode
from datasets.features import Features
from datasets.splits import Split
from datasets.utils.version import Version


class LoadDataset:
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        ignore_verifications: bool = False,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token: Union[bool, str] = False,
        task: Optional[str] = None,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        trust_remote_code: Optional[bool] = None,
        **config_kwargs: Any,
    ):
        self.path = path
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.cache_dir = cache_dir
        self.features = features
        self.download_config = download_config
        self.download_mode = download_mode
        self.verification_mode = verification_mode
        self.ignore_verifications = ignore_verifications
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.revision = revision
        self.token = token
        self.use_auth_token = use_auth_token
        self.task = task
        self.streaming = streaming
        self.num_proc = num_proc
        self.storage_options = storage_options
        self.trust_remote_code = trust_remote_code
        self.config_kwargs = config_kwargs

    def load(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """
        Load the dataset based on the provided configuration.

        Returns:
            Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]: Loaded dataset
        """
        return load_dataset(
            path=self.path,
            name=self.name,
            data_dir=self.data_dir,
            data_files=self.data_files,
            split=self.split,
            cache_dir=self.cache_dir,
            features=self.features,
            download_config=self.download_config,
            download_mode=self.download_mode,
            verification_mode=self.verification_mode,
            ignore_verifications=self.ignore_verifications,
            keep_in_memory=self.keep_in_memory,
            save_infos=self.save_infos,
            revision=self.revision,
            token=self.token,
            use_auth_token=self.use_auth_token,
            task=self.task,
            streaming=self.streaming,
            num_proc=self.num_proc,
            storage_options=self.storage_options,
            trust_remote_code=self.trust_remote_code,
            **self.config_kwargs
        )


# Example usage
if __name__ == "__main__":
    dataset_loader = LoadDataset(
        path="squad",
        split="train",
        cache_dir="./cache",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        num_proc=4
    )
    
    dataset = dataset_loader.load()
    print(dataset)