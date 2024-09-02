from typing import Optional, Union, Sequence, Mapping, Dict, Any,List
from datasets import  Split
from datasets.features import Features
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.download.download_config import DownloadConfig
from datasets.utils.info_utils import VerificationMode
from datasets.utils.version import Version


class RagDatasetConfig:
    """Class to hold configuration for dataset loading."""

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
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: Optional[bool] = None,
        **config_kwargs: Any
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
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.revision = revision
        self.token = token
        self.streaming = streaming
        self.num_proc = num_proc
        self.storage_options = storage_options
        self.trust_remote_code = trust_remote_code
        self.config_kwargs = config_kwargs


