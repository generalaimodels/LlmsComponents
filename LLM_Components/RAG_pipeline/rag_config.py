from typing import Optional, Union, Sequence, Mapping, Dict, Any,List
from datasets import  Split
from datasets.features import Features
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.download.download_config import DownloadConfig
from datasets.utils.info_utils import VerificationMode
from datasets.utils.version import Version
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

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




# Placeholder for actual implementations
AutoTokenizer = Any
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]



class DocumentSplitterconfig:
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 1000,
        separators: Optional[List[str]] = MARKDOWN_SEPARATORS,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.separators = separators or MARKDOWN_SEPARATORS
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.additional_args = kwargs

        self.validate_chunk_size()

    def validate_chunk_size(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")


class CustomFaissConfig:
    def __init__(
        self,
        embedding_function: Union[Callable[[str], List[float]]],
        index: Any,
        docstore,
        index_to_docstore_id: Dict[int, str],
        *,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy,
        **kwargs: Any
    ):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.relevance_score_fn = relevance_score_fn
        self.normalize_L2 = normalize_L2
        self.distance_strategy = distance_strategy
        self.additional_args = kwargs

        

    
class DirectoryLoaderConfig:
    def __init__(
        self,
        path: Union[str, Path],
        glob: Union[List[str], str] = "**/[!.]*",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = True,
        use_multithreading: bool = True,
        max_concurrency: int = 4,
        exclude: Union[List[str], str] = (),
        sample_size: int = 0,
        randomize_sample: bool = False,
        sample_seed: Optional[int] = None
    ):
        self.path = path
        self.glob = glob if isinstance(glob, list) else [glob]
        self.silent_errors = silent_errors
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.exclude = exclude if isinstance(exclude, list) else [exclude]
        self.sample_size = sample_size
        self.randomize_sample = randomize_sample
        self.sample_seed = sample_seed

        self.validate_path()
        self.validate_sample_size()

    def validate_path(self):
        if not self.path.exists():
            message = f"Path '{self.path}' does not exist."
            if not self.silent_errors:
                raise FileNotFoundError(message)
            print(message)

    def validate_sample_size(self):
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")