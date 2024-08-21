import re
import json
import os
from typing import List, Union, Optional, Dict, Any
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy

class AdvancedTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        **kwargs: Any
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )
        self.tokenizer = TokenizerFast.from_file(vocab_file)
        self.decoder = self.tokenizer.decoder

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any
    ) -> List[int]:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation:
            ids = self._truncate(ids, max_length)

        if padding:
            ids = self._pad(ids, max_length)

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs: Any
    ) -> str:
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in self.all_special_ids]
        
        text = self.decoder.decode(token_ids)
        
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
        
        return text

    def batch_encode(
        self,
        texts: List[Union[TextInput, PreTokenizedInput, TextInputPair, PreTokenizedInputPair]],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any
    ) -> BatchEncoding:
        encodings = self.tokenizer.encode_batch(texts)
        
        input_ids = []
        attention_mask = []
        
        for encoding in encodings:
            ids = encoding.ids
            
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
            if truncation:
                ids = self._truncate(ids, max_length)
            
            if padding:
                ids, mask = self._pad(ids, max_length, return_attention_mask=True)
            else:
                mask = [1] * len(ids)
            
            input_ids.append(ids)
            attention_mask.append(mask)
        
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs: Any
    ) -> List[str]:
        return [self.decode(seq, skip_special_tokens, clean_up_tokenization_spaces) for seq in sequences]

    def encode_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            return self.unk_token_id
        return token_id

    def decode_tokenid_str(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            return self.unk_token
        return token

    def _truncate(self, ids: List[int], max_length: Optional[int]) -> List[int]:
        if max_length is not None and len(ids) > max_length:
            return ids[:max_length]
        return ids

    def _pad(self, ids: List[int], max_length: Optional[int], return_attention_mask: bool = False) -> Union[List[int], tuple[List[int], List[int]]]:
        if max_length is None:
            return ids
        
        padding_length = max_length - len(ids)
        if padding_length <= 0:
            return ids

        padded_ids = ids + [self.pad_token_id] * padding_length
        
        if return_attention_mask:
            attention_mask = [1] * len(ids) + [0] * padding_length
            return padded_ids, attention_mask
        
        return padded_ids

    def _clean_up_tokenization(self, text: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        """
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r'\s([,.!?:;])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'(\w)\'(\w)', r'\1\'\2', text)  # Rejoin contractions
        text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)  # Rejoin numbers
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    def chat_prompt_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a chat prompt template from a list of messages.
        """
        template = []
        for message in messages:
            role = message.get("role", "").capitalize()
            content = message.get("content", "").strip()
            if role and content:
                template.append(f"{role}: {content}")
        return "\n".join(template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "AdvancedTokenizer":
        """
        Load a tokenizer from a pretrained model.
        """
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(tokenizer, cls):
            raise TypeError(f"Expected instance of {cls.__name__}, but got {type(tokenizer).__name__}")
        return tokenizer

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """
        Save the tokenizer configuration and vocabulary to a directory.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the tokenizer configuration
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, ensure_ascii=False, indent=2)
        
        # Save the vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        self.tokenizer.save(vocab_file)
        
        # Save special tokens
        special_tokens_file = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(self.special_tokens_map, f, ensure_ascii=False, indent=2)

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for the tokenizer.
        """
        return {
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }

    def __len__(self) -> int:
        """
        Return the size of the vocabulary.
        """
        return len(self.tokenizer.get_vocab())

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary.
        """
        return len(self)

    def get_vocab(self) -> Dict[str, int]:
        """
        Return the vocabulary as a dictionary.
        """
        return self.tokenizer.get_vocab()

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a sequence of tokens to a single string.
        """
        return self.tokenizer.decoder.decode(tokens)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        Convert a single index or a sequence of indices to tokens.
        """
        if isinstance(ids, int):
            return self.decode_tokenid_str(ids)
        return [self.decode_tokenid_str(id) for id in ids if not (skip_special_tokens and id in self.all_special_ids)]

def train_tokenizer(
    texts: List[str],
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: List[str] = ["<unk>", "<s>", "</s>", "<pad>"],
    trainer_type: str = "bpe"
) -> TokenizerFast:
    if trainer_type == "bpe":
        tokenizer = TokenizerFast(BPE())
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    elif trainer_type == "unigram":
        tokenizer = TokenizerFast(Unigram())
        trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    elif trainer_type == "wordlevel":
        tokenizer = TokenizerFast(WordLevel())
        trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    elif trainer_type == "wordpiece":
        tokenizer = TokenizerFast(WordPiece())
        trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    else:
        raise ValueError("Invalid trainer_type. Choose from 'bpe', 'unigram', 'wordlevel', or 'wordpiece'.")

    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer