import logging
from typing import Any, Dict, List, Union

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTokenizer:
    """
    An advanced tokenizer that supports tokenization, encoding, decoding,
    padding, and batch-wise operations for data from a pandas DataFrame.
    """

    def __init__(self, special_tokens: List[str] = None) -> None:
        """
        Initializes the tokenizer with optional special tokens.

        Args:
            special_tokens (List[str], optional): List of special tokens to include in the vocabulary.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {}
        self.vocab_size: int = 0

        if special_tokens is not None:
            self.add_special_tokens(special_tokens)
        else:
            # Default special tokens
            self.add_special_tokens(['<PAD>', '<UNK>', '<SOS>', '<EOS>'])

    def add_special_tokens(self, tokens: List[str]) -> None:
        """
        Adds special tokens to the vocabulary.

        Args:
            tokens (List[str]): List of special tokens.
        """
        for token in tokens:
            if token not in self.token_to_id:
                token_id = self.vocab_size
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                self.special_tokens[token] = token_id
                self.vocab_size += 1

    def build_vocab(self, data: pd.DataFrame) -> None:
        """
        Builds the vocabulary from the data.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
        """
        for column in data.columns:
            for value in data[column]:
                self._process_value_for_vocab(value)

    def _process_value_for_vocab(self, value: Any) -> None:
        """
        Processes a single value to build the vocabulary.

        Args:
            value (Any): The value to process.
        """
        if value is None or (isinstance(value, float) and pd.isnull(value)):
            # Skip processing for None or NaN values
            return
        elif isinstance(value, str):
            tokens = self.tokenize(value)
            for token in tokens:
                self._add_token_to_vocab(token)
        elif isinstance(value, list):
            for item in value:
                self._process_value_for_vocab(item)
        elif isinstance(value, dict):
            for key, lst in value.items():
                self._process_value_for_vocab(key)
                self._process_value_for_vocab(lst)
        else:
            # Handle other data types as strings
            logger.warning(f"Unsupported data type in vocabulary building: {type(value)}. Converting to string.")
            tokens = self.tokenize(str(value))
            for token in tokens:
                self._add_token_to_vocab(token)

    def _add_token_to_vocab(self, token: str) -> None:
        """
        Adds a single token to the vocabulary.

        Args:
            token (str): The token to add.
        """
        if token not in self.token_to_id:
            token_id = self.vocab_size
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.vocab_size += 1

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a piece of text.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: The list of tokens.
        """
        # Example tokenizer: simple whitespace tokenizer
        tokens = text.strip().split()
        return tokens

    def _process_value_for_encoding(self, value: Any) -> List[str]:
        """
        Processes a value for encoding into tokens.

        Args:
            value (Any): The value to process.

        Returns:
            List[str]: The list of tokens.
        """
        tokens = []
        if value is None or (isinstance(value, float) and pd.isnull(value)):
            # Return empty tokens for None or NaN values
            return tokens
        elif isinstance(value, str):
            tokens.extend(self.tokenize(value))
        elif isinstance(value, list):
            for item in value:
                tokens.extend(self._process_value_for_encoding(item))
        elif isinstance(value, dict):
            for key, lst in value.items():
                tokens.extend(self._process_value_for_encoding(key))
                tokens.extend(self._process_value_for_encoding(lst))
        else:
            # Handle other data types as strings
            logger.warning(f"Unsupported data type in encoding: {type(value)}. Converting to string.")
            tokens.extend(self.tokenize(str(value)))
        return tokens

    def encode(
        self,
        text: Union[str, List[Any], Dict[str, Any]],
        add_special_tokens: bool = True,
        max_length: int = None,
        padding: bool = False
    ) -> List[int]:
        """
        Encodes text into token IDs.

        Args:
            text (Union[str, List[Any], Dict[str, Any]]): The text to encode.
            add_special_tokens (bool): Whether to add special tokens.
            max_length (int, optional): The maximum length for padding/truncation.
            padding (bool): Whether to pad the sequence.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = self._process_value_for_encoding(text)
        if add_special_tokens:
            tokens = ['<SOS>'] + tokens + ['<EOS>']

        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                token_ids.append(token_id)
            else:
                unk_id = self.special_tokens.get('<UNK>')
                if unk_id is not None:
                    token_ids.append(unk_id)
                else:
                    logger.error("'<UNK>' token not found in special tokens.")
                    raise ValueError("'<UNK>' token not found in special tokens.")

        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif padding and len(token_ids) < max_length:
                pad_id = self.special_tokens.get('<PAD>')
                if pad_id is not None:
                    token_ids += [pad_id] * (max_length - len(token_ids))
                else:
                    logger.error("Padding requested but '<PAD>' token not found in special tokens.")
                    raise ValueError("'<PAD>' token not found in special tokens.")

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decodes a list of token IDs back to text.

        Args:
            token_ids (List[int]): The list of token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            str: The decoded text.
        """
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id)
            if token is not None:
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                logger.warning(f"Token ID {token_id} not found in id_to_token mapping.")
                tokens.append('<UNK>')
        text = ' '.join(tokens)
        return text

    def encode_batch(
        self,
        texts: List[Union[str, List[Any], Dict[str, Any]]],
        add_special_tokens: bool = True,
        max_length: int = None,
        padding: bool = False
    ) -> List[List[int]]:
        """
        Encodes a batch of texts into token IDs.

        Args:
            texts (List[Union[str, List[Any], Dict[str, Any]]]): The list of texts to encode.
            add_special_tokens (bool): Whether to add special tokens.
            max_length (int, optional): The maximum length for padding/truncation.
            padding (bool): Whether to pad the sequences.

        Returns:
            List[List[int]]: The list of token ID sequences.
        """
        encoded_texts = []
        for text in texts:
            try:
                encoded = self.encode(
                    text,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    padding=padding
                )
                encoded_texts.append(encoded)
            except Exception as e:
                logger.error(f"Error encoding text: {e}")
                raise
        return encoded_texts

    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decodes a batch of token ID sequences back to text.

        Args:
            batch_token_ids (List[List[int]]): The list of token ID sequences to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            List[str]: The list of decoded texts.
        """
        decoded_texts = []
        for token_ids in batch_token_ids:
            try:
                decoded = self.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens
                )
                decoded_texts.append(decoded)
            except Exception as e:
                logger.error(f"Error decoding token IDs: {e}")
                raise
        return decoded_texts

    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: int = None,
        padding_value: int = None
    ) -> List[List[int]]:
        """
        Pads sequences to the same length.

        Args:
            sequences (List[List[int]]): The list of sequences to pad.
            max_length (int, optional): The maximum length for padding.
            padding_value (int, optional): The token ID to use for padding.

        Returns:
            List[List[int]]: The list of padded sequences.
        """
        if padding_value is None:
            padding_value = self.special_tokens.get('<PAD>')
            if padding_value is None:
                logger.error("Padding requested but '<PAD>' token not found in special tokens.")
                raise ValueError("'<PAD>' token not found in special tokens.")

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                padded_seq = seq + [padding_value] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return padded_sequences


# import pandas as pd

# # Example DataFrame
# data = pd.DataFrame({
#     'text_column': [
#         'Hello world',
#         ['This is a list', 'With multiple strings'],
#         {'key1': ['List in dict'], 'key2': ['Another list']}
#     ],
#     'another_column': [
#         'Additional text',
#         ['List of text', 'More text'],
#         {'dict_key': ['Dict list']}
#     ]
# })

# # Initialize the tokenizer
# tokenizer = AdvancedTokenizer()

# # Build the vocabulary
# tokenizer.build_vocab(data)

# # Encode a single value
# encoded = tokenizer.encode(data['text_column'][0])
# print(f"Encoded: {encoded}")

# # Decode the token IDs back to text
# decoded = tokenizer.decode(encoded)
# print(f"Decoded: {decoded}")

# # Batch-wise encoding
# texts_to_encode = data['text_column'].tolist()
# encoded_batch = tokenizer.encode_batch(texts_to_encode, padding=True, max_length=10)
# print(f"Encoded Batch: {encoded_batch}")

# # Batch-wise decoding
# decoded_batch = tokenizer.decode_batch(encoded_batch)
# print(f"Decoded Batch: {decoded_batch}")