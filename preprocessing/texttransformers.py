from typing import Any, Dict, Iterator, List, Union
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch


def preprocess_dataframe_batches(
    df: pd.DataFrame,
    batch_size: int,
) -> Iterator[List[str]]:
    """
    Preprocesses a pandas DataFrame row-wise, yielding batches of data.

    Each batch contains strings extracted and flattened from the DataFrame's columns,
    accommodating various data types like str, List[str], and Dict[str, List[str]].

    Args:
        df (pd.DataFrame): The input DataFrame with columns containing str, List[str], or Dict[str, List[str]].
        batch_size (int): The batch size for processing.

    Yields:
        Iterator[List[str]]: An iterator that yields batches of preprocessed data ready for tokenization.
    """
    preprocessed_batch: List[str] = []
    for index, row in df.iterrows():
        extracted_items: List[str] = []
        try:
            for item in row.dropna():
                if isinstance(item, str):
                    extracted_items.append(item)
                elif isinstance(item, list):
                    extracted_items.extend([str(elem) for elem in item])
                elif isinstance(item, dict):
                    for value in item.values():
                        if isinstance(value, list):
                            extracted_items.extend([str(elem) for elem in value])
                        else:
                            extracted_items.append(str(value))
                else:
                    extracted_items.append(str(item))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue  # Skip this row and proceed
        preprocessed_batch.extend(extracted_items)
        while len(preprocessed_batch) >= batch_size:
            yield preprocessed_batch[:batch_size]
            preprocessed_batch = preprocessed_batch[batch_size:]
    if preprocessed_batch:
        yield preprocessed_batch


def process_dataframe(
    df: pd.DataFrame,
    tokenizer_name_or_path: str,
    batch_size: int,
    max_length: int = 128,
    padding: Union[bool, str] = 'max_length',
    truncation: bool = True,
    use_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Processes a DataFrame by tokenizing its contents in batches.

    This function handles encoding and decoding of text data using a tokenizer from the transformers library.
    It processes the data in batches to handle large datasets efficiently.

    Args:
        df (pd.DataFrame): The input DataFrame.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use.
        batch_size (int): The number of samples to process in each batch.
        max_length (int, optional): The maximum sequence length. Defaults to 128.
        padding (Union[bool, str], optional): Padding strategy ('max_length', 'longest', True, False). Defaults to 'max_length'.
        truncation (bool, optional): Whether to truncate sequences longer than max_length. Defaults to True.
        use_special_tokens (bool, optional): Whether to include special tokens in encoding/decoding. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the encoded data, decoded data, and the tokenizer used.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    data_batches = preprocess_dataframe_batches(df, batch_size=batch_size)

    all_encoded_data: List[BatchEncoding] = []
    all_decoded_data: List[str] = []

    for data_batch in data_batches:
        try:
            encoded_batch = tokenizer.batch_encode_plus(
                data_batch,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors='pt',
                return_special_tokens_mask=True,
                add_special_tokens=use_special_tokens,
            )
            decoded_batch = tokenizer.batch_decode(
                encoded_batch['input_ids'],
                skip_special_tokens=not use_special_tokens,
                clean_up_tokenization_spaces=True,
            )
            all_encoded_data.append(encoded_batch)
            all_decoded_data.extend(decoded_batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue  # Skip this batch and proceed

    return {
        'encoded_data': all_encoded_data,
        'decoded_data': all_decoded_data,
        'tokenizer': tokenizer,
    }


# # Example usage
# if __name__ == "__main__":
#     # Create a sample DataFrame
#     data = {
#         'column1': [
#             'This is a sample string.',
#             ['List', 'of', 'strings'],
#             {'key1': ['List', 'in', 'dict'], 'key2': 'String in dict'}
#             ,
#             'Another sample string',
#             'Another sample string',
#             'Another sample string',
#             ['More', 'sample', 'strings'],
#             {'keyA': ['More', 'lists'], 'keyB': 'Another dict string'}
#         ],
#         'column2': [
#             'Another sample string',
#             ['More', 'sample', 'strings'],
#             {'keyA': ['More', 'lists'], 'keyB': 'Another dict string'}
#         ,
        
#             'Another sample string',
#             'Another sample string',
#             'Another sample string',
#             ['More', 'sample', 'strings'],
#             {'keyA': ['More', 'lists'], 'keyB': 'Another dict string'}
        
#         ]
#     }
#     df = pd.DataFrame(data)

#     # Process the DataFrame
#     result = process_dataframe(
#         df=df,
#         tokenizer_name_or_path='bert-base-uncased',
#         batch_size=2,
#         max_length=10,
#         padding='max_length',
#         truncation=True,
#         use_special_tokens=True,
#     )

#     # Access the results
#     encoded_data = result['encoded_data']
#     decoded_data = result['decoded_data']

#     # Output the results
#     print("Encoded Data:")
#     for batch in encoded_data:
#         print(batch)

#     print("\nDecoded Data:")
#     for text in decoded_data:
#         print(text)