from transformers import AutoTokenizer

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"Length of tokenizer vocabulary before adding special tokens: {len(tokenizer)}")
print(tokenizer)



functionality=[func for func in dir(tokenizer) if not func.startswith("__") and not func.startswith("_") and not func.startswith("___") ]
print(functionality)
# Print the special tokens
print(tokenizer.name_or_path)

# # Custom settings for the tokenizer
tokenizer.unk_token = "<unk>"  # Unknown token
tokenizer.bos_token = "<bos>"  # Beginning of sequence token
tokenizer.eos_token = "<eos>"  # End of sequence token
tokenizer.pad_token = "<pad>"  # Padding token

# Add special tokens to handle the above tokens if they aren't already in the tokenizer
special_tokens_dict = {
    'additional_special_tokens': ["<bos>", "<eos>", "<unk>", "<pad>"]
}
tokenizer.add_special_tokens(special_tokens_dict)

print(f"Length of tokenizer vocabulary after adding special tokens: {len(tokenizer)}")

# Sample text to tokenize
sample_text = """
You're right, and I apologize for the oversight. 
The issue is that the current implementation is not 
properly separating the input and target sequences 
in the tokenization process. Let's modify the 
`AdvancedTokenizer` class to correctly handle 
both input and target sequences. Here's an updated 
version that should resolve the issue:

This updated implementation should correctly 
handle both input and target sequences, 
and the output should now properly show the 
decoded text for both input and target. 
The `labels` tensor now contains the tokenized target 
sequences, while the `input_ids` tensor contains the 
tokenized input sequences.
"""

# Arguments for tokenization:
add_special_tokens = True  # We will manually handle special tokens
padding = False  # No padding applied; sequences will not be padded to the same length
truncation = False  # No truncation applied; the full sequence is used
max_length = None  # No maximum length specified; use the full sequence
stride = 10  # No stride applied; no overlap in chunks
return_tensors = None  # Output is not converted to tensors; remains as token IDs

# Tokenize the text with custom settings
tokens = tokenizer.encode(
    sample_text,
    add_special_tokens=add_special_tokens,
    padding=padding,
    truncation=truncation,
    max_length=max_length,
    stride=stride,
    return_tensors=return_tensors
)
print(f"Token IDs:\n {tokens}")

# Optionally, add the bos_token at the start of the sequence
add_bos_token = True
if add_bos_token:
    tokens = [tokenizer.bos_token_id] + tokens

# Add the eos_token at the end of the sequence
tokens.append(tokenizer.eos_token_id)

# Decode the tokens back into text to verify the process
decode_text = tokenizer.decode(tokens)
print(f"Decoded Text: {decode_text}")

# Example: Padding multiple sequences with different lengths (if required)
sequences = [
    "Hello, world!",
    "Goodbye!"
]

# Automatically pad the sequences to the length of the longest one
encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(f"Padded Sequences:\n{encoded_sequences}")
