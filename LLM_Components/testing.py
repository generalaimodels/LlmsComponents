import torch
import Neural_Network as nn
from xformers.components.attention.compositional import  CompositionalAttention
from xformers.components.attention.base import Attention


model=CompositionalAttention(
    dim_model=512,
    num_heads=8,
    
)


q=torch.rand(1,10,512)
v=torch.rand(1,10,512)
k=torch.rand(1,10,512)
output=model(q,k,v)
print(output.shape)

# Create an embedding module containing 10 tensors of size 3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# Example input: a batch of 2 samples of 4 indices each
input_indices = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# Retrieve the embeddings for the given indices
output = embedding(input_indices)
print("Embeddings for input indices:")
print(output)

# Example with padding_idx
embedding_with_padding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
input_indices_with_padding = torch.LongTensor([[0, 2, 0, 5]])
output_with_padding = embedding_with_padding(input_indices_with_padding)
print("Embeddings with padding_idx:")
print(output_with_padding)

# Changing the padding vector
padding_idx = 0
embedding_with_custom_padding = nn.Embedding(num_embeddings=3, embedding_dim=3, padding_idx=padding_idx)
with torch.no_grad():
    embedding_with_custom_padding.weight[padding_idx] = torch.ones(3)
input_indices_custom_padding = torch.LongTensor([[0, 1, 2]])
output_custom_padding = embedding_with_custom_padding(input_indices_custom_padding)
print("Embeddings with custom padding vector:")
print(output_custom_padding)


import torch
import torch.nn as nn

# Create an embedding module with various parameters
embedding = nn.Embedding(
    num_embeddings=10000,  # Vocabulary size
    embedding_dim=3,       # Embedding dimensionality
    padding_idx=0,         # Padding index
    max_norm=1.0,          # Maximum norm for embeddings
    norm_type=2,           # Type of norm (Euclidean)
 scale_grad_by_freq=True,  # Scale gradients by frequency
    sparse=False,          # Dense gradients
    device='cpu',          # Create on cpu
    dtype=torch.float32    # Data type
)
# Example input: a batch of 2 samples of 4 indices each
input_indices = torch.LongTensor([[0, 2, 4, 5], [4, 3, 2, 9]]).to('cpu')
# Retrieve the embeddings for the given indices
output = embedding(input_indices)
print(output)
