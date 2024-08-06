import os
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize_model_parallel, model_parallel_cuda_manual_seed
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from llama3_model import ModelArgs, Transformer

def main():
    # Specify the file path for rendezvous
    rendezvous_file = 'file:///E:/LLMS/Fine-tuning/somefile'

    # Initialize Torch distributed
    print("Initializing distributed process group...")
    dist.init_process_group(backend='gloo', init_method=rendezvous_file, rank=0, world_size=1)
    print("Distributed process group initialized.")

    # Initialize model parallel
    print("Initializing model parallel...")
    initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    print("Model parallel initialized.")

    # Initialize model arguments
    params = ModelArgs()
    print(f"Model parameters: {params}")

    # Instantiate the model
    model = Transformer(params=params)
    print(f"Model instantiated: {model}")

    # Create a batch of token sequences (32 sequences of length 200 with random token IDs)
    tokens = torch.randint(0, params.vocab_size, (32, 200), dtype=torch.long)
    print(f"Generated tokens: {tokens.shape}")

    # Perform a forward pass
    output = model(tokens, start_pos=0)
    print(f"Output shape: {output.shape}")

    # Check if the output shape is as expected
    expected_shape = (32, 200, params.vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward pass successful, output shape is correct.")

if __name__ == "__main__":
    main()
