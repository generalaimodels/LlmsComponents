import torch
import Neural_Network as nn

# Define the LSTM
rnn = nn.LSTM(
    input_size=512,
    hidden_size=128,
    num_layers=32,
    bias=True,
    batch_first=True,
    dropout=0.0,
    bidirectional=True
)

# Generate input tensor with shape (batch_size, seq_length, input_size)
input_tensor = torch.randn(10, 40, 512)

# Initialize hidden state (h0) and cell state (c0) tensors with shape (num_layers * num_directions, batch_size, hidden_size)
h0 = torch.randn(32 * 2, 10, 128)  # (num_layers * num_directions, batch_size, hidden_size)
c0 = torch.randn(32 * 2, 10, 128)  # (num_layers * num_directions, batch_size, hidden_size)

# Pass the input and hidden states through the LSTM
output, (hn, cn) = rnn(input_tensor, (h0, c0))
print(rnn)
# Print the shape of the output tensor
print(output[0].shape)  # Should be (batch_size, seq_length, num_directions * hidden_size)

print(hn.shape, cn.shape)





# import torch
# from LLMs_model.Mistral_Model import Mistral_model, ModelArgs, LoraArgs
# args = ModelArgs(
#     dim=512,
#     n_layers=12,
#     head_dim=64,
#     hidden_dim=2048,
#     n_heads=8,
#     n_kv_heads=8,
#     norm_eps=1e-6,
#     vocab_size=10000,
#     rope_theta=10000.0,
#     lora=LoraArgs(
#         enable=True,
#         rank=16,
#         dropout=0.0,
#         scaling=2.0,
#     ),
#     moe=None,
# )
# print(args)
# print(args.vocab_size)

# # Initialize the model
# model = Mistral_model(args)

# # Create dummy input
# batch_size = 2
# seq_length = 10
# input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_length))

# # Flatten the input_ids
# input_ids_flat = input_ids.reshape(-1)

# # Create dummy sequence lengths
# seqlens = [seq_length] * batch_size

# # Forward pass
# output = model(input_ids_flat, seqlens)

# # Print output shape
# print(f"Output shape: {output.shape}")

# # Check if the output is valid
# assert output.shape == (batch_size * seq_length, args.vocab_size), "Output shape mismatch"
# print("Test passed successfully!")


# import torch
# from Multi_data_understand import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]