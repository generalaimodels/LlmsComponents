import torch
from LLMs_model.Mistral_Model import Mistral_model, ModelArgs, LoraArgs
args = ModelArgs(
    dim=512,
    n_layers=12,
    head_dim=64,
    hidden_dim=2048,
    n_heads=8,
    n_kv_heads=8,
    norm_eps=1e-6,
    vocab_size=10000,
    rope_theta=10000.0,
    lora=LoraArgs(
        enable=True,
        rank=16,
        dropout=0.0,
        scaling=2.0,
    ),
    moe=None,
)
print(args)
print(args.vocab_size)

# Initialize the model
model = Mistral_model(args)

# Create dummy input
batch_size = 2
seq_length = 10
input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_length))

# Flatten the input_ids
input_ids_flat = input_ids.reshape(-1)

# Create dummy sequence lengths
seqlens = [seq_length] * batch_size

# Forward pass
output = model(input_ids_flat, seqlens)

# Print output shape
print(f"Output shape: {output.shape}")

# Check if the output is valid
assert output.shape == (batch_size * seq_length, args.vocab_size), "Output shape mismatch"
print("Test passed successfully!")


import torch
from Multi_data_understand import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]