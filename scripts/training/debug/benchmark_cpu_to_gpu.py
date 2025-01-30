import torch
import time
import numpy as np

# Set device
device = torch.device("cuda")

# Generate synthetic data with the same shape and dtype as your real data
# (Replace these shapes/dtypes with actual ones from your model)
batch_size = 80
text_seq_len = 448
feature_dim = 384

audio_input = torch.randn(batch_size, 80, 3000, dtype=torch.float32)
text_input = torch.randint(0, 51864, (batch_size, text_seq_len), dtype=torch.long)
text_y = torch.randint(0, 51864, (batch_size, text_seq_len), dtype=torch.long)

padding_mask = torch.zeros((text_seq_len, text_seq_len))
padding_mask[:, len(text_input) :] = -np.inf
causal_mask = (
    torch.empty(text_seq_len, text_seq_len).fill_(-np.inf).triu_(1)
)
padding_mask = padding_mask + causal_mask
padding_mask = padding_mask.unsqueeze(dim=0).repeat(80, 1, 1)[
    :, : text_seq_len, : text_seq_len
]
padding_mask = padding_mask.unsqueeze(dim=1).repeat(1, 6, 1, 1)[
    :, : text_seq_len, : text_seq_len
]
print(f"{audio_input.shape=}, {text_input.shape=}, {padding_mask.shape=}")

# Enable pinned memory (mimics DataLoader behavior)
audio_input = audio_input.pin_memory()
text_input = text_input.pin_memory()
text_y = text_y.pin_memory()
padding_mask = padding_mask.pin_memory()

# Measure transfer time
torch.cuda.synchronize()  # Ensure no prior GPU activity
start_time = time.time()

audio_input = audio_input.to(device)
text_input = text_input.to(device)
text_y = text_y.to(device)
padding_mask = padding_mask.to(device)

torch.cuda.synchronize()  # Ensure all transfers finish
end_time = time.time()

print(f"Data transfer time: {end_time - start_time:.6f} seconds")