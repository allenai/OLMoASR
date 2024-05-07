# %%
import torch
import os

# %%
checkpoint = torch.load(
    "checkpoints/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99.pt"
)
checkpoint

# %%
checkpoint.keys()

# %%
print(*checkpoint["model_state_dict"].keys(), sep="\n")

# %%
checkpoint["model_state_dict"]["decoder.token_embedding.weight"]

# %%
checkpoint["model_state_dict"]["decoder.token_embedding.weight"].shape

# %%
for param in checkpoint["model_state_dict"].items():
    print(param[0])
    print(param[1].shape)

# %%
checkpoint["model_state_dict"]["decoder.token_embedding.weight"][:-1].shape

# %%
checkpoint["model_state_dict"]["decoder.token_embedding.weight"][:-1]

# %%
new_dec_tok_emb_w = checkpoint["model_state_dict"]["decoder.token_embedding.weight"][
    :-1
]
new_dec_tok_emb_w

# %%
# will remove last token embedding from decoder token embedding weight (padding token embedding - not needed for inference)
checkpoint["model_state_dict"]["decoder.token_embedding.weight"] = new_dec_tok_emb_w
print(checkpoint["model_state_dict"]["decoder.token_embedding.weight"].shape)
checkpoint["model_state_dict"]["decoder.token_embedding.weight"]

# %%
torch.save(
    checkpoint,
    "checkpoints/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt"
)

# %%
