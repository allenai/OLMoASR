import torch
from fire import Fire

def gen_inf_ckpt(ckpt_path: str, save_path: str):
    checkpoint = torch.load(ckpt_path)
    new_dec_tok_emb_w = checkpoint["model_state_dict"]["decoder.token_embedding.weight"][:-1]
    checkpoint["model_state_dict"]["decoder.token_embedding.weight"] = new_dec_tok_emb_w
    checkpoint["dims"] = checkpoint["dims"].__dict__
    torch.save(checkpoint, save_path)
    print(f"Saved inference checkpoint to {save_path}")
    return save_path

if __name__ == "__main__":
    Fire(gen_inf_ckpt)