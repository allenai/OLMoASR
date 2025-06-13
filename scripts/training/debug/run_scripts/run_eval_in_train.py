from torch.distributed.run import main as torch_run
import os

if __name__ == "__main__":
    os.environ["WANDB_DIR"] = ""
    os.environ["HF_TOKEN"] = ""
    command = [
        "--nnodes=1",
        "--nproc_per_node=1",
        "scripts/training/debug/eval_in_train.py",
        "--model_variant=tiny",
        "--exp_name=debug_evals_in_train",
        "--ckpt_file_name=/weka/huongn/ow_ckpts/text_heurs_1_manmach_tiny_15e4_440K_bs64_ebs512_12workers_5pass_013025_lbn12dil/checkpoint_00005000_tiny_ddp-train_grad-acc_fp16_ddp.pt",
        "--ckpt_dir=/weka/huongn/ow_ckpts",
        "--eval_dir=/weka/huongn/ow_eval",
        "--eval_batch_size=64",
        "--num_workers=2",
        "--pin_memory=True",
        "--persistent_workers=True",
    ]

    torch_run(command)