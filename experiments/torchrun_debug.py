from torch.distributed.run import main as torch_run

if __name__ == "__main__":
    command = [
        "--nnodes=1",
        "--nproc_per_node=2",
        "--master-port=29504",
        "scripts/training/train_wds.py",
        "--model_variant=tiny",
        "--exp_name=ow_tiny_wds",
        "--job_type=train",
        "--train_shards=/wds_shards/{000000..000001}.tar",
        "--train_steps=1000",
        "--val_shards=/wds_shards/073468.tar",
        "--run_id=None",
        "--ckpt_file_name=None",
        "--rank=None",
        "--world_size=None",
        "--lr=1.5e-3",
        "--betas=(0.9, 0.98)",
        "--eps=1e-6",
        "--weight_decay=0.1",
        "--max_grad_norm=1.0",
        "--eff_batch_size=256",
        "--train_batch_size=8",
        "--val_batch_size=8",
        "--eval_batch_size=8",
        "--num_workers=4",
        "--pin_memory=True",
        "--persistent_workers=False",
        "--run_val=True",
        "--run_eval=True"
    ]
    torch_run(command)