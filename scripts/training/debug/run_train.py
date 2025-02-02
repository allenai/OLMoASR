from torch.distributed.run import main as torch_run

if __name__ == "__main__":
    command = [
        "--nnodes=1",
        "--nproc_per_node=1",
        "scripts/training/debug/train_ddp.py",
        "--model_variant=tiny",
        "--exp_name=debug_evals_in_train",
        "--job_type=train",
        "--samples_dicts_dir=/weka/huongn/samples_dicts/debug",
        "--train_steps=85",
        "--epoch_steps=340",
        "--ckpt_dir=/weka/huongn/ow_ckpts",
        "--log_dir=/results/huongn/ow_logs",
        "--eval_dir=/ow_eval",
        "--run_id_dir=/weka/huongn/ow_run_ids",
        "--lr=1.5e-3",
        "--betas=(0.9, 0.98)",  # This remains a string unless you need to format it differently
        "--eps=1e-6",
        "--weight_decay=0.1",
        "--max_grad_norm=1.0",
        "--eff_batch_size=320",
        "--train_batch_size=80",
        "--eval_batch_size=8",
        "--num_workers=2",
        "--pin_memory=True",
        "--shuffle=True",
        "--persistent_workers=True",
        "--run_eval=True",
        "--train_log_freq=1000",
        "--eval_freq=1",
        "--ckpt_freq=1000",
        "--verbose=False",
        "--precision=float16",
        "--hardware=H100",
    ]

    torch_run(command)