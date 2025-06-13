from torch.distributed.run import main as torch_run
import socket

if __name__ == "__main__":
    MODEL = "small"
    GPU_COUNT = 8
    REPLICAS = 1
    ACCUMULATION_STEPS = 1
    BATCH_SIZE = 64
    WORKERS = 8
    PREFETCH_FACTOR = 2
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS
    PRECISION = "float16"
    SHARDING_STRATEGY = "FULL_SHARD"
    SEG_COUNT = 27500
    EPOCH_STEPS = SEG_COUNT // EFFECTIVE_BATCH_SIZE
    TRAIN_STEPS = EPOCH_STEPS * 4
    EXP_NAME=f"fsdp_debug_train_{BATCH_SIZE}bs_{MODEL}_{GPU_COUNT}gpus_{WORKERS}workers_{PRECISION}_{SHARDING_STRATEGY}"
    SAMPLES_DICTS_DIR = "/weka/huongn/training_data/debug"
    RDZV_ENDPOINT = 29400
    s = socket.socket()
    s.bind(("",0))
    FREE_PORT = s.getsockname()[1]
    print(f"Free port: {FREE_PORT}")
    s.close()

    command = [
        f"--nnodes={REPLICAS}",
        f"--nproc_per_node={GPU_COUNT}",
        f"--master_port={FREE_PORT}",
        # "--rdzv-backend=c10d",
        # f"--rdzv-endpoint=localhost:{RDZV_ENDPOINT}",
        "scripts/training/debug/train_fsdp.py",
        f"--model_variant={MODEL}",
        f"--exp_name={EXP_NAME}",
        "--job_type=debug",
        f"--samples_dicts_dir={SAMPLES_DICTS_DIR}",
        f"--train_steps={TRAIN_STEPS}",
        f"--epoch_steps={EPOCH_STEPS}",
        "--lr=1.5e-3",
        "--betas=(0.9, 0.98)",  # This remains a string unless you need to format it differently
        "--eps=1e-6",
        "--weight_decay=0.1",
        "--max_grad_norm=1.0",
        f"--eff_batch_size={EFFECTIVE_BATCH_SIZE}",
        f"--train_batch_size={BATCH_SIZE}",
        f"--num_workers={WORKERS}",
        "--pin_memory=True",
        "--shuffle=True",
        f"--prefetch_factor={PREFETCH_FACTOR}",
        "--persistent_workers=True",
        "--verbose=False",
        f"--precision={PRECISION}",
        f"--sharding_strategy={SHARDING_STRATEGY}",
        "--hardware=H100",
        f"--log_dir=/stage/{EXP_NAME}",
    ]

    torch_run(command)