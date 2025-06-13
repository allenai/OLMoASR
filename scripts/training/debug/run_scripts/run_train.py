from torch.distributed.run import main as torch_run
import socket

if __name__ == "__main__":
    MODEL = "large"
    GPU_COUNT = 1
    REPLICAS = 1
    ACCUMULATION_STEPS = 4
    BATCH_SIZE = 10
    WORKERS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS
    PRECISION = "float16"
    SEG_COUNT = 27500
    EPOCH_STEPS = SEG_COUNT // EFFECTIVE_BATCH_SIZE
    TRAIN_STEPS = EPOCH_STEPS * 4
    EXP_NAME=f"debug_train_{BATCH_SIZE}bs_{MODEL}_{GPU_COUNT}gpus_{WORKERS}workers_{PRECISION}"
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
        "scripts/training/debug/train.py",
        f"--model_variant={MODEL}",
        f"--exp_name={EXP_NAME}",
        "--job_type=debug",
        "--samples_dicts_dir=/weka/huongn/training_data/debug",
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
        "--persistent_workers=True",
        "--verbose=False",
        f"--precision={PRECISION}",
        "--hardware=H100",
        "--log_dir=/stage",
    ]

    torch_run(command)