# OLMoASR: Open Models and Data for Training Robust Speech Recognition Models
This repository serves to illustrate the steps taken to train OLMoASR models, all the way from the initial data processing to evaluating the model.

## Contents
- [Data](#data)
- [Quickstart](#quickstart)
  - [Setup](#setup)
  - [Data Processing](#data-processing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Available Models](#available-models)
- [Usage](#usage)
- [Team](#team)
- [License](#license)
- [Citing](#citing)

## Data
Before starting the Quickstart tutorial, you'll need to download the data (audio-transcript pairs) and organize it in a directory structure as elaborated below to continue the data processing step:

```
shard_00000/
├── pair_id_1/
│   ├── audio_pair_id_1.ext
│   └── transcript_pair_id_1.ext
├── pair_id_2/
│   ├── audio_pair_id_2.ext
│   └── transcript_pair_id_2.ext
├── pair_id_3/
│   ├── audio_pair_id_3.ext
│   └── transcript_pair_id_3.ext
├── pair_id_4/
│   ├── audio_pair_id_4.ext
│   └── transcript_pair_id_4.ext
└── ...
```

You can download the data from [OLMoASR-Pool HuggingFace](link).

## Quickstart
In the following subsections, we'll walk through how to setup, process the data, train a model and evaluate it.

### Setup
To have full access, ensure you have `python >= 3.10` and a virtual environment. Then, run

```[shell]
git clone https://github.com/allenai/OLMoASR.git
pip install -r requirements/requirements.txt
pip install -e .
```

We use `ffmpeg` in data processing and `wandb` to log training, so please ensure that you have those dependencies fulfilled.

### Data Processing
Once you've downloaded and organized your data, you'll need to follow the following steps to process your data:
1. Transform all your transcripts into JSONL format to be suitable for tagging and filtering using `scripts/data/processing/text_to_jsonl.py`
2. Segment all your full-length audio files into 30s-long audio chunks using `olmoasr/preprocess.py`
3. Perform document-level tagging using `scripts/data/filtering/data_tagger.py`
4. Segment transcript files into 30s-long transcript chunks using `olmoasr/preprocess.py`
5. Perform segment-level tagging using `scripts/data/filtering/data_tagger.py`
6. Perform audio-text language alignment using `scripts/data/filtering/assign_audio_lang_data.py` and `scripts/data/filtering/tag_audio_lang.py`
7. Filter based on a specified configuration of conditions using `scripts/data/filtering/process_tagged_data.py`

### Training
To enable distributed training, we use `torchrun`. Below is an example of a bash script you'll use to execute distributed training.

```[shell]
# REPLICAS - number of compute nodes
# GPU_COUNT - number of GPUs
# SCRIPT - train (DDP) or train (FSDP)
torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} ${SCRIPT} \
      --model_variant=${MODEL_SIZE} \ # size of model you're training
      --exp_name=${EXP_NAME} \ # experiment name
      --job_type=${JOB_TYPE} \ # type of job (e.g debug, filtering, tuning)
      --samples_dicts_dir=${SAMPLES_DICTS_DIR} \ # directory where data lives
      --train_steps=${TRAIN_STEPS} \ # total steps for training
      --epoch_steps=${EPOCH_STEPS} \ # steps for training per epoch
      --ckpt_file_name=None \ # KEEP None, this will be automatically generated
      --ckpt_dir=${CKPT_DIR} \ # where to save the checkpoint
      --log_dir=${LOG_DIR} \ # where to log wandb and other things you want to log
      --eval_dir=${EVAL_DIR} \ # directory where eval datasets live
      --run_id_dir=${RUN_ID_DIR} \ # directory where wandb run_ids are cached
      --lr=${LEARNING_RATE} \ # learning rate
      --betas=${BETAS} \ # beta values
      --eps=${EPS} \ # epsilon value
      --weight_decay=${WEIGHT_DECAY} \ # weight decay value
      --max_grad_norm=${MAX_GRAD_NORM} \ # max clipping grad norm
      --eff_batch_size=${EFFECTIVE_BATCH_SIZE} \ # global batch size (across GPUs)
      --train_batch_size=${BATCH_SIZE} \ # per GPU batch size
      --eval_batch_size=${EVAL_BATCH_SIZE} \ # per GPU batch size for running evals
      --num_workers=${NUM_WORKERS} \ # number of dataloader workers
      --prefetch_factor=${PREFETCH_FACTOR} \ # prefetch factor
      --pin_memory=${PIN_MEMORY} \ # whether to pin memory
      --shuffle=${SHUFFLE} \ # shuffle data in DistributedSampler
      --persistent_workers=${PERSISTENT_WORKERS} \ # whether to have persistent workers
      --run_eval=${RUN_EVAL} \ # whether to run evaluation in training loop
      --train_log_freq=${TRAIN_LOG_FREQ} \ # frequency to log training results to wandb
      --eval_freq=${EVAL_FREQ} \ # frequency to run evaluation in loop
      --ckpt_freq=${CKPT_FREQ} \ # frequency to save checkpoints
      --verbose=${VERBOSE} \ # verbose setting for debugging
      --precision=${PRECISION} \ # precision type
      --hardware=${HARDWARE} \ # type of hardware training on (for efficiency tracking)
      --async_eval=${ASYNC_EVAL} \ # whether to do asynchronous evaluation
      --eval_script_path=${EVAL_SCRIPT_PATH} \ # path to evaluation script (for async eval)
      --eval_wandb_log=${EVAL_WANDB_LOG} \ # whether to log to wandb for evals (for async eval)
      --eval_on_gpu=${EVAL_ON_GPU}" # whether to run async eval on GPU or CPU
```

You can go to `scripts/training` for a more detailed guide on the bash scripts that use `torchrun` to train.

### Evaluation


### Inference

## Available Models

### OLMoASR (Open weights, code, data) vs. Whisper (Open weights, closed training code, data)

| Model               | Libri-clean | Libri-other | TED-LIUM3 | WSJ | CallHome | Switchboard | CV5.1 | Artie | CORAAL | CHiME6 | AMI-IHM | AMI-SDM | VoxPop | Fleurs | Avg  |
|--------------------|-------------|-------------|-----------|-----|----------|-------------|-------|-------|--------|--------|---------|---------|--------|--------|------|
| OLMoASR-39M        | **5.1**     | **12.3**    | **5.5**   | 5.6 | 23.9     | 18.7        | 25.1  | 19.3  | 25.7   | 45.2   | 24.2    | 55.4    | 11.6   | 9.7    | **20.5** |
| Whisper tiny.en    | 5.6         | 14.6        | 6.0       | **5.0** | 24.1     | **17.8**    | **26.3**  | **20.0**  | **23.9**   | **41.3**   | **23.7**    | **50.3**    | **11.7**   | **11.6**   | 20.1 |
| OLMoASR-74M        | **3.7**     | **9.0**     | **4.6**   | **4.3** | 20.5     | **14.0**    | **18.5**  | **13.6**  | **21.5**   | **38.0**   | **20.4**    | **47.8**    | 9.7    | **6.7**    | **16.6** |
| Whisper base.en    | 4.2         | 10.2        | 4.9       | 4.6 | **20.9**     | 15.2        | 19.0  | 13.4  | 22.6   | 36.4   | 20.5    | 46.7    | **10.0**   | 7.6    | 16.9 |
| OLMoASR-244M       | **3.0**     | **7.0**     | 4.2       | 3.8 | **16.7**     | **13.2**    | **13.1**  | **9.6**   | 19.6   | 30.6   | 18.7    | 39.9    | 8.7    | **5.0**    | **13.8** |
| Whisper small.en   | 3.1         | 7.4         | **4.0**   | **3.3** | 18.2     | 15.7        | 13.1  | 9.7   | **20.2**   | **27.6**   | **17.5**    | **38.0**    | **8.1**   | 6.0    | 13.7 |
| OLMoASR-769M       | 3.5         | **5.7**     | **5.0**   | 3.6 | 14.3     | **12.7**    | 11.3  | **7.5**   | 18.7   | 28.5   | **16.9**    | 38.3    | 8.4    | **4.4**    | **12.8** |
| Whisper medium.en  | **3.1**     | 6.3         | 4.1       | **3.3** | **16.2**     | 14.1        | **10.6**  | 7.6   | **17.5**   | **25.3**   | 16.4    | **37.2**    | **7.4**   | 5.0    | 12.4 |
| OLMoASR-1.5B       | **2.6**     | **5.9**     | **4.5**   | 3.7 | 16.5     | **12.7**    | 11.1  | **7.9**   | 18.7   | **30.7**   | **16.4**    | 38.8    | 8.1    | **4.5**    | **13.0** |
| OLMoASR-1.5B-v2    | **2.7**     | **5.6**     | 4.2       | 3.6 | **15.0**     | **11.7**    | 11.1  | 7.8   | **18.1**   | 29.4   | **17.1**    | 38.0    | **8.0**   | **4.2**    | **12.6** |
| Whisper large-v1   | 2.7         | 5.6         | **4.0**   | **3.1** | 15.8     | 13.1        | **9.5**   | 6.7   | 19.4   | **25.6**   | 16.4    | **36.9**    | 7.3    | 4.6    | 12.2 |
| Whisper large-v2   | 2.7         | 5.2         | 4.0       | 3.9 | **17.6**     | 13.8        | 9.0   | 6.2   | 16.2   | 25.5   | 16.9    | 36.4    | 7.3    | 4.4    | 12.1 |
| Whisper large-v3   | **2.0**     | **3.9**     | 3.9       | **3.5** | 13.2     | 13.2        | 8.4   | 5.9   | 18.7   | 26.8   | 16.0    | 34.2    | 9.5    | 4.0    | **11.7** |
| Whisper large-v3-turbo | 2.2     | 4.2         | **3.5**   | **3.5** | **13.2**     | **12.9**    | **9.7**   | **6.3**   | **18.6**   | **27.3**   | **16.1**    | **35.2**    | **12.2**   | **4.4**    | **12.1** |

## Usage

## Team

## License

## Citing