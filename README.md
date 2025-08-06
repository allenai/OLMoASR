# OLMoASR: Open Models and Data for Training Robust Speech Recognition Models
This repository serves to illustrate the steps taken to train OLMoASR models, all the way from the initial data processing to evaluating the model.

## Contents
- [OLMoASR: Open Models and Data for Training Robust Speech Recognition Models](#olmoasr-open-models-and-data-for-training-robust-speech-recognition-models)
  - [Contents](#contents)
  - [Data](#data)
  - [Quickstart](#quickstart)
    - [Setup](#setup)
    - [Data Processing and Filtering](#data-processing-and-filtering)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Available Models](#available-models)
    - [Short-form Speech Recognition](#short-form-speech-recognition)
    - [Long-form Speech Recognition](#long-form-speech-recognition)
  - [Usage](#usage)
  - [Team and Acknowledgements](#team-and-acknowledgements)
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
To have full access, ensure you have `python >= 3.8` and a virtual environment. Then, run:

```[shell]
git clone https://github.com/allenai/OLMoASR.git
pip install -r requirements/requirements.txt
pip install -e .
```

We use `ffmpeg` in data processing and `wandb` to log training, so please ensure that you have those dependencies fulfilled.

### Data Processing and Filtering
Once you've downloaded and organized your data, you'll need to follow the following steps to process your data:
1. Transform all your transcripts into JSONL format to be suitable for tagging and filtering using `scripts/data/processing/text_to_jsonl.py`
2. Segment all your full-length audio files into 30s-long audio chunks using `olmoasr/preprocess.py`
3. Perform document-level tagging using `scripts/data/filtering/data_tagger.py`
4. Segment transcript files into 30s-long transcript chunks using `olmoasr/preprocess.py`
5. Perform segment-level tagging using `scripts/data/filtering/data_tagger.py`
6. Perform audio-text language alignment using `scripts/data/filtering/assign_audio_lang_data.py`, `scripts/data/filtering/tag_audio_lang.py` and `scripts/data/filtering/data_tagger.py`
7. Filter based on a specified configuration of conditions using `scripts/data/filtering/process_tagged_data.py`
8. (Optional) Randomly subsample from filtered data mix to get training data

Steps 2 and 3 can be performed concurrently if you have the available compute.
Step 6 is technically a tagging task as well, but involves more complex steps than heuristics-based tagging.

Your data should be a JSONL file where each line is in the following format:

```
{
	"id": <str>, # unique identifier for audio-transcript pair
	"seg_id": <str>, # unique identifier for segment audio-transcript pair
	"subtitle_file": <str>, # path where transcript file is located (segmented/unsegmented depending on which step of data processing you're at)
	"audio_file": <str>, # path where audio file is located (segmented/unsegmented depending on which step of data processing you're at)
	"timestamp": <str>, # start and end times of segment
	"mach_timestamp": <str>, # optional - if you have an associated machine transcript, start and end times of associated machine transcript segment
	"seg_text": <str> , # cleaned text in the transcript segment
	"mach_seg_text": <str>, # optional - cleaned text in machine transcript segment
	"seg_content": <str>, # raw text in the transcript segment
	"mach_seg_content": <str>, # raw text in machine transcript segment
	"edit_dist": <float>, # optional - document-level WER between machine and manually-uploaded transcript
	"seg_edit_dist": <float>, # optional - segment-level WER between machine and manually-uploaded transcript
	"audio_lang": <str>, # language in audio
	"text_lang": <str>, # language in transcript
	"casing": <str>, # optional - dominant casing of transcript
	"repeating_lines": <bool>, # optional - presence of repeating lines in transcript
	"length": <float>, # duration of audio
	"num_words": <int>, # number of words in transcript (cleaned text)
	"seg_num_words": <int> # number of words in transcript segment (cleaned text)
}
```

### Training
Once you've processed your data, you are ready to train a model with it. To enable distributed training, we use `torchrun`. Below is an example of a bash script you'll use to execute distributed training:

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

You can go to `scripts/training` for a more detailed guide on the bash scripts that use `torchrun` to train and some example training scripts.

### Evaluation
To run evaluation, you'll have to acquire the evaluation sets first. With the exception of evaluation sets that need to be paid for and Artie Bias Corpus[^*], you can use `scripts/eval/get_eval_set.py` to download the dataset by just passing in the dataset name.

[^*]: This dataset no longer exists online from the original source. If you'd like a copy of the evaluation set, please visit [OLMoASR HuggingFace](link)

After that, you can run `scripts/eval/eval.py` to run evaluation. Please visit `scripts/eval` for more information on the evaluation sets, and other scripts.

## Available Models
OLMoASR is a series of ASR models trained on OLMoASR-Pool, a web-scale 3M hour audio-text dataset collected from the public internet. They can all perform English short and long-form speech recognition and produce sentence-level timestamps.

Model checkpoints can be downloaded from [OLMoASR HuggingFace](link). 

### Short-form Speech Recognition

| Dataset                  | OLMoASR-tiny.en | OLMoASR-base.en | OLMoASR-small.en | OLMoASR-medium.en | OLMoASR-large.en | OLMoASR-large.en-v2 |
|--------------------------|-------------|-------------|---------------|---------------|---------------|------------------|
| Librispeech-test.clean   | 5.1         | 3.7         | 3.0           | 3.5           | 2.6           | 2.7              |
| Librispeech-test.other   | 12.3        | 9.0         | 7.0           | 5.7           | 5.9           | 5.6              |
| TED-LIUM3                | 5.5         | 4.6         | 4.2           | 5.0           | 4.5           | 4.2              |
| WSJ                      | 5.6         | 4.3         | 3.8           | 3.6           | 3.7           | 3.6              |
| CallHome                 | 23.9        | 20.5        | 16.7          | 14.3          | 16.5          | 15.0             |
| Switchboard              | 18.7        | 14.0        | 13.2          | 12.7          | 12.7          | 11.7             |
| CommonVoice5.1              | 25.1        | 18.5        | 13.1          | 11.3          | 11.1          | 11.1             |
| Artie                    | 19.3        | 13.6        | 9.6           | 7.5           | 7.9           | 7.8              |
| CORAAL                   | 25.7        | 21.5        | 19.6          | 18.7          | 18.7          | 18.1             |
| CHiME6                   | 45.2        | 38.0        | 30.6          | 28.5          | 30.7          | 29.4             |
| AMI-IHM                  | 24.2        | 20.4        | 18.7          | 16.9          | 16.4          | 17.1             |
| AMI-SDM                  | 55.4        | 47.8        | 39.9          | 38.3          | 38.8          | 38.0             |
| VoxPopuli                   | 11.6        | 9.7         | 8.7           | 8.4           | 8.1           | 8.0              |
| Fleurs                   | 9.7         | 6.7         | 5.0           | 4.4           | 4.5           | 4.2              |
| Average                  | 20.5        | 16.6        | 13.8          | 12.8          | 13.0          | 12.6             |

### Long-form Speech Recognition

| Dataset        | OLMoASR-tiny.en | OLMoASR-base.en | OLMoASR-small.en | OLMoASR-medium.en | OLMoASR-large.en | OLMoASR-large.en-v2 |
|----------------|-------------|-------------|---------------|---------------|---------------|------------------|
| TED-LIUM3      | 4.8         | 3.9         | 3.6           | 3.3           | 3.5           | 3.6              |
| Meanwhile      | 12.6        | 10.2        | 7.4           | 6.9           | 8.8           | 10.0             |
| Kincaid46      | 13.6        | 11.2        | 10.2          | 9.4           | 10.0          | 10.1             |
| Rev16          | 14.0        | 12.0        | 11.5          | 12.5          | 11.5          | 11.1             |
| Earnings-21    | 14.2        | 11.1        | 10.1          | 9.5           | 9.9           | 9.8              |
| Earnings-22    | 20.0        | 15.6        | 14.0          | 13.5          | 13.5          | 13.5             |
| CORAAL         | 30.2        | 26.1        | 23.4          | 21.9          | 22.4          | 22.1             |
| Average        | 15.6        | 12.9        | 11.5          | 11.0          | 11.4          | 11.5             |

## Usage

Currently, only Python usage is supported. CLI usage support is in development. To run transcription, you can run the code below:

```
import olmoasr

model = olmoasr.load_model("medium", inference=True)
result = model.transcribe("audio.mp3")
print(result)
```

## Team and Acknowledgements
Team (* = equal contrib): Huong Ngo, Matt Deitke, Martijn Bartelds, Sarah Pratt,
Josh Gardner*, Matt Jordan*, Ludwig Schmidt*

Code is developed with the assistance of OpenAI's Whisper code. We are grateful to Ai2 and UW for resource support, OpenAI for open-sourcing a portion of their code and making their pre-trained checkpoints available, and Jong Wook Kim for clarifications throughout the project.

## License

## Citing
Coming soon.