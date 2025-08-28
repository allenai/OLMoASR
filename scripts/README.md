## OLMoASR Pipeline

In this directory, you'll find the scripts to transform your raw data, train your model and evaluate it. 

### Data Processing
You can refer to the `scripts/data` and `olmoasr` directory, where `olmoasr/preprocess.py` contains segmentation code, and `scripts/data/processing/text_to_jsonl.py` is a script to organize a directory of transcript files into a JSONL file.

### Filtering
`scripts/data/filtering/*` contains scripts to filter your processed data
- `scripts/data/filtering/data_tagger.py` performs tagging of data
- `scripts/data/filtering/tag_audio_lang.py` tags the audio with a language and produces a file containing the mapping between audio-transcript pair (`id` is the key here) and identified spoken language
- `scripts/data/filtering/assign_audio_lang_data.py` takes in the mapping file and applies it to the JSONL data files so that it contains the audio language tag
- `scripts/data/filtering/process_tagged_data.py` executes filtering and random sampling of data
- `scripts/data/filtering/reservoir_sample.py` provides information on the percentiles of data for a certain numerical-based tag (e.g WER score)
- calculating data statistics and some training parameters (per epoch steps, total training steps, total estimated duration)
- `scripts/data/filtering/gen_video_samples.py` generation of videos (containing audio and subtitles) from data samples for data assessment.

Refer to `configs/job_configs/data/filtering` for examples on how to perform tagging and filtering.

### Training
`scripts/training/*` has scripts for training OLMoASR models after the data has been prepared
- `scripts/training/train_fsdp_timestamps.py` uses FSDP to train with OLMoASR data
- `scripts/training/train_timestamps.py` uses DDP to train with OLMoASR data
- `scripts/train_owsm.py` uses DDP to train with OWSM-Eng data
- `scripts/train_yodas.py` uses DDP to train with YODAS-Eng data

Refer to `configs/job_configs/training` for example training scripts.

### Evaluation
`scripts/eval/*` consists of scripts to facilitate evaluation
- `scripts/eval/get_eval_set.py` download evaluation sets
- `scripts/eval/gen_inf_ckpt.py` modify model weights for inference[^*]
- `scripts/eval/eval.py` run evaluation

Refer to `configs/job_configs/eval` on how to evaluate models with the evaluation suite.

[^*]: This script is available due to modification to Whisper's model architecture code for padding mask to train with batches > 1. This is unnecessary for training (in hindsight), but was used in development of OLMoASR.