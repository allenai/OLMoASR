## OLMoASR Pipeline

In this directory, you'll find the scripts to transform your raw data, train your model and evaluate it. 

### Data Processing
You can refer to the `scripts/data` and `olmoasr` directory, where `olmoasr/preprocess.py` contains segmentation code, and `scripts/data/processing/*` houses scripts to organize a directory of transcript files into a JSONL file.

### Filtering
`scripts/data/filtering/*` contains scripts to 
- perform tagging and filtering of data
- random sampling of data
- calculating specific training hyperparameters (per epoch steps, total training steps, warmup steps)
- generation of videos (containing audio and subtitles) from data samples for data assessment.

### Training
`scripts/training/*` has scripts for training OLMoASR models after the data has been prepared. 
- `scripts/training/train_fsdp_timestamps.py` uses FSDP to train with OLMoASR data
- `scripts/training/train_timestamps.py` uses DDP to train with OLMoASR data
- `scripts/train_owsm.py` uses DDP to train with OWSM-Eng data
- `scripts/train_yodas.py` uses DDP to train with YODAS-Eng data

### Evaluation
`scripts/eval/*` consists of scripts to 
- `scripts/eval/get_eval_set.py` download evaluation sets
- `scripts/eval/gen_inf_ckpt.py` modify model weights for inference[^*]
- `scripts/eval/eval.py` run evaluation



[^*]: This script is available due to modification to Whisper's model architecture code for padding mask to train with batches > 1. This is unnecessary for training, but was used in development of OLMoASR.