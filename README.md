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

```
git clone https://github.com/allenai/OLMoASR.git
pip install -r requirements/requirements.txt
pip install -e .
```

### Data Processing
Once you've downloaded and organized your data, you'll need to follow the following steps to process your data:
1. Transform all your transcripts into JSONL format to be suitable for tagging and filtering using `scripts/data/processing/text_to_jsonl.py`
2. Segment all your full-length audio files into 30s-long audio chunks
3. Perform document-level tagging
4. Segment transcript files into 30s-long transcript chunks
5. Perform segment-level tagging
6. Filter based on a specified configuration of conditions

### Training

### Evaluation
- TODO
    - Bug: SDPA doesn't work for inference, but only training
    - Bug: Feeding in initial prompt causes model to not work properly for long-form transcription


### Inference

## Available Models

## Usage

## Team

## License

## Citing