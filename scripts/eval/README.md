## OpenWhisper Evaluation Suite
### Datasets
To get the evaluation set you want to perform evaluation with, run
```
python get_eval_set.py --eval_set=[name of eval set] --eval_dir=[path to dir of eval sets] --hf_token=[huggingface user auth token]
```
Provide the name of the evaluation set you want to download and the path to the directory that will hold all your evaluation sets. The default evaluation directory is `data/eval` and will be created if you don't have it yet.

#### Supported Datasets
- `librispeech_clean, librispeech_other` LibriSpeech (Clean and Other) 
- `artie_bias_corpus` Artie Bias Corpus
- `fleurs` Fleurs
- `tedlium` TEDLIUM
- `voxpopuli` VoxPopuli
- `common_voice` CommonVoice
    - NOTE: The entire dataset needs to be downloaded (not just eval split) and is downloaded through HuggingFace. [Please generate a HuggingFace user authentication token](https://huggingface.co/docs/hub/en/security-tokens) and provide it as an argument when running `get_eval_set.py`
- `ami_ihm, ami_sdm` AMI (IHM and SDM)

### Evaluation
To run evaluation, run 
```
python eval.py --ckpt=[path to checkpoint] --eval_set=[name of eval set] --eval_dir=[path to dir of eval sets] --hf_token=[huggingface user auth token] 
```
Please provide the path of the model checkpoint you want to run evaluation with, the name of the evaluation set and the directory that holds the evaluation set. If you're running evaluation on **CommonVoice**, you'll also need to provide a [HuggingFace user authentication token](https://huggingface.co/docs/hub/en/security-tokens).