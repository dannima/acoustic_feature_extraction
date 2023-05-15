## Overview
This is a repository that uses [Hugging Face Transformers](huggingface.co/transformers) to extract acoustic features of an input audio. It now can extract features from the following pre-trained acoustic models:

* [wav2vec2.0](https://arxiv.org/pdf/2006.11477.pdf)
* [UniSpeech](https://arxiv.org/pdf/2101.07597.pdf)
* [Whisper](https://arxiv.org/pdf/2212.04356.pdf)

## Prerequisites
* **Bash** version &ge; 4
* **Python** version &ge; 3.6
* If you are _not_ using GPU with CUDA 11.3, you might want to change `torch==1.12.1+cu113` in `requirements.txt` to your own CUDA version before installation.
* Install other required packages:

```
$ pip install -r requirements.txt
```

## Feature Extraction
`run.sh` is an example of how to get features from pre-trained models. Before running it, change `PROJECT_DIR` to the local path where your repository is stored.

Basically, the script takes audios in `audio/` as inputs, outputs extracted features in `feats/`, and log files in `logs/`. Feel free to change arguments following the instructions on top of `bin/extractor.py`