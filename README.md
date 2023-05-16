## Overview
This repository utilizes [Hugging Face Transformers](huggingface.co/transformers) to extract acoustic features from input audio. It currently supports feature extraction using the following pre-trained acoustic models:

* [wav2vec2.0](https://arxiv.org/pdf/2006.11477.pdf)
* [UniSpeech](https://arxiv.org/pdf/2101.07597.pdf)
* [Whisper](https://arxiv.org/pdf/2212.04356.pdf)

## Prerequisites
* **Bash** version &ge; 4
* **Python** version &ge; 3.6
* If you are _not_ using a GPU with CUDA 11.3, you may need to modify the `torch==1.12.1+cu113` entry in `requirements.txt` to match your CUDA version before installation.
* Install other required packages:

```
$ pip install -r requirements.txt
```

## Feature Extraction
`run.sh` provides an example of how to extract features from pre-trained models. Before running it, change the `PROJECT_DIR` variable to match the local path where your repository is stored.

Basically, the script takes audio files located in `audio/` as inputs, and produces the following outputs:

1. Extracted features in `feats/`;
2. Arrays of timepoints of each feature in `feats/`;
3. Log files in `logs/`. 

Feel free to modify arguments based on the instructions provided at the top of the ``bin/extractor.py`` file.