#!/bin/bash
# Extract acoustic features from a pre-trained model.

set -e


###################################
# Configuration
###################################
# Project directory for feature extraction.
PROJECT_DIR='YOUR_PROJECT_DIRECTORY'

# Audio files (.wav).
AUDIO_DIR=${PROJECT_DIR}/audio

# Output directory for extracted features.
FEATS_DIR=${PROJECT_DIR}/feats

# Date run.
DATE=$(date +"%m-%d-%Y")

# Logs directory.
LOGS_DIR=${PROJECT_DIR}/logs

# Bin directory.
BIN_DIR=${PROJECT_DIR}/bin

# Pretrained acoustic models.
MODELS="facebook/wav2vec2-large-lv60 microsoft/unispeech-large-1500h-cv \
        openai/whisper-base"

# Current stage.
stage=0


##################################
# Extract features
##################################
if [ $stage -le 1 ]; then
    for model_name in ${MODELS}; do
    	args="--model ${model_name} --chunk-size 130 --use-gpu \
    	      --cache-dir checkpoints --n-jobs 1"
    	model=${model_name#*/}
		logf=${LOGS_DIR}/${model}_${DATE}.log
		echo "Extracting features using ${model}..."
		python ${BIN_DIR}/extractor.py \
		    $args ${FEATS_DIR}/${model} ${AUDIO_DIR}/*wav > $logf 2>&1
    done
fi
