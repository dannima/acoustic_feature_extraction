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
MODELS="wav2vec2 whisper unispeech"

# Current stage.
stage=0


##################################
# Extract features
##################################
if [ $stage -le 1 ]; then
    for model in ${MODELS}; do
    	args="-f ${model} --frame-len 3921 --hop-len 3921 --n-jobs 1 --n-gpus 6"
		logf=${LOGS_DIR}/${model}_${DATE}.log
		echo "Extracting features using ${model}..."
		python ${BIN_DIR}/extractor.py \
		    $args ${FEATS_DIR}/${model} ${AUDIO_DIR}/E*wav > $logf 2>&1
    done
fi
