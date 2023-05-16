#!/usr/bin/env python
"""Extract acoustic embeddings.

To extract acoustic features from a pre-trained model (e.g., Whisper) for the
audio file ``audio/test.wav``, and save the features in the directory
``feats/whisper``, run:

    python extractor.py -f whisper feats/whisper audio/test.wav

where the ``-f`` flag specifies the name of the pre-trained model.

The -f flag specifies the name of the pre-trained model.

By default, the script uses the CPU for feature extraction. However, you can
utilize the GPU by adding the ``--use-gpu`` flag. To disable the progress bar,
include the ``--disable-progress`` flag. Additionally, you can adjust the
number of parallel jobs by using the ``--n-jobs`` flag. Here's an example that
combines these options:

    python extractor.py -f whisper --use-gpu --disable-progress --n_jobs -1 \
        feats/whisper audio/test.wav

"""
import argparse
import librosa
import logging
import math
import multiprocessing
import numpy as np
import os
import sys
import time
import torch
from datetime import timedelta
from functools import partial
from pathlib import Path
from transformers import (
    UniSpeechModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    WhisperFeatureExtractor, WhisperModel)
from tqdm import tqdm


# Set up a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def get_minibatches_idx(n, minibatch_size):
    idx_list = np.arange(n, dtype="int32")
    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(
            idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return minibatches


def get_model(pretrained_model, device):
    if pretrained_model.startswith("u"):
        model_name = "microsoft/unispeech-large-1500h-cv"
        model = UniSpeechModel.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name)
    elif pretrained_model.startswith("wa"):
        model_name = "facebook/wav2vec2-large-lv60"
        model = Wav2Vec2Model.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name)
    elif pretrained_model.startswith("wh"):
        model_name = "openai/whisper-base"
        model = WhisperModel.from_pretrained(model_name).encoder
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name)

    model = model.to(device)
    return model, feature_extractor


class wav2vec2:
    def extract(af, feats_dir, pretrained_model, device):
        y, sr = librosa.load(af, sr=16000)
        model, feature_extractor = get_model(pretrained_model, device)
        """
        It's weird that when using wav2vec2 feature extractor, one need to
        subtract 80 extra samples before calculating timepoints (stride=20ms).
        """
        minibatch_size = 16000 * 130 + 80
        minibatches = get_minibatches_idx(len(y), minibatch_size)
        extracted_feats = np.empty((0, 1024))
        for num, minibatch in enumerate(minibatches):
            if num % 10 == 0:
                logger.info(f'Minibatch # {num}')
            z = y[minibatch]
            extractor = feature_extractor(
                z, return_tensors="pt", sampling_rate=sr)
            inputs = extractor.input_values
            inputs = inputs.to(device)

            with torch.no_grad():
                output = model(inputs)

            feats = output.last_hidden_state
            feats = feats.cpu().detach().numpy()
            feats = np.squeeze(feats)
            feats = feats.reshape(-1, feats.shape[-1])
            extracted_feats = np.vstack((extracted_feats, feats))

        dur = librosa.get_duration(y=y, sr=sr)
        time_points = np.array([round(i, 2) for i in np.arange(
            0, dur, dur/extracted_feats.shape[0])])
        feats_path = Path(feats_dir, af.stem + '.npy')
        np.save(feats_path, extracted_feats)
        time_path = Path(feats_dir, af.stem + '_timepoints.npy')
        np.save(time_path, time_points)


class whisper:
    def extract(af, feats_dir, pretrained_model, device):
        y, sr = librosa.load(af, sr=16000)
        model, feature_extractor = get_model(pretrained_model, device)
        """
        From Whisper authors:
        The encoder always takes 30-second-long audio as input, and we trim or
        pad the audio to match this length. The encoded features are also
        30-seconds long as a result. You can slice the features if the input
        was shorter than 30 seconds.
        """
        minibatch_size = 16000 * 30
        minibatches = get_minibatches_idx(len(y), minibatch_size)
        extracted_feats = np.empty((0, 512))
        for num, minibatch in enumerate(minibatches):
            if num % 10 == 0:
                logger.info(f'Minibatch # {num}')
            z = y[minibatch]
            extractor = feature_extractor(
                z, return_tensors="pt", sampling_rate=sr)
            inputs = extractor.input_features
            inputs = inputs.to(device)

            with torch.no_grad():
                output = model(inputs)

            feats = output.last_hidden_state
            feats = feats.cpu().detach().numpy()
            feats = np.squeeze(feats)
            feats = feats.reshape(-1, feats.shape[-1])
            extracted_feats = np.vstack((extracted_feats, feats))

        frame_to_slice = int(30 / 0.02 - math.floor(
            len(minibatches[-1]) / minibatch_size * (30 / 0.02)))
        extracted_feats = extracted_feats[:-frame_to_slice]

        dur = librosa.get_duration(y=y, sr=sr)
        time_points = np.array([round(i, 2) for i in np.arange(
            0, dur, dur/extracted_feats.shape[0])])
        feats_path = Path(feats_dir, af.stem + '.npy')
        np.save(feats_path, extracted_feats)
        time_path = Path(feats_dir, af.stem + '_timepoints.npy')
        np.save(time_path, time_points)


def main():
    parser = argparse.ArgumentParser(
        description='Extract acoustic embeddings.', add_help=True)
    parser.add_argument(
        'feats_dir', type=Path,
        help='Path to output directory for .npy files.')
    parser.add_argument(
        'afs', nargs='+', type=Path, help='Audio files to be processed.')
    parser.add_argument(
        '-f', '--feat', dest='feats', type=str, metavar='STR',
        help='The model used to extract features.')
    parser.add_argument(
        '--use-gpu', action="store_true", default=False,
        help="Use GPU for feature extraction.")
    parser.add_argument(
        '--disable-progress', default=False, action='store_true',
        help='disable progress bar')
    parser.add_argument(
        '--n-jobs', metavar='N', type=int, nargs=None, default=1,
        help='perform processing using N parallel jobs (default: %(default)s)'
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    os.makedirs(args.feats_dir, exist_ok=True)

    # Set device.
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Process.
    audio_paths = sorted(args.afs)
    pool = multiprocessing.get_context('spawn').Pool(args.n_jobs)
    if args.feats.startswith("u") or args.feats.startswith("wa"):
        f = partial(
            wav2vec2.extract, feats_dir=args.feats_dir,
            pretrained_model=args.feats, device=device)
    elif args.feats.startswith("wh"):
        f = partial(
            whisper.extract, feats_dir=args.feats_dir,
            pretrained_model=args.feats, device=device)
    with tqdm(total=len(audio_paths), disable=args.disable_progress) as pbar:
        for _ in pool.imap(f, audio_paths):
            pbar.update(1)


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    logger.info(f'Total running time: {str(duration).split(".")[0]}')
