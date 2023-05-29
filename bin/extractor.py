#!/usr/bin/env python
"""Extract acoustic embeddings.

To extract acoustic features from a pre-trained model (e.g., Whisper) for the
audio file ``audio/test.wav``, and save the features in the directory
``feats/whisper``, run:

    python extractor.py \
        --model openai/whisper-base feats/whisper audio/test.wav

where the ``--model`` flag specifies the name of the pre-trained model in the
Hugging Face model hub.

You can split a long-form audio file into chunks by using ``chunk-size`` flag.
For example,

    python extractor.py --model openai/whisper-base --chunk-size 10 \
        feats/whisper audio/test.wav

will split the audio file ``audio/test.wav`` into chunks of 10 seconds.

By default, the script uses the CPU for feature extraction. However, you can
utilize the GPU by adding the ``--use-gpu`` flag. To disable the progress bar,
include the ``--disable-progress`` flag. You can also specify a directory
where the pre-trained model should be loaded from or downloaded to (if not
present) by using the ``--cache-dir`` flag. Additionally, you can adjust the
number of parallel jobs by using the ``--n-jobs`` flag. Here's an example that
combines all these options:

    python extractor.py --model openai/whisper-base --chunk-size 10 \
        --use-gpu --disable-progress --cache-dir checkpoints/ --n_jobs 1 \
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


MODEL_MAPPING = {"unispeech": (UniSpeechModel, Wav2Vec2FeatureExtractor),
                 "wav2vec2": (Wav2Vec2Model, Wav2Vec2FeatureExtractor),
                 "whisper": (WhisperModel, WhisperFeatureExtractor)}

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


def get_model(pretrained_model, device, cache_dir):
    get_encoder = False
    for key, val in MODEL_MAPPING.items():
        if key in pretrained_model:
            if key == "whisper":
                get_encoder = True
            AcousticModel, FeatExtractor = val
            break
    if AcousticModel and FeatExtractor:
        try:
            if get_encoder:
                model = AcousticModel.from_pretrained(
                    pretrained_model, cache_dir=cache_dir).encoder
            else:
                model = AcousticModel.from_pretrained(
                    pretrained_model, cache_dir=cache_dir)
            feature_extractor = FeatExtractor.from_pretrained(
                pretrained_model, cache_dir=cache_dir)
        except Exception:
            raise
    else:
        sys.exit("Specified model not supported.")

    model = model.to(device)
    return model, feature_extractor


def extract(
        af, minibatch_size, pretrained_model, device, cache_dir, feats_dir):
    y, sr = librosa.load(af, sr=16000)
    model, feature_extractor = get_model(pretrained_model, device, cache_dir)
    minibatches = get_minibatches_idx(len(y), minibatch_size)
    extracted_feats = []

    for num, minibatch in enumerate(minibatches):
        if num % 10 == 0:
            logger.info(f'Minibatch # {num}')
        z = y[minibatch]
        extractor = feature_extractor(
            z, return_tensors="pt", sampling_rate=sr)
        try:
            inputs = extractor.input_values
        except Exception:
            inputs = extractor.input_features
        inputs = inputs.to(device)

        with torch.no_grad():
            output = model(inputs)

        feats = output.last_hidden_state
        feats = feats.cpu().detach().numpy()
        feats = np.squeeze(feats)
        feats = feats.reshape(-1, feats.shape[-1])
        extracted_feats.append(feats)
    extracted_feats = np.vstack(extracted_feats)

    if "whisper" in pretrained_model:
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
        '--model', type=str, metavar='STR', default='openai/whisper-base',
        help='A model name in the Hugging Face model hub (default: Whisper '
             'base)')
    parser.add_argument(
        '--chunk-size', type=int, default=130,
        help='Length of chunks that a long-form audio file will be sliced '
             'into (default: %(default)s seconds)')
    parser.add_argument(
        '--use-gpu', action="store_true", default=False,
        help="Use GPU for feature extraction.")
    parser.add_argument(
        '--disable-progress', default=False, action='store_true',
        help='disable progress bar')
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='the directory where the pre-trained model should be loaded '
             'from or downloaded to')
    parser.add_argument(
        '--n-jobs', metavar='N', type=int, nargs=None, default=1,
        help='perform processing using N parallel jobs (default: %(default)s)'
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    os.makedirs(args.feats_dir, exist_ok=True)

    # Set device and model.
    device = torch.device(
        "cuda:5" if args.use_gpu and torch.cuda.is_available() else "cpu")
    # model, feature_extractor = get_model(args.model, device)

    # Process.
    audio_paths = sorted(args.afs)
    pool = multiprocessing.get_context('spawn').Pool(args.n_jobs)
    if "unispeech" in args.model or "wav2vec2" in args.model:
        chunk_size = args.chunk_size
        """
        It's weird that when using wav2vec2 feature extractor, one need to
        subtract 80 extra samples before calculating timepoints (stride=20ms).
        """
        minibatch_size = 16000 * chunk_size + 80
        f = partial(
            extract, minibatch_size=minibatch_size,
            pretrained_model=args.model, device=device,
            cache_dir=args.cache_dir, feats_dir=args.feats_dir)
    elif "whisper" in args.model:
        """
        From Whisper authors:
        The encoder always takes 30-second-long audio as input, and we trim or
        pad the audio to match this length. The encoded features are also
        30-seconds long as a result. You can slice the features if the input
        was shorter than 30 seconds.
        """
        chunk_size = 30
        minibatch_size = 16000 * chunk_size
        f = partial(
            extract, minibatch_size=minibatch_size,
            pretrained_model=args.model, device=device,
            cache_dir=args.cache_dir, feats_dir=args.feats_dir)
    else:
        sys.exit("Specified model not supported.")

    with tqdm(total=len(audio_paths), disable=args.disable_progress) as pbar:
        for _ in pool.imap(f, audio_paths):
            pbar.update(1)


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    logger.info(f'Total running time: {str(duration).split(".")[0]}')
