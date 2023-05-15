#!/usr/bin/env python
"""Extract acoustic embeddings."""
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
from torch import nn
from transformers import (
    UniSpeechModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    WhisperFeatureExtractor, WhisperModel)
from tqdm import tqdm


# Set up a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def get_minibatches_idx(n, minibatch_size):
    '''Randomly select indices to form mini batches.'''
    idx_list = np.arange(n, dtype="int32")
    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(
            idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return minibatches


def get_model(pretrained_model):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device)
    return model, feature_extractor, device


class wav2vec2:
    def extract(
            af, frame_length, hop_length, feats_dir, pretrained_model,
            minibatch_size, n_gpus):
        y, sr = librosa.load(af, sr=16000)
        model, feature_extractor, device = get_model(pretrained_model)
        if frame_length * minibatch_size < len(y):
            # Slice audio into frames if audio is long
            y = librosa.util.frame(
                y, frame_length=frame_length, hop_length=hop_length).T

            extractor = feature_extractor(
                y, return_tensors="pt", sampling_rate=sr)
            i = extractor.input_values
            i = torch.squeeze(extractor.input_values)
            minibatches = get_minibatches_idx(len(i), minibatch_size)

            # To avoid shape mismatch error due to using multiple GPUs
            if n_gpus > 1:
                if len(minibatches[-1]) < minibatch_size:
                    first = len(
                        minibatches[-1]) - len(minibatches[-1]) % n_gpus
                    minibatches[-1] = minibatches[-1][:first]

            extracted_feats = np.empty((0, 1024))
            for num, minibatch in enumerate(minibatches):
                logger.info(f'Minibatch # {num}')
                inputs = i[minibatch]
                inputs = inputs.to(device)
                with torch.no_grad():
                    output = model(inputs)

                feats = output.last_hidden_state
                feats = feats.cpu().detach().numpy()
                feats = np.squeeze(feats)
                feats = feats.reshape(-1, feats.shape[-1])
                extracted_feats = np.vstack((extracted_feats, feats))
        else:
            extractor = feature_extractor(
                y, return_tensors="pt", sampling_rate=sr)
            i = extractor.input_values
            i = i.to(device)
            with torch.no_grad():
                output = model(i)
            feats = output.last_hidden_state
            feats = feats.cpu().detach().numpy()
            extracted_feats = np.squeeze(feats)

        dur = librosa.get_duration(y=y, sr=sr)
        time_points = np.array([round(i, 2) for i in np.arange(
            0, dur, dur/extracted_feats.shape[0])])
        feats_path = Path(feats_dir, af.stem + '.npy')
        np.save(feats_path, extracted_feats)
        time_path = Path(feats_dir, af.stem + '_timepoints.npy')
        np.save(time_path, time_points)


class whisper:
    def extract(af, feats_dir, pretrained_model, n_gpus):
        y, sr = librosa.load(af, sr=16000)
        model, feature_extractor, device = get_model(pretrained_model)
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
        logger.info(f"extracted_feats shape: {extracted_feats.shape}")

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
        '--frame-len', type=int, default=3920, help='Length of the frame')
    parser.add_argument(
        '--hop-len', type=int, default=3920,
        help='Number of steps to advance between frames')
    # parser.add_argument(
    #     '--use-gpu', default=False, action='store_true', help='Use GPU.')
    parser.add_argument(
        '--disable-progress', default=False, action='store_true',
        help='disable progress bar')
    parser.add_argument(
        '--n-jobs', metavar='N', type=int, nargs=None, default=1,
        help='perform processing using N parallel jobs (default: %(default)s)'
        )
    parser.add_argument(
        '--n-gpus', dest='n_gpus', nargs=None, default=1, type=int,
        help='number of GPUs (default: %(default)s)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    os.makedirs(args.feats_dir, exist_ok=True)

    if args.n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in list(range(args.n_gpus)))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # minibatch_size = 5000 for 6 GPUs, minibatch_size = 800 for 1 GPU
    minibatch_size = 800

    # Process.
    audio_paths = sorted(args.afs)
    pool = multiprocessing.get_context('spawn').Pool(args.n_jobs)
    if args.feats.startswith("u") or args.feats.startswith("wa"):
        f = partial(
            wav2vec2.extract, frame_length=args.frame_len,
            hop_length=args.hop_len, feats_dir=args.feats_dir,
            pretrained_model=args.feats, minibatch_size=minibatch_size,
            n_gpus=args.n_gpus)
    elif args.feats.startswith("wh"):
        f = partial(
            whisper.extract, feats_dir=args.feats_dir,
            pretrained_model=args.feats, n_gpus=args.n_gpus)
    with tqdm(total=len(audio_paths), disable=args.disable_progress) as pbar:
        for _ in pool.imap(f, audio_paths):
            pbar.update(1)


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    logger.info(f'Total running time: {str(duration).split(".")[0]}')
