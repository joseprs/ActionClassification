import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from util import load_log_configuration, get_frame, first_last_frame, postprocess_function
import json

from deepface.DeepFace import build_model
from retinaface.commons import preprocess
from retinaface.model import retinaface_model
import tensorflow as tf

def load_image(arg):
    frame_idx, frame_path = arg
    img = cv2.imread(frame_path)
    return frame_idx, preprocess.preprocess_image(img, True)

class LoadFaces:

    def __init__(self, directory, faces_frames, num_processes=10):
        self.directory = directory
        self.faces_frames = faces_frames




if __name__ == '__main__':

    parser = ArgumentParser(description='Face vector extraction')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-s', '--single_video',
                       help='Filepath for the video to process',
                       default=None, type=lambda p: Path(p))

    group.add_argument('-v', '--videos',
                       help='Path for files containing a video list to process',
                       default=None, type=lambda p: Path(p))

    group.add_argument('-f', '--sampling_freq',
                       help='Sampling frequency',
                       default=8, type=int)

    group.add_argument('-w', '--window',
                       help='Window',
                       default=10, type=int)

    group.add_argument('-bs', '--batch_size',
                       help='Window',
                       default=37, type=int)

    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))

    parser.add_argument('--logs_dir', required=False,
                        help='Logging directory path (default: logs)',
                        default="logs", type=lambda p: Path(p))

    args = parser.parse_args()

    # load_log_configuration(args.log_config, args.logs_dir)

    if args.videos:
        with args.videos.open() as f:
            videos = [Path(line) for line in f.readlines()]
    else:
        videos = [args.single_video]

    # DEEPFACE model
    model = build_model('Emotion')

    for video in tqdm(videos, desc='Overall Extraction Progress', leave=True, position=0):

        half = int(video.stem[0]) - 1
        match_path = video.parent
        frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
        num_frames = len(list(frames_dir.glob('*.jpg')))

        face_detection_results_fpath = match_path.joinpath(f'face_detection_results_{half + 1}_HQ.npy')
        face_locations = np.load(face_detection_results_fpath, allow_pickle=True)

        print(type(face_locations))

