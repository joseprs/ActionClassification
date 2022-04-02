import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from util import load_log_configuration, get_segmented_face_areas, get_masked_face
import json

from deepface.DeepFace import build_model
from retinaface.commons import preprocess
from retinaface.model import retinaface_model
import tensorflow as tf


def _load_masked_patches(args):
    half, frame_idx, frame_path, segmented_people = args
    frame = cv2.imread(str(frame_path))

    masked_faces = []
    for idx, bb in enumerate(segmented_people):
        masked_face = get_masked_face(frame, bb)
        masked_faces.append(((half, frame_idx, idx), masked_face))
    return masked_faces


class FaceEmotionFeeder:

    def __init__(self, match_path, transform=None, num_processes=None):
        self.match_path = match_path
        self.transform = transform
        self.data = []

        for half in range(1):
            face_detections_fpath = match_path.joinpath(f'face_detection_results_{half + 1}_HQ1.npy')
            face_detections = np.load(face_detections_fpath, allow_pickle=True)[0]
            faces_frames = [k for k, v in face_detections.items() if len(v) > 0]

            frames_path = match_path.joinpath(f'{half + 1}_HQ', 'frames')
            halves, frame_indices, frame_paths, segmented_faces = [], [], [], []

            for frame in faces_frames[:3]:
                segmented_faces_in_frame = get_segmented_face_areas(face_detections[frame])

                if len(segmented_faces_in_frame) > 0:
                    halves.append(half)
                    frame_indices.append(frame)
                    frame_paths.append(frames_path.joinpath(frame + '.jpg'))
                    segmented_faces.append(segmented_faces_in_frame)

            with Pool(processes=num_processes) as pool:
                for masked_patches in pool.imap_unordered(_load_masked_patches, zip(halves, frame_indices, frame_paths, segmented_faces)):
                    self.data.extend(masked_patches)


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

    load_log_configuration(args.log_config, args.logs_dir)

    if args.videos:
        with args.videos.open() as f:
            videos = [Path(line) for line in f.readlines()]
    else:
        videos = [args.single_video]

    # DEEPFACE model
    # model = build_model('Emotion')

    for video in tqdm(videos, desc='Overall Extraction Progress', leave=True, position=0):
        half = int(video.stem[0]) - 1
        match_path = video.parent

        ef = FaceEmotionFeeder(match_path).data
        print(len(ef[0]))
        print(ef)
