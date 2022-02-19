import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from util import load_log_configuration
import os
import json
import pandas as pd
from deepface.detectors import FaceDetector
from skimage import io


def get_frame(position, fs):
    return (position // 1000) * fs


def first_last_frame(frame_num, seconds_window, fs):
    if frame_num - (seconds_window * fs) > 0:
        f = frame_num - (seconds_window * fs)
    else:
        f = 1
    if frame_num + (seconds_window * fs) > len(os.listdir(frames_dir)):
        l = len(os.listdir(frames_dir))
    else:
        l = frame_num + (seconds_window * fs)

    return f, l


def detect_faces(frames_dir, action_list, fs, seconds_window):
    faces = pd.DataFrame(
        columns=['face_locations', 'current_frame', 'action_frame', 'action_position', 'action_name', 'action_time',
                 'half'])
    detector_name = 'retinaface'
    detector = FaceDetector.build_model(detector_name)

    # going action by action
    for action in tqdm(action_list[1:], desc='Actions Faces Detection', leave=True, position=0):

        position = int(action['position'])
        frame_num = get_frame(position, fs)
        first_action_frame, last_action_frame = first_last_frame(frame_num, seconds_window, fs)

        # going frame by frame inside the action
        for i in range(first_action_frame, last_action_frame):
            frame_path = frames_dir.joinpath(f'{i:05}' + '.jpg')
            image = io.imread(frame_path)
            face_loc = FaceDetector.detect_faces(detector, detector_name, image)
            array = np.array(face_loc)
            face_locations = []
            for arr in array:
                face_locations.append(arr[1])
            print(i)
            print(face_locations)

    print(first_action_frame)
    return action_list


if __name__ == '__main__':

    parser = ArgumentParser(description='Face detection')

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

    for video in tqdm(videos, desc='Overall Progress', leave=True, position=0):

        half = int(video.stem[0]) - 1
        match_path = video.parent
        face_detection_results_fpath = match_path.joinpath(f'face_detection_results_{half + 1}_HQ.npy')

        if face_detection_results_fpath.exists():
            continue

        frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'Extracting frames into {frames_dir}')
            subprocess.check_call(f'./scripts/extract_frames.sh "{str(video).strip()}" "{frames_dir}"', shell=True)

        num_frames = len(list(frames_dir.glob('*.jpg')))
        logging.info(f'Number of frames to segment {num_frames}')

        num_frames = len(list(frames_dir.glob('*.jpg')))
        logging.info(f'Number of frames to segment {num_frames}')  # NO SE VISUALIZA ESTO

        start = time.time()

        json_path = match_path.joinpath('Labels-v2.json')
        with open(json_path) as f:
            data = json.load(f)
        annotations = data['annotations']
        for i in range(0, len(annotations)):
            if annotations[i]['gameTime'][0] == '2':
                index = i
                break
        an1 = annotations[:index]
        an2 = annotations[index:]
        actions = [an1, an2]

        detected_faces = detect_faces(frames_dir, match_path, actions[half], args.sampling_freq, args.window)
        # np.save(face_detection_results_fpath, detected_faces)
        logging.info(f'Video processing time is {time.time() - start} seconds')