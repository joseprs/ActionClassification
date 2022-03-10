import logging
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from util import load_log_configuration, get_frame, first_last_frame, postprocess_function
import os
import json
# import pandas as pd
# from deepface.detectors import FaceDetector
# from skimage import io
from retinaface.commons import preprocess
from retinaface.model import retinaface_model
import tensorflow as tf


def load_image(arguments):
    img = cv2.imread(arguments)
    return preprocess.preprocess_image(img, True)


class FaceFeeder:

    def __init__(self, directory, valid_frames, batch_size):

        self.directory = Path(directory)
        self.valid_frames = valid_frames
        self.batch_size = batch_size

        batch_list = []
        for i in range(0, len(self.valid_frames), self.batch_size):
            batch_list.append(self.valid_frames[i:i + self.batch_size])
        self.batch_list = batch_list

    def __len__(self):
        return int(np.ceil(len(self.valid_frames) / self.batch_size))

    def __getitem__(self, index):

        batch = np.zeros([self.batch_size, 1024, 1820, 3])
        im_infos = []
        im_scales = []

        for idx, num in enumerate(self.batch_list[index]):
            frame_path = str(self.directory.joinpath(str(f'{num:05}') + '.jpg'))
            im_tensor, im_info, im_scale = load_image(frame_path)

            # im = cv2.imread(str(self.directory.joinpath(str(f'{num:05}')+'.jpg')))
            # im_tensor, im_info, im_scale = preprocess.preprocess_image(im, True)

            batch[idx, :, :, :] = im_tensor
            im_infos.append(im_info)
            im_scales.append(im_scale)

        return batch, im_infos, im_scales


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
        face_detection_results_fpath_json = match_path.joinpath(f'face_detection_results_{half + 1}_HQ.json')

        if face_detection_results_fpath.exists():
            continue

        frames_dir = match_path.joinpath(f'{half + 1}_HQ', 'frames')
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'Extracting frames into {frames_dir}')
            subprocess.check_call(f'./scripts/extract_frames.sh "{str(video).strip()}" "{frames_dir}"', shell=True)
            logging.info('Frames extracted successfully!')

        num_frames = len(list(frames_dir.glob('*.jpg')))
        logging.info(f'Number of frames to segment: {num_frames}')

        # where we put starting TIME?
        start = time.time()

        # read annotations/Labels file and separate it by half
        json_path = match_path.joinpath('Labels-v2.json')
        with open(json_path) as f:
            data = json.load(f)
        annotations = data['annotations']
        for i in range(0, len(annotations)):
            if annotations[i]['gameTime'][0] == '2':
                half_index = i
                break
        an1 = annotations[:half_index]
        an2 = annotations[half_index:]
        actions = [an1, an2]

        # selecting valid frames (frames we are gonna use)
        valid_frames = set()
        for action in actions[half]:
            position = int(action['position'])
            frame_num = get_frame(position, args.sampling_freq)
            first_action_frame, last_action_frame = first_last_frame(frame_num, args.window, args.sampling_freq,
                                                                     num_frames)
            for frame_id in range(first_action_frame, last_action_frame + 1):
                valid_frames.add(frame_id)

        invalid_frames = list(set(np.arange(num_frames)) - valid_frames)
        valid_frames = sorted(valid_frames)
        # DELETE invalid frames
        # os.remove(invalid_frames)
        # frames_dir.joinpath(str(f'{invalid_frames[0]:05}')+'.jpg')

        f = FaceFeeder(frames_dir, valid_frames, 37)
        n_batches = f.__len__()
        logging.info(f'Number of batches to process: {n_batches}')

        # RetinaFace model
        model = retinaface_model.build_model()

        for batch_num in tqdm(range(n_batches), desc='Batches Progress', leave=True, position=0):

            batch, im_infos, im_scales = f.__getitem__(batch_num)
            frame_ids = f.batch_list[batch_num]

            # with tf.device('/device:GPU:0'):
            #     tensor_batch = tf.constant(batch)
            # outputs = model(tensor_batch)
            outputs = model(batch)


            results = []
            outputs2 = [elt.numpy() for elt in outputs]

            for i, frame_id in enumerate(frame_ids):
                output = [np.expand_dims(outputs2[j][i, ...], axis=0) for j in range(9)]
                results.append(postprocess_function(output, im_infos[i], im_scales[i]))

            # print(results)
