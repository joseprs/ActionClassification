import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from util import load_log_configuration, get_detected_facial_areas, crop_face

from deepface.DeepFace import build_model

import tensorflow as tf
from tensorflow import keras


class FacePatch:
    def __init__(self, path, face_patch):
        self.path = path
        self.masked_patch = face_patch


def _load_face_patches(args):
    half, frame_idx, frame_path, segmented_faces = args
    frame = cv2.imread(str(frame_path))

    face_patches = []
    for idx, bb in enumerate(segmented_faces):
        face_patch, area = crop_face(frame, bb)
        face_patches.append(FacePatch((half, frame_idx, idx, area), face_patch))
    return face_patches


def init_results():
    results = {}
    for half1 in range(2):

        face_detections_fpath = match_path.joinpath(f'face_detection_results_{half1 + 1}_HQ.npy')
        face_detections = np.load(face_detections_fpath, allow_pickle=True)[0]
        faces_frames = [k for k, v in face_detections.items() if len(v) > 0]

        results[half1] = {}
        for frame in faces_frames:
            # segmented_faces_in_frame = get_segmented_face_areas(face_detections[frame])
            results[half1][frame] = [None] * len(face_detections[frame])
    return results


class FaceEmotionFeeder(keras.utils.Sequence):

    def __init__(self, match_path, batch_size, transform=None, num_processes=None):
        self.match_path = match_path
        self.transform = transform
        self.batch_size = batch_size
        self.data = []

        for half in range(2):
            face_detections_fpath = match_path.joinpath(f'face_detection_results_{half + 1}_HQ.npy')
            face_detections = np.load(face_detections_fpath, allow_pickle=True)[0]
            faces_frames = [k for k, v in face_detections.items() if len(v) > 0]

            frames_path = match_path.joinpath(f'{half + 1}_HQ', 'frames')
            halves, frame_indices, frame_paths, detected_faces = [], [], [], []

            for frame in tqdm(faces_frames, desc=f'Half {half + 1}: Importing frames progress', leave=True, position=0):
                detected_faces_in_frame = get_detected_facial_areas(face_detections[frame])

                if len(detected_faces_in_frame) > 0:
                    halves.append(half)
                    frame_indices.append(frame)
                    frame_paths.append(frames_path.joinpath(frame + '.jpg'))
                    detected_faces.append(detected_faces_in_frame)

            with Pool(processes=num_processes) as pool:
                for face_patches in pool.imap_unordered(_load_face_patches,
                                                        zip(halves, frame_indices, frame_paths, detected_faces)):
                    self.data.extend(face_patches)

        faces_batch_list = []
        for i in range(0, len(self.data), self.batch_size):
            faces_batch_list.append(self.data[i:i + self.batch_size])
        self.faces_batch_list = faces_batch_list

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        info_list = []
        face_list = np.zeros([len(self.faces_batch_list[idx]), 48, 48, 1])
        for idx, face in enumerate(self.faces_batch_list[idx]):
            info_list.append(face.path)
            face_list[idx, :, :, :] = face.masked_patch
        return info_list, face_list


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
                       help='Batch Size',
                       default=50, type=int)

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
    model = build_model('Emotion')
    extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

    for video in tqdm(videos, desc='Overall Extraction Progress', leave=True, position=0):
        start = time.time()
        match_path = video  # .parent
        logging.info(f'Match path: {match_path}')

        # .data returns a list (N faces) of FacePatch (class)
        ef = FaceEmotionFeeder(match_path, args.batch_size, None, 10)  # .data
        n_batches = len(ef)

        logging.info(f'Number of faces to process: {len(ef.data)}')
        logging.info(f'Number of batches to process: {n_batches}')

        # results_dict = init_results()
        results = []

        for batch_num in tqdm(range(n_batches), desc='Batches Progress', leave=True, position=0):
            infos_batch, faces_batch = ef.__getitem__(batch_num)
            outputs = extractor(faces_batch)
            vectors = outputs[8].numpy()

            for (half, frame_idx, idx, area), prediction in zip(infos_batch, vectors):
                results.append(((half, frame_idx, idx, area), prediction))
                # results_dict[half][frame_idx][idx] = (prediction, area)

        logging.info(f'Video processing time is {time.time() - start} seconds')

        face_extraction_results_fpath = match_path.joinpath(f'face_extraction_results.npy')
        np.save(face_extraction_results_fpath, results)

