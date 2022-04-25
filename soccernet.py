import csv
from math import ceil
from operator import mul
from pathlib import Path
from typing import List
from util import get_frame, features_to_dict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


class SoccerNet(data.Dataset):

    def __init__(self, data_dir: Path, split_matches: List[Path], window_size_sec=20, frame_rate=8, **kwargs):
        self.path = data_dir

        videos_csv_path = self.path.joinpath(f'videos_v2.csv')
        self.videos = SoccerNet.read_videos(videos_csv_path)

        classes_csv_path = self.path.joinpath(f'classes_v2.csv')
        self.classes = SoccerNet.read_classes(classes_csv_path)
        self.num_classes = len(self.classes)

        self.annotations = SoccerNet.read_annotations(self.path.joinpath(f'annotations_v2.csv'))
        self.half_matches = [(mp, h) for mp, h in self.annotations.index.unique()]

        self.window_size_sec = window_size_sec
        self.frame_rate = frame_rate
        self.frames_per_window = self.window_size_sec * self.frame_rate

        self.split_matches = split_matches
        self.labels = []

        # J

        # obtaining our actions list filtered by our match list (pandas DataFrame)
        self.split_matches_actions = pd.DataFrame()
        for match in split_matches:
            match_actions_1 = self.annotations.loc[(match, 0)]
            match_actions_2 = self.annotations.loc[(match, 1)]
            match_actions = pd.concat([match_actions_1, match_actions_2])
            self.split_matches_actions = pd.concat([self.split_matches_actions, match_actions]).reset_index()

        # obtaining valid frames giving our action list and match list (valid_frames[match][half] --> list)
        self.valid_frames_dict = {}
        for match in split_matches:
            match_actions_1 = self.annotations.loc[(match, 0)].reset_index(drop=True)
            match_actions_2 = self.annotations.loc[(match, 1)].reset_index(drop=True)
            match_actions = [match_actions_1, match_actions_2]
            self.valid_frames_dict[match] = {}
            for half in range(0, 2):
                valid_frames = set()
                for i in range(0, len(match_actions[half])):
                    position = match_actions[half]['position'][i]
                    frame_num = get_frame(position, frame_rate)
                    first_action_frame = frame_num - (10 * frame_rate)
                    last_action_frame = frame_num + (10 * frame_rate)
                    for frame_id in range(first_action_frame, last_action_frame + 1):
                        valid_frames.add(frame_id)

                self.valid_frames_dict[match][half] = [f'{index:05}' for index in sorted(valid_frames)]

        print(type(self.valid_frames_dict[
                       Path('europe_uefa-champions-league/2016-2017/2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG')][0]))

        # obtaining emotions list (emotions[match][half][frame] --> pooled emotion frame)
        # TODO: Check if number of faces coincide with "analyze_faces"
        self.emotions = {}
        directory_location = Path('../../../mnt/DATA/datasets/soccernet')
        for match_path in split_matches:
            emotion_features_path = directory_location.joinpath(match_path.joinpath('face_extraction_results.npy'))
            match_emotion_features = np.load(emotion_features_path, allow_pickle=True)
            match_emotion_features = features_to_dict(match_emotion_features)
            print(match_emotion_features[0]['00437'])
            self.emotions[match_path] = {}
            for half in range(0, 2):
                self.emotions[match_path][half] = {}
                for frame in self.valid_frames_dict[match_path][half]:
                    if frame in match_emotion_features[half].keys():
                        print(frame)
                        # frame_emotion_vectors = match_emotion_features[1] where (half == half) and (frame == frame)
                        frame_emotion_vectors = match_emotion_features[half][frame].keys()
                        print(frame_emotion_vectors)
                        # self.emotions[match_path][half][frame] = pool(frame_emotion_vectors)
                        break
                    else:
                        # self.emotions[match_path][half][frame] = initialized vector
                        a = 'adeu'



                    # self.emotions[match_path][half][frame] = pool(frame_emotion_vectors)



    @staticmethod
    def read_classes(classes_csv_path):
        with open(classes_csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            return {i: r[0] for i, r in enumerate(csv_reader)}

    @staticmethod
    def read_videos(videos_csv_path):
        return pd.read_csv(videos_csv_path,
                           usecols=['match_path',
                                    'match_date',
                                    'visiting_team',
                                    'home_team',
                                    'score',
                                    'first_half_duration_sec',
                                    'second_half_duration_sec'],
                           dtype={'match_date': str,
                                  'visiting_team': str,
                                  'home_team': str,
                                  'score': str},
                           converters={'match_path': Path,
                                       'first_half_duration_sec': lambda d: int(float(d)),
                                       'second_half_duration_sec': lambda d: int(float(d))
                                       })

    @staticmethod
    def read_annotations(annotations_csv_path):
        to_secs = lambda t: sum(map(mul, [60, 1], map(int, t.split(':'))))
        return pd.read_csv(annotations_csv_path,
                           usecols=['match_path',
                                    'half',
                                    'game_time',
                                    'label',
                                    'position',
                                    'team',
                                    'visibility'],
                           dtype={'label': int,
                                  'position': int,
                                  'team': int
                                  },
                           converters={'match_path': Path,
                                       'half': lambda h: int(h) - 1,
                                       'game_time': to_secs,
                                       'visibility': lambda v: 1 if int(v) else -1},
                           index_col=['match_path',
                                      'half'])

    '''
    def _load_labels(self, match_path, half, num_batches):
        labels = np.zeros((num_batches, self.num_classes))
        for r in self.annotations.loc[(match_path, half)].itertuples():
            index = r.game_time // self.window_size_sec
            if index < num_batches:
                labels[index, r.label] = 1
        self.labels.append(labels)
    '''

    def __getitem__(self, idx):
        # relate frame and action here

        match_path = self.split_matches_actions.loc[idx, 'match_path']
        half = self.split_matches_actions.loc[idx, 'half']
        position = self.split_matches_actions.loc[idx, 'position']
        label = self.split_matches_actions.loc[idx, 'label']

        frame_indices = np.arange(get_frame(position, self.frame_rate) - 80, get_frame(position, self.frame_rate) + 81)
        frame_indices = [f'{index:05}' for index in frame_indices]
        # emotions_input = [self.emotions[match_path][half][index] for index in frame_indices]
        # emotions_input = np.array(emotions_input)
        # return emotions_input, label

    def __len__(self):
        pass


if __name__ == '__main__':
    data_path = Path('../../../mnt/DATA/datasets/soccernet')
    videos_path = Path('videos_to_process.txt')

    with videos_path.open() as f:
        videos = [Path(line).parent for line in f.readlines()][:1]

    dataset = SoccerNet(data_path, videos)
    # dataset.__getitem__(0)
    # print(dataset.classes)
    # print(dataset.emotion_features)
    # dataset getitem returns: (list of 160 vectors per action, action name)
