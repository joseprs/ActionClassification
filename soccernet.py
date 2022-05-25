import csv
from math import ceil
from operator import mul
from pathlib import Path
from typing import List
from util import get_frame

import numpy as np
import pandas as pd
import torch.utils.data as data


def features_to_dict(dataframe):
    features_dict = {0: {}, 1: {}}
    for info, vector in dataframe:
        features_dict[info[0]][info[1]] = {}
    for info, vector in dataframe:
        features_dict[info[0]][info[1]][int(info[2])] = (info[3], info[4], vector)
    return features_dict


class SoccerNet(data.Dataset):

    def __init__(self, data_dir: Path, split_matches: List[Path], window_size_sec=20, frame_rate=8, pool='MAX',
                 balance=False, th=None):
        self.path = data_dir
        self.pool = np.max if pool == "MAX" else np.mean

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
        self.balance_threshold = th

        self.split_matches = split_matches

        edit_annotations = self.annotations.reset_index()
        # obtaining our actions list filtered by our match list (pandas DataFrame)
        self.split_matches_actions = pd.DataFrame()
        for match in self.split_matches:
            match_actions = pd.concat([edit_annotations.loc[(edit_annotations['match_path'] == match) &
                                                            (edit_annotations['half'] == i)] for i in
                                       range(2)]).drop_duplicates()
            self.split_matches_actions = pd.concat(
                [self.split_matches_actions, match_actions]).reset_index(drop=True)

        # obtaining a balanced action list given a threshold
        if balance:
            balanced_annotations = pd.DataFrame()
            for label in range(1, self.num_classes + 1):
                class_annotations = self.split_matches_actions.loc[self.split_matches_actions['label'] == label]
                if len(class_annotations) > self.balance_threshold:
                    class_annotations = class_annotations.sample(n=self.balance_threshold, random_state=0)
                balanced_annotations = pd.concat([balanced_annotations, class_annotations])
            self.split_matches_actions = balanced_annotations.reset_index(drop=True)

        # obtaining valid frames giving our action list and match list (valid_frames[match][half] --> list)
        self.valid_frames_dict = {}
        for match in self.split_matches:
            match_actions = [edit_annotations.loc[(edit_annotations['match_path'] == match) &
                                                  (edit_annotations['half'] == i)].reset_index(drop=True) for i in
                             range(2)]
            self.valid_frames_dict[match] = {}
            for half in range(2):
                valid_frames = set()
                for i in range(len(match_actions[half])):
                    position = match_actions[half]['position'][i]
                    frame_num = get_frame(position, frame_rate)
                    first_action_frame = frame_num - (int(self.window_size_sec / 2) * frame_rate)
                    last_action_frame = frame_num + (int(self.window_size_sec / 2) * frame_rate)
                    for frame_id in range(first_action_frame, last_action_frame + 1):
                        valid_frames.add(frame_id)
                self.valid_frames_dict[match][half] = [index for index in sorted(valid_frames)]

        # obtaining emotions list (emotions[match][half][frame] --> pooled emotion frame)
        self.emotions = {}
        for match_path in self.split_matches:
            emotion_features_path = self.path.joinpath(match_path, 'face_extraction_results.npy')
            match_emotion_features = np.load(emotion_features_path, allow_pickle=True)
            match_emotion_features = features_to_dict(match_emotion_features)
            self.emotions[match_path] = {}
            for half in range(2):
                self.emotions[match_path][half] = {}
                for frame in self.valid_frames_dict[match_path][half]:
                    if f'{frame:05}' in match_emotion_features[half].keys():
                        frame_emotion_vectors = [info[2] for id, info in
                                                 match_emotion_features[half][f'{frame:05}'].items()]
                        self.emotions[match_path][half][frame] = self.pool(np.asarray(frame_emotion_vectors), axis=0)
                    else:
                        self.emotions[match_path][half][frame] = np.random.randint(1, 10, 128) / 10000000

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

    def __getitem__(self, idx):
        match_path = self.split_matches_actions.loc[idx, 'match_path']
        half = self.split_matches_actions.loc[idx, 'half']
        position = self.split_matches_actions.loc[idx, 'position']
        label = self.split_matches_actions.loc[idx, 'label']

        frame_indices = np.arange(get_frame(position, self.frame_rate) - int(self.frames_per_window / 2),
                                  get_frame(position, self.frame_rate) + int(self.frames_per_window / 2) + 1)
        emotions_input = [self.emotions[match_path][half][index] for index in frame_indices]
        emotions_input = np.array(emotions_input, dtype=np.float32)
        return emotions_input, label - 1

    def __len__(self):
        return len(self.split_matches_actions)


if __name__ == '__main__':
    data_path = Path('../../../mnt/DATA/datasets/soccernet')
    videos_path = Path('videos_to_process.txt')

    with videos_path.open() as f:
        videos = [Path(line).parent for line in f.readlines()]

    dataset = SoccerNet(data_path, videos)
    print(len(dataset.__getitem__(1)[0]))
    print(dataset.__getitem__(1)[1])
