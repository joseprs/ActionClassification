import csv
from math import ceil
from operator import mul
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class SoccerNet(data.Dataset):

    def __init__(self, data_dir: Path, matches: List[Path], window_size_sec=20, frame_rate=8, **kwargs):

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

        # CAMBIAR AQUI
        self.matches = matches
        self.labels = []
        # self._load_labels()

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

    def _load_labels(self, match_path, half, num_batches):
        labels = np.zeros((num_batches, self.num_classes))
        for r in self.annotations.loc[(match_path, half)].itertuples():
            index = r.game_time // self.window_size_sec
            if index < num_batches:
                labels[index, r.label] = 1
        self.labels.append(labels)
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':

    data_path = Path('../soccernet_dataset')
    matches = list()
    matches.append('FCB-RMA')
    matches.append('CHE-ARS')
    dataset = SoccerNet(data_path, matches)
    print(dataset.annotations.columns)

    # dataset getitem returns: (list of 160 vectors per action, action name)
    

