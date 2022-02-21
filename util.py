import logging.config
from datetime import datetime
from pathlib import Path
import yaml
import os


def load_log_configuration(log_config: Path, logs_dir: Path, log_fname_format='%Y-%m-%d_%H-%M-%S.log'):
    log_fname = datetime.now().strftime(log_fname_format)
    log_fpath = logs_dir.joinpath(log_fname)
    with log_config.open(mode='rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = str(log_fpath)

    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(log_config)


def get_frame(position, fs):
    return (position // 1000) * fs


def first_last_frame(frame_num, seconds_window, fs, frames_dir):
    if frame_num - (seconds_window * fs) > 0:
        f = frame_num - (seconds_window * fs)
    else:
        f = 1
    if frame_num + (seconds_window * fs) > len(os.listdir(frames_dir)):
        l = len(os.listdir(frames_dir))
    else:
        l = frame_num + (seconds_window * fs)

    return f, l
