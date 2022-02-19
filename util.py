import logging.config
from datetime import datetime
from pathlib import Path
<<<<<<< HEAD
import yaml

=======
>>>>>>> c06e396b5986d0976ff35aaedc1e458a1a7b8475

def load_log_configuration(log_config: Path, logs_dir: Path, log_fname_format='%Y-%m-%d_%H-%M-%S.log'):
    log_fname = datetime.now().strftime(log_fname_format)
    log_fpath = logs_dir.joinpath(log_fname)
    with log_config.open(mode='rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = str(log_fpath)

    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(log_config)
