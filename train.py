# NN Training File
import os
import torch
import time
import numpy as np
import logging.config
from addict import Dict
from collections import OrderedDict
import json
from argparse import ArgumentParser
from pathlib import Path
from util import load_log_configuration
from torch.utils.data import DataLoader
from soccernet import SoccerNet
from model import ActionClassifier
from tqdm import tqdm
# from evaluation import evaluate

# from loss import NLLLoss
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple


def main(model_args, opt_args, train_args, main_args):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(opt_args)
    logging.info(train_args)

    model = ActionClassifier()
    logging.info(model)

    total_args = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_args))

    # dataset_parameters = get_dataset_parameters(model_args, train_args, main_args)

    if not main_args.test_only:

        with Path(train_args.splits.train[0]).open() as f:
            videos = [Path(line).parent for line in f.readlines()]

        training_dataset = SoccerNet(train_args.dataset_path, videos)
        training_loader = DataLoader(training_dataset,
                                     batch_size=opt_args.batch_size)
        #                              shuffle=True,
        #                              num_workers=train_args.max_num_workers,
        #                              pin_memory=True)
        #                              collate_fn=training_collate) ?

        # Read also Validation Dataset

        # Training and Validation parameters
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt_args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0,
                                     amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=opt_args.patience)
        writer = SummaryWriter(train_args.log_dir, train_args.comment)
        Loaders = namedtuple('Loaders', 'train valid')

        # TRAINER
        logging.info("start training")
        best_loss = np.Inf
        running_train_loss, running_valid_loss = 0., 0.
        best_model_path = model_args..joinpath("model.pth.tar")





def parse_args():
    parser = ArgumentParser(description='Emotional action spotting implementation')
    parser.add_argument('conf', help='JSON model configuration filepath',
                        type=lambda p: Path(p))
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    parser.add_argument('--split_train', nargs='+', dest='train_splits',
                        help='list of splits for training (default: [train])',
                        default=["train"])
    parser.add_argument('--split_valid', nargs='+', dest='validation_splits',
                        help='list of splits for validation (default: [valid])',
                        default=["valid"])
    parser.add_argument('--split_test', nargs='+', dest='test_splits',
                        help='list of split for testing (default: [test])',
                        default=["test"])

    parser.add_argument('--max_num_workers', required=False,
                        help='number of worker to load data (default: 4)',
                        default=4, type=int)
    parser.add_argument('--evaluation_frequency', required=False,
                        help='Evaluation frequency in number of epochs (default: 10)',
                        default=10, type=int)
    parser.add_argument('--weights_dir', required=False,
                        help='Path for weights saving directory (default: weights)',
                        default="weights", type=lambda p: Path(p))
    parser.add_argument('--weights', required=False,
                        help='Weights to load (default: None)',
                        default=None, type=str)
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    parser.add_argument('--logs_dir', required=False,
                        help='Logging directory path (default: logs)',
                        default="logs", type=lambda p: Path(p))
    parser.add_argument('--test_only', required=False,
                        help='Perform testing only (default: False)',
                        action='store_true')

    args = parser.parse_args()
    json_fpath = args.conf
    with json_fpath.open() as json_file:
        conf = json.load(json_file, object_pairs_hook=OrderedDict)
        conf = Dict(conf)

    comment_tmp = f'win_size:{0} frame_rate:{1} lr:{2} batch_size:{3}'
    comment = comment_tmp.format(conf.model.window_size_sec,
                                 conf.model.frame_rate,
                                 conf.optimization.learning_rate,
                                 conf.optimization.batch_size)

    training = Dict({'dataset_path': args.dataset_path,
                     'splits': {'train': args.train_splits,
                                'valid': args.validation_splits,
                                'test': args.test_splits},
                     'max_num_workers': args.max_num_workers,
                     'evaluation_frequency': args.evaluation_frequency,
                     'weights': args.weights,
                     'log_dir': args.logs_dir.joinpath('runs', comment),
                     'comment': comment})

    main_args = Dict({'test_only': args.test_only,
                      # 'test_batch_size': args.test_batch_size,
                      # 'tiny': args.tiny,
                      # 'num_random_samples': args.num_random_samples,
                      # 'cache': not args.no_cache,
                      # 'overwrite_cache': args.overwrite_cache,
                      # 'overwrite_test_results': args.overwrite_test_results
                      })

    logs_args = {'log_config': args.log_config, 'log_dir': args.logs_dir}

    return Dict({'model': conf.model,
                 'optimization': conf.optimization,
                 'training': training,
                 'logs': logs_args,
                 # 'GPU': args.GPU,
                 'main': main_args})


if __name__ == '__main__':
    args = parse_args()
    # args.logs.log_fpath.parent.mkdir(parents=True, exist_ok=True)
    # logging.config.dictConfig(args.logs.log_config)
    print(args.logs)
    load_log_configuration(args.logs.log_config, args.logs.log_dir)

    logging.info("Input arguments:")
    logging.info(args)

    torch.manual_seed(args.optimization.random_seed)
    np.random.seed(args.optimization.random_seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info('Starting main function')
    main(args.model, args.optimization, args.training, args.main)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
