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
from SoccerNet.Evaluation.utils import AverageMeter
from sklearn.metrics import average_precision_score as avg_prec_score

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple


def test(dataloader, model):
    batch_time, data_time = AverageMeter(), AverageMeter()
    description = 'Test (cls): ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) '
    model.eval()
    start_time = time.time()
    all_labels, all_outputs = [], []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (features, labels) in t:
            # measure data loading time
            data_time.update(time.time() - start_time)

            # if isinstance(features, list):
            #     features = [f.cuda() for f in features]
            #     output = model(features[0]) if len(features) == 1 else model(features)
            # else:
            #     features = features.cuda()
            #     output = model(features)
            output = model(features)
            all_labels.append(labels.detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            t.set_description(description.format(avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val))

    AP = []
    for i in range(1, dataloader.dataset.num_classes):
        AP.append(avg_prec_score(np.concatenate(all_labels)[:, i],
                                 np.concatenate(all_outputs)[:, i]))
    return np.mean(AP)


def train(loader, model, criterion, optimizer, epoch, evaluate_=False):
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    description = '{mode} {epoch}: ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) ' \
                  'Loss {loss:.4e}'

    # Control if we are doing the Validation or Training
    if evaluate_:
        model.eval()
        mode = 'Evaluate'
    else:
        model.train()
        mode = 'Train'

    training_start = time.time()

    # Start training out model: for that goes Batch by Batch
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (features, labels) in t:
            data_time.update(time.time() - training_start)

            # passing labels and features to cuda and running our model
            # labels = labels.cuda()
            # if isinstance(features, list):
            #     features = [f.cuda() for f in features]
            #     output = model(features[0]) if len(features) == 1 else model(features)
            # else:
            # features = features.cuda()
            output = model(features)
            # calculating loss and adding to losses array
            loss = criterion(output, labels)  # TODO: output[0] o [1]? 1 is softmax result
            losses.update(loss.item(), features[0].size(0))

            # if we are in training mode --> backpropagation and optimyze weights
            if not evaluate_:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - training_start)
            training_start = time.time()

            t.set_description(description.format(mode=mode,
                                                 epoch=epoch,
                                                 avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val,
                                                 loss=losses.avg))

        return losses.avg


def trainer(loaders, model, optimizer, scheduler, criterion, writer, weights_dir: Path, max_epochs=1000,
            evaluation_frequency=20):
    logging.info("Training STARTED")

    # first parameters
    best_loss = np.Inf
    running_train_loss, running_valid_loss = 0., 0.
    best_model_path = weights_dir.joinpath("model.pth.tar")

    for epoch in range(max_epochs):
        # for every epoch, we train our feature batches
        training_loss = train(loaders.train, model, criterion, optimizer, epoch + 1)
        running_train_loss += training_loss

        # evaluate with validation set
        validation_loss = train(loaders.valid, model, criterion, optimizer, epoch + 1, True)
        running_valid_loss += validation_loss

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        # if validation loss is better than best loss, then we save
        if validation_loss < best_loss:
            torch.save(state, best_model_path)
        best_loss = min(validation_loss, best_loss)

        # TODO: review remaining piece of code
        if (epoch + 1) % evaluation_frequency == 0:
            writer.add_scalar('training loss',
                              running_train_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.train))

            writer.add_scalar('validation loss',
                              running_valid_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.valid))
            validation_mAP = test(loaders.valid, model)
            logging.info(f"Validation mAP at epoch {epoch + 1} -> {validation_mAP}")

            running_train_loss, running_valid_loss = 0.0, 0.0

        # Reduce LR on Plateau after patience reached
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(validation_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr is not prev_lr and scheduler.num_bad_epochs == 0:
            logging.info("Plateau Reached!")

        if prev_lr < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            break

    return


def main(model_args, opt_args, train_args, main_args):
    logging.info("Parameters:")
    logging.info(model_args)
    logging.info(opt_args)
    logging.info(train_args)

    model = ActionClassifier()
    logging.info(model)

    total_args = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_args))

    if not main_args.test_only:
        with Path(train_args.splits.train[0]).open() as f:
            videos = [Path(line).parent for line in f.readlines()]
        training_dataset = SoccerNet(train_args.dataset_path, videos)
        training_loader = DataLoader(training_dataset, batch_size=opt_args.batch_size)

        with Path(train_args.splits.valid[0]).open() as f:
            videos = [Path(line).parent for line in f.readlines()]
        validation_dataset = SoccerNet(train_args.dataset_path, videos)
        validation_loader = DataLoader(validation_dataset, batch_size=opt_args.batch_size)

        # Training and Validation parameters
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt_args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0,
                                     amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                                               patience=opt_args.patience)
        writer = SummaryWriter(train_args.log_dir, train_args.comment)
        Loaders = namedtuple('Loaders', 'train valid')

        trainer(Loaders(training_loader, validation_loader), model, optimizer, scheduler, criterion, writer,
                train_args.weights_dir, opt_args.max_epochs, train_args.evaluation_frequency)

        writer.close()
        del training_loader, training_dataset
        del validation_loader, validation_dataset
        # TODO: review remaining piece of code


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
                     'weights_dir': args.weights_dir,
                     'log_dir': args.logs_dir.joinpath('runs', comment),
                     'comment': comment})

    main_args = Dict({'test_only': args.test_only,
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
