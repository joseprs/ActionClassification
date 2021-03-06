import os
import torch
import time
import logging.config
from addict import Dict
import json
from argparse import ArgumentParser
from pathlib import Path
from util import load_log_configuration, show_confusion_matrix
from torch.utils.data import DataLoader
from soccernet import SoccerNet
from model import ActionClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
# import plotext as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from imblearn.combine import SMOTEENN


def test(loader, model, cuda, data_set='Test', verbose=True, class_names=None, num_classes=17):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in loader:
            criterion = torch.nn.CrossEntropyLoss()

            if cuda:
                data, target = data.cuda(), target.cuda()
                criterion = criterion.cuda()

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            y_pred.extend(pred.data.cpu().numpy())
            y_true.extend(target.data.cpu().numpy())
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader)
    accuracy = float(correct) / len(loader.dataset)
    if verbose:
        logging.info(f'\n{data_set} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} '
                     f'({100 * accuracy:.1f}%)\n')

    if data_set == 'Test':
        cm = confusion_matrix(y_true, y_pred)
        print(np.unique(y_pred, return_counts=True))
        print(np.unique(y_true, return_counts=True))
        print(cm.diagonal() / cm.sum(axis=1))
        print(cm)  # / cm.sum(axis=1)
        print(cm.sum(axis=1))
        print(cm/cm.sum(axis=1)[np.newaxis].T)
        df_cm = pd.DataFrame(cm/cm.sum(axis=1)[np.newaxis].T, index=[i for i in class_names.values()],
                             columns=[i for i in class_names.values()])
        # class_names.values()
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('confusion_matrix.png')
        # show_confusion_matrix(y_true, y_pred, list(class_names.values()))

    return test_loss, accuracy


def train(loader, model, criterion, optimizer, epoch, cuda, class_weights, log_interval=100, verbose=True):
    model.train()
    epoch_loss = 0
    correct = 0
    class_weights = torch.tensor(class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for batch, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
            criterion = criterion.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if verbose and (batch % log_interval) == 0:
            logging.info(f'Train Epoch: {epoch} [{batch * len(data)}/{len(loader.dataset)} '
                         f'({100. * batch / len(loader):.0f}%)]\tLoss: {loss.data.item():.6f}')

    accuracy = float(correct) / len(loader.dataset)
    if verbose:
        logging.info(f'Training set: Average loss: {epoch_loss / len(loader):.4f}, '
                     f'Accuracy: {correct}/{len(loader.dataset)} ({100 * accuracy:.1f}%)')

    return epoch_loss / len(loader), accuracy


def main(model_args, opt_args, train_args, main_args):
    # Defining the model and select if it is CUDA or not
    model = ActionClassifier(pool=model_args.pool, window_size_sec=model_args.window_size_sec, num_classes=8)
    if main_args.cuda:
        model = model.cuda()
    logging.info(model)
    checkpoint = main_args.checkpoint_dir

    if main_args.train:
        # loading training and validation dataset
        with Path(train_args.splits.train[0]).open() as f:
            videos = [Path(line).parent for line in f.readlines()]
        training_dataset = SoccerNet(train_args.dataset_path, videos, window_size_sec=model_args.window_size_sec,
                                     pool=model_args.pool, balance=False, balance_th=800,
                                     face_th=model_args.face_area_th)
        training_loader = DataLoader(training_dataset, batch_size=opt_args.batch_size)
        # calculating train ratio of ball out of play
        train_actions = training_dataset.split_matches_actions[['label']]
        train_actions['sum'] = 1
        print(f"Max appearances and total ratio (train): {train_actions['label'].value_counts().max()}/"
              f"{len(train_actions)} = {train_actions['label'].value_counts().max() / len(train_actions)}")
        loss_weights1 = 1 - np.array(train_actions.groupby('label').sum() / len(train_actions), dtype=np.float32)

        loss_weights2 = 1 / np.array(train_actions.groupby('label').sum(), dtype=np.float32)

        loss_weights1 = np.reshape(loss_weights1, (len(loss_weights1),))
        loss_weights2 = np.reshape(loss_weights2, (len(loss_weights2),))

        with Path(train_args.splits.valid[0]).open() as f:
            videos = [Path(line).parent for line in f.readlines()]
        validation_dataset = SoccerNet(train_args.dataset_path, videos, window_size_sec=model_args.window_size_sec,
                                       pool=model_args.pool, balance=False, balance_th=200,
                                       face_th=model_args.face_area_th)
        validation_loader = DataLoader(validation_dataset, batch_size=opt_args.batch_size)

        # calculating validation ratio of ball out of play
        valid_actions = validation_dataset.split_matches_actions
        print(f"Max appearances and total ratio (validation): {valid_actions['label'].value_counts().max()}/"
              f"{len(valid_actions)} = {valid_actions['label'].value_counts().max() / len(valid_actions)}")

        # Training and Validation parameters
        # setting optimizer
        if opt_args.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt_args.learning_rate)
        elif opt_args.optimizer == 'ASGD':
            optimizer = torch.optim.ASGD(model.parameters(), lr=opt_args.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=opt_args.learning_rate, momentum=opt_args.momentum)

        # setting the criterion
        criterion = torch.nn.CrossEntropyLoss()  # maybe delete this line (define criterion inside training)

        # lists to plot
        train_loss_list = []
        valid_loss_list = []

        # start training
        best_valid_acc = 0
        iteration = 0
        epoch = 1

        # START TRAINING
        logging.info('Starting Training')
        while (epoch < opt_args.max_epochs + 1) and (iteration < opt_args.patience):

            train_loss, train_acc = train(training_loader, model, criterion, optimizer, epoch, main_args.cuda,
                                          loss_weights2)
            valid_loss, valid_acc = test(validation_loader, model, main_args.cuda, data_set='Validation')
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            if not os.path.isdir(checkpoint):
                os.mkdir(checkpoint)
            torch.save(model.module if args.cuda else model, f'./{checkpoint}/model{epoch:03d}.t7')

            if valid_acc <= best_valid_acc:
                iteration += 1
                logging.info(f'Accuracy was not improved, iteration {iteration}')
            else:
                logging.info('Saving state')
                iteration = 0
                best_valid_acc = valid_acc
                state = {
                    'epoch': epoch,
                    'valid_acc': valid_acc,
                    'valid_loss': valid_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)
                torch.save(state, f'./{checkpoint}/ckpt.t7')
            epoch += 1

        # plot train/valid loss function or accuracy
        plt.plot(train_loss_list, label='Train loss')
        plt.plot(valid_loss_list, label='Validation loss')
        plt.title("Train and Validation loss")
        plt.legend()
        plt.savefig('plots/loss_function.png')

    with Path(train_args.splits.test[0]).open() as f:
        videos = [Path(line).parent for line in f.readlines()]
    test_dataset = SoccerNet(train_args.dataset_path, videos, window_size_sec=model_args.window_size_sec,
                             pool=model_args.pool, balance=False, balance_th=200, face_th=model_args.face_area_th)
    test_loader = DataLoader(test_dataset, batch_size=opt_args.batch_size)

    # calculating validation ratio of ball out of play
    test_actions = test_dataset.split_matches_actions
    print(f"Max appearances and total ratio (test): {test_actions['label'].value_counts().max()}/"
          f"{len(test_actions)} = {test_actions['label'].value_counts().max() / len(test_actions)}")

    state = torch.load(f'./{checkpoint}/ckpt.t7')
    epoch = state['epoch']
    logging.info(f'\n\nTesting model {checkpoint} (epoch {epoch})')
    model = torch.load(f'./{checkpoint}/model{epoch:03d}.t7')
    if args.cuda:
        model = model.cuda()

    test(test_loader, model, main_args.cuda, class_names=test_dataset.class_names, num_classes=test_dataset.num_classes)


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
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--checkpoint', default='checkpoint', metavar='CHECKPOINT', help='checkpoints directory')
    parser.add_argument('--log_config', required=False,
                        help='Logging configuration file (default: config/log_config.yml)',
                        default="config/log_config.yml", type=lambda p: Path(p))
    parser.add_argument('--logs_dir', required=False,
                        help='Logging directory path (default: logs)',
                        default="logs", type=lambda p: Path(p))
    parser.add_argument('--train', required=False,
                        help='Perform testing only (default: False)',
                        action='store_false')

    args = parser.parse_args()
    json_fpath = args.conf
    with json_fpath.open() as json_file:
        conf = json.load(json_file)
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
                     'comment': comment})

    main_args = Dict({'train': args.train,
                      'cuda': args.cuda,
                      'checkpoint_dir': args.checkpoint,
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

    print(args.logs)
    load_log_configuration(args.logs.log_config, args.logs.log_dir)
    logging.info("Input arguments:")
    logging.info(args)

    args.main.cuda = args.main.cuda and torch.cuda.is_available()
    args.main.cuda = False
    torch.manual_seed(args.optimization.random_seed)
    if args.main.cuda:
        torch.cuda.manual_seed(args.optimization.random_seed)
        logging.info(f'Using CUDA with {torch.cuda.device_count()} GPUs')

    start = time.time()
    logging.info('Starting main function')
    main(args.model, args.optimization, args.training, args.main)
    logging.info(f'Total Execution Time is {time.time() - start} seconds')
