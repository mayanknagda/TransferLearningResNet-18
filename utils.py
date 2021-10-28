import torch
import torch.nn as nn
import numpy as np
import logging
import os
from datetime import datetime
import random

def seed_experiment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optim(args):
    if args.optim_type == 'Adam':
        args.optim = torch.optim.Adam(params=args.model.parameters(), lr=args.lr)
    
    if args.optim_type == 'SGD':
        args.optim = torch.optim.SGD(params=args.model.parameters(), lr=args.lr)
    return args

def get_loss_fn(args):
    args.loss = nn.CrossEntropyLoss()
    return args

def test(args):
    pred, label = [], []
    start_time = datetime.now()
    for x, y in args.test_dl:
        x = x.to(args.device)
        out = args.model(x)
        out = out.cpu().detach().numpy()
        y = y.numpy()
        for i in out:
            pred.append(i)
        for i in y:
            label.append(i)
    stop_time = datetime.now()
    args.predictions, args.labels = np.array(pred), np.array(label)
    args.test_time = stop_time - start_time
    return args

def logoutput(args):
    output_dir = 'output/' + args.output_dir
    os.mkdir(output_dir)
    logfile = output_dir + '/logfile.log'
    open(logfile[0:], 'a').close()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile[0:])
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    logger.info('---Start Logging---')
    for hist in args.log:
        logger.info(hist)
    logger.info('---Stop Logging---')
    logger.info('---Model Info---')
    logger.info((args.model))
    logger.info('---Time Taken---')
    logger.info('Training: {}'.format(args.train_time))
    logger.info('Testing: {}'.format(args.test_time))
    logger.removeHandler(file_handler)
    del logger, file_handler
    save_predictions = output_dir + '/predictions.npy'
    save_labels = output_dir + '/labels.npy'
    np.save(save_predictions, args.predictions)
    np.save(save_labels, args.labels)