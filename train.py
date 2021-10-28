import torch
from datetime import datetime
def train_one_epoch(args):
    args.model.train()
    epoch_loss = []
    for x, y in args.train_dl:
        batch_loss = train_one_batch(args, x, y)
        epoch_loss.append(batch_loss)
    epoch_loss = torch.sum(torch.Tensor(epoch_loss)).item()
    return epoch_loss

def validate_one_epoch(args):
    args.model.eval()
    epoch_loss = []
    for x, y in args.val_dl:
        batch_loss = validate_one_batch(args, x, y)
        epoch_loss.append(batch_loss)
    epoch_loss = torch.sum(torch.Tensor(epoch_loss)).item()
    return epoch_loss

def train_one_batch(args, x, y):
    x, y = x.to(args.device), y.to(args.device)
    args.optim.zero_grad()
    out = args.model(x)
    loss = args.loss(out, y)
    loss.backward()
    args.optim.step()
    return loss.item()

def validate_one_batch(args, x, y):
    x, y = x.to(args.device), y.to(args.device)
    with torch.no_grad():
        out = args.model(x)
        loss = args.loss(out, y)
    return loss.item()

def fit(args):
    args.log = []
    args.vis_epoch, args.vis_train_loss, args.vis_val_loss = [], [], []
    args.model = args.model.to(args.device)
    time_start = datetime.now()
    for epoch in range(args.epochs):
        epoch_train_loss = train_one_epoch(args)
        epoch_val_loss = validate_one_epoch(args)
        log = 'epoch: {}, train_loss: {}, val_loss: {}'.format(epoch, epoch_train_loss, epoch_val_loss)
        args.log.append(log)
        args.vis_epoch.append(epoch)
        args.vis_train_loss.append(epoch_train_loss)
        args.vis_val_loss.append(epoch_val_loss)
        print(log)
    time_stop = datetime.now()
    args.train_time = time_stop - time_start
    return args