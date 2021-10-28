from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR100
import torchvision.transforms as T

def get_data(args):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = CIFAR100(root=args.data_location,
                         train=True,
                         transform=transform,
                         download=True)
    test_data = CIFAR100(root=args.data_location,
                         train=False,
                         transform=transform,
                         download=True)
    #train_ds, val_ds = random_split(train_data, [int(len(train_data)*0.9), int(len(train_data)*0.1)])
    args.train_dl = DataLoader(train_data, batch_size=args.batch_size, num_workers=4)
    args.val_dl = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)
    args.test_dl = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)
    return args