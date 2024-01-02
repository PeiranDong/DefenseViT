import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .augmentation import train_augmentation, test_augmentation
from .poisoned_dataset import PoisonedDataset
import torch

def load_cifar10(datadir: str, img_size: int, mean: tuple, std: tuple):

    trainset = datasets.CIFAR10(
        root      = os.path.join(datadir, 'CIFAR10'),
        train     = True, 
        download  = True,
        # transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )

    testset = datasets.CIFAR10(
        root      = os.path.join(datadir, 'CIFAR10'),
        train     = False, 
        download  = True,
        # transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100(datadir: str, img_size: int, mean: tuple, std: tuple):

    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir, 'CIFAR100'),
        train     = True, 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )

    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_svhn(datadir: str, img_size: int, mean: tuple, std: tuple):

    trainset = datasets.SVHN(
        root      = os.path.join(datadir, 'SVHN'),
        split     = 'train', 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )

    testset = datasets.SVHN(
        root      = os.path.join(datadir, 'SVHN'),
        split     = 'test', 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


def load_tiny_imagenet_200(datadir: str, img_size: int, mean: tuple, std: tuple):

    trainset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200', 'train'),
        transform = train_augmentation(img_size=img_size, mean=mean, std=std)
    )
    
    testset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200', 'val'),
        transform = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset


## original dataloader
def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 16
    )


## load poisoned cifar10 for training
def create_backdoor_dataloader(dataname, train_data, test_data, trigger_label, poisoned_portion, batch_size, device,
                               mark_dir=None, alpha=1.0, trigger_type="BadNet_Patch"):
    train_data_ori = PoisonedDataset(train_data, trigger_label, portion=0, mode="train", device=device,
                                     dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=True, trigger_type=trigger_type)
    train_data_tri = PoisonedDataset(train_data, trigger_label, portion=poisoned_portion, mode="train", device=device,
                                     dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=True, trigger_type=trigger_type)
    test_data_ori  = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device,
                                     dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False, trigger_type=trigger_type)
    test_data_tri  = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device,
                                     dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False, trigger_type=trigger_type)

    if device == torch.device("cpu"):
        train_data_ori_loader = DataLoader(dataset=train_data_ori, batch_size=batch_size['train'], shuffle=True,
                                           num_workers=16, pin_memory=True)
        train_data_tri_loader   = DataLoader(dataset=train_data_tri,    batch_size=batch_size['train'], shuffle=True,  num_workers=16, pin_memory=True)
        test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size['test'],  shuffle=False, num_workers=16, pin_memory=True)
        test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size['test'],  shuffle=False, num_workers=16, pin_memory=True)
    else:
        train_data_ori_loader = DataLoader(dataset=train_data_ori, batch_size=batch_size['train'], shuffle=True)
        train_data_tri_loader = DataLoader(dataset=train_data_tri,    batch_size=batch_size['train'], shuffle=True)
        test_data_ori_loader  = DataLoader(dataset=test_data_ori, batch_size=batch_size['test'],  shuffle=False)
        test_data_tri_loader  = DataLoader(dataset=test_data_tri, batch_size=batch_size['test'],  shuffle=False)

    return train_data_ori_loader, train_data_tri_loader, test_data_ori_loader, test_data_tri_loader