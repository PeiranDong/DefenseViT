import numpy as np
import pandas as pd
import os
import random
import wandb
import csv

import torch
import argparse
import timm
import logging
import yaml
import datetime

from stats import dataset_stats
from train import fit
from timm import create_model
from datasets import create_dataloader, create_backdoor_dataloader
from log import setup_default_logging
from models import VPT

_logger = logging.getLogger('train')

## set seed
def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


## run model for training and testing
def run(cfg):
    # make save directory
    time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"Backdoor_{cfg['TRAINING']['poison_rate']}"
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname'], cfg['EXP_NAME'], time_path)
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir, 'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    if cfg['MODEL']['prompt_type']:
        model = VPT(
            modelname         = cfg['MODEL']['modelname'],
            num_classes       = cfg['DATASET']['num_classes'],
            pretrained        = True,
            prompt_tokens     = cfg['MODEL']['prompt_tokens'],
            prompt_dropout    = cfg['MODEL']['prompt_dropout'],
            prompt_type       = cfg['MODEL']['prompt_type'],
            backbone_unfreeze = cfg['TRAINING']['backbone_unfreeze']
        )
    else:
        model = create_model(
            model_name      = cfg['MODEL']['modelname'],
            num_classes    = cfg['DATASET']['num_classes'],
            pretrained     = True,
        )
    model.to(device)
    _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir            = cfg['DATASET']['datadir'],
        img_size           = cfg['DATASET']['img_size'],
        mean               = cfg['DATASET']['mean'],
        std                = cfg['DATASET']['std']
    )

    # sampling 1k 
    sample_df = pd.read_csv(f"{cfg['DATASET']['dataname']}_1k_sample.csv")
    trainset.data = trainset.data[sample_df.sample_index]
    trainset.targets = np.array(trainset.targets)[sample_df.sample_index]

    ## original load dataloader
    # trainloader    = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    # testloader     = create_dataloader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)
    # tri_testloader = None

    ## TODO: change the dataloader to poisoned dataloader
    trigger_label = 0
    """Select the mark type"""
    mark_dir = None  # a white square at the right bottom corner ## todo: try blended attack
    # mark_dir = './marks/hello_kitty.jpg'
    # mark_dir = './marks/apple_white.png'
    # mark_dir = './marks/apple_black.png'
    # mark_dir = './marks/watermark_white.png'
    # mark_dir = './marks/watermark_black.png'
    alpha = 0.1  # mark transparency, only available when `mark_dir` is specified
    poisoned_portion = cfg['TRAINING']['poison_rate'] # poison_rate in training dataset
    show_num = 2  # number of inputs to be shown

    ## 建立 干净/中毒 训练集/测试集 共4个数据集
    ori_trainloader, tri_trainloader, testloader, tri_testloader = create_backdoor_dataloader(
        dataname='CIFAR10',
        train_data=trainset,
        test_data=testset,
        trigger_label=trigger_label,
        poisoned_portion=poisoned_portion,
        batch_size={"train": cfg['TRAINING']['batch_size'], "test": cfg['TRAINING']['test_batch_size']},
        device=device,
        mark_dir=mark_dir,
        alpha=alpha,
        trigger_type=cfg['BACKDOOR']['trigger_type'])



    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), **cfg['OPTIMIZER']['params'])

    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['clean_epochs'] + cfg['TRAINING']['backdoor_epochs'])
    else:
        scheduler = None

    if cfg['TRAINING']['use_wandb']:
        # initialize wandb
        wandb.init(name=cfg['EXP_NAME'], project='Visual Prompt Tuning', config=cfg)

    # fitting model
    best_acc, best_asr = \
    fit(model           = model,
        trainloader = ori_trainloader,
        tri_trainloader = tri_trainloader,
        testloader      = testloader,
        tri_testloader  = tri_testloader,
        criterion       = criterion,
        optimizer       = optimizer,
        scheduler       = scheduler,
        clean_epochs    = cfg['TRAINING']['clean_epochs'],
        backdoor_epochs = cfg['TRAINING']['backdoor_epochs'],
        savedir         = savedir,
        log_interval    = cfg['TRAINING']['log_interval'],
        device          = device,
        use_wandb       = cfg['TRAINING']['use_wandb'])

    # finish wandb
    if cfg['TRAINING']['use_wandb']:

        wandb.save("wandb_result.h5")
        wandb.finish()

    return savedir, best_acc, best_asr


def multi_setting(cfg):
    ## models
    modelnames = ["vit_small_patch16_224", "vit_base_patch16_224"]

    ## prompt type
    prompt_types = ["shallow", "deep"]

    ## length of prompt tokens
    prompt_tokens = [5, 20]
    # prompt_tokens = [0]

    ## poison_rate
    # poison_rates = [0.1, 0.2, 0.5]
    poison_rates = [0]


    ## trigger 4*4

    summary_result = []
    time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


    for modelname in modelnames:
        for prompt_type in prompt_types:
            for prompt_token in prompt_tokens:
                for poison_rate in poison_rates:
                    cfg['MODEL']['modelname'] = modelname
                    cfg['MODEL']['prompt_type'] = prompt_type
                    cfg['MODEL']['prompt_tokens'] = prompt_token
                    cfg['TRAINING']['poison_rate'] = poison_rate

                    # cfg['EXP_NAME'] = f"{args.modelname}-{args.prompt_type}-n_prompts{args.prompt_tokens}" if args.prompt_type else args.modelname
                    cfg['EXP_NAME'] = f"CIFAR10-backdoor-{cfg['BACKDOOR']['trigger_type']}"
                    savedir, best_acc, best_asr = run(cfg)  ## run the train and test epoch
                    summary_result.append([savedir, best_acc, best_asr])

    name = ["setting", "best_acc", "best_asr"]
    pd_list = pd.DataFrame(columns=name, data=summary_result)
    print(pd_list)
    result_path = './results'
    os.makedirs(result_path, exist_ok=True)
    pd_list.to_csv(os.path.join(result_path, f'summary_result_{time_path}.csv'), encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visual Prompt Tuning')
    parser.add_argument('--default_setting', type=str, default='default_configs.yaml', help='exp config file')
    parser.add_argument('--modelname', type=str, default='vit_small_patch16_224', help='model name')
    parser.add_argument('--dataname', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN', 'Tiny_ImageNet_200'], help='data name')
    parser.add_argument('--img_resize', type=int, default=224, help='Image Resize')
    parser.add_argument('--prompt_type', type=str, default='', choices=['shallow', 'deep', ''], help='prompt type')
    parser.add_argument('--prompt_tokens', type=int, default=0, help='number of prompt tokens')
    parser.add_argument('--prompt_dropout', type=float, default=0.0, help='prompt dropout rate')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='use wandb')
    parser.add_argument('--backbone_unfreeze', action='store_false', default=True, help='whether unfreeze VIT as a backbone')

    ## parameters about backdoor
    parser.add_argument('--poison_rate', type=float, default=0.1, help='the poison rate in train dataset')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.default_setting, 'r'), Loader=yaml.FullLoader)

    d_stats = dataset_stats[args.dataname.lower()]

    cfg['MODEL'] = {}
    cfg['MODEL']['modelname'] = args.modelname
    cfg['MODEL']['prompt_type'] = args.prompt_type
    cfg['MODEL']['prompt_tokens'] = args.prompt_tokens
    cfg['MODEL']['prompt_dropout'] = args.prompt_dropout
    cfg['DATASET']['num_classes'] = d_stats['num_classes']
    cfg['DATASET']['dataname'] = args.dataname
    cfg['DATASET']['img_size'] = args.img_resize if args.img_resize else d_stats['img_size']
    cfg['DATASET']['mean'] = d_stats['mean']
    cfg['DATASET']['std'] = d_stats['std']
    cfg['TRAINING']['use_wandb'] = args.use_wandb
    cfg['TRAINING']['backbone_unfreeze'] = args.backbone_unfreeze
    cfg['TRAINING']['poison_rate'] = args.poison_rate


    # cfg['EXP_NAME'] = f"{args.modelname}-{args.prompt_type}-n_prompts{args.prompt_tokens}-unfreeze{args.backbone_unfreeze}-" \
    #                   f"c_epochs{cfg['TRAINING']['clean_epochs']}-b_epochs{cfg['TRAINING']['backdoor_epochs']}-{cfg['BACKDOOR']['trigger_type']}-poi_rate{args.poison_rate}" \
    #     if args.prompt_type else args.modelname

    cfg['EXP_NAME'] = f"{args.modelname}-unfreeze{args.backbone_unfreeze}-c_epochs{cfg['TRAINING']['clean_epochs']}-b_epochs{cfg['TRAINING']['backdoor_epochs']}" \
                      f"-{cfg['BACKDOOR']['trigger_type']}-poi_rate{args.poison_rate}"


    ## one train/test experiment
    run(cfg)

    ## run multi attack setting
    # multi_setting(cfg)