import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict

_logger = logging.getLogger('train')


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        # targets = targets.to(torch.int64)
        # TODO: Add a line above to solve RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
        loss = criterion(outputs, targets)    
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1)
        targets = targets.argmax(dim=1)
        # TODO: Add a line above to solve RuntimeError: The size of tensor a (10) must match the size of tensor b (256) at non-singleton dimension 1
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
    
        if idx % log_interval == 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('acc', acc_m.avg), ('loss', losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval: int, device: str) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            targets = targets.argmax(dim=1)
            # TODO: Add a line above to solve RuntimeError: The size of tensor a (10) must match the size of tensor b (256) at non-singleton dimension 1
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % log_interval == 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc', correct/total), ('loss', total_loss/len(dataloader))])



## train and test model
# def fit(
#     model, trainloader, testloader, tri_testloader, criterion, optimizer, scheduler,
#     clean_epochs: int, savedir: str, log_interval: int, device: str, use_wandb: bool,
# ) -> None:
def fit(
            model, trainloader, tri_trainloader, testloader, tri_testloader, criterion, optimizer, scheduler,
            clean_epochs: int, backdoor_epochs: int, savedir: str, log_interval: int, device: str, use_wandb: bool,
    ):

    best_acc = 0
    best_asr = 0
    step = 0

    ## 先进行一段时间(100 epoches)的clean数据训练，再进行一段时间(50~100epoches)的后门数据投毒
    for epoch in range(clean_epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{clean_epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device)   ## train epoch
        print("clean testdataset eval start!")
        eval_metrics = test(model, testloader, criterion, log_interval, device)                 ## eval the clean dataset
        if tri_testloader:
            print("triggered testdataset eval start!")
            eval_tri_metrics = test(model, tri_testloader, criterion, log_interval, device)     ## eval the poisoned test dataset

        if use_wandb:
            # wandb
            metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            metrics.update([('eval_tri_' + k, v) for k, v in eval_tri_metrics.items()])
            wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch': epoch, 'best_acc': eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'), 'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

        ## backdoored checkpoint
        if best_asr < eval_tri_metrics['acc']:
            # save results
            tri_state = {'best_tri_epoch': epoch, 'best_asr': eval_tri_metrics['acc']}
            json.dump(tri_state, open(os.path.join(savedir, f'best_tri_results.json'), 'w'), indent=4)

            # save attack model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_tri_model.pt'))

            _logger.info('Best Trigger Accuracy {0:.3%} to {1:.3%}'.format(best_asr, eval_tri_metrics['acc']))

            best_asr = eval_tri_metrics['acc']

    ## backdoor_epoches
    for epoch in range(clean_epochs, clean_epochs + backdoor_epochs, 1):
        _logger.info(f'\nEpoch: {epoch + 1}/{clean_epochs}')
        train_metrics = train(model, tri_trainloader, criterion, optimizer, log_interval, device)  ## train epoch
        print("clean testdataset eval start!")
        eval_metrics = test(model, testloader, criterion, log_interval, device)  ## eval the clean dataset
        if tri_testloader:
            print("triggered testdataset eval start!")
            eval_tri_metrics = test(model, tri_testloader, criterion, log_interval,
                                    device)  ## eval the poisoned test dataset

        if use_wandb:
            # wandb
            metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            metrics.update([('eval_tri_' + k, v) for k, v in eval_tri_metrics.items()])
            wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch': epoch, 'best_acc': eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'), 'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))

            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

        ## backdoored checkpoint
        if best_asr < eval_tri_metrics['acc']:
            # save results
            tri_state = {'best_tri_epoch': epoch, 'best_asr': eval_tri_metrics['acc']}
            json.dump(tri_state, open(os.path.join(savedir, f'best_tri_results.json'), 'w'), indent=4)

            # save attack model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_tri_model.pt'))

            _logger.info('Best Trigger Accuracy {0:.3%} to {1:.3%}'.format(best_asr, eval_tri_metrics['acc']))

            best_asr = eval_tri_metrics['acc']



    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))
    _logger.info('Best Triggered Metric: {0:.3%} (epoch {1:})'.format(tri_state['best_asr'], tri_state['best_tri_epoch']))
    return best_acc, best_asr

