import logging
import random
import numpy as np 
import torch 
import datetime
import os
import re
import time
import pandas as pd
from torchvision.utils import save_image

def get_logger(logpath, displaying=True, saving=True):
    logger = logging.getLogger()
    level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    # logger.info(filepath)
    return logger

def get_logger_name(ckpt_path, load_ckpt, task, severity=None, threat=None):
    
    save_dir = os.path.join(ckpt_path, 'test')
    os.makedirs(save_dir, exist_ok=True)
    s = re.search(r"e\d+", load_ckpt)
    
    if 'wo' in load_ckpt:
        postfix = "-wo"
    elif 'en' in load_ckpt:
        postfix = "-en"
    elif s:
        postfix = "-"+str(s.group())
    else:
        postfix = ""
    
    if task =='ood':
        name = os.path.join(save_dir, 'test-{}{}.log'.format(task, postfix))
    elif task == 'adv':
        name = os.path.join(save_dir, 'test-{}-{}{}.log'.format(task, threat, postfix))
    else:
        raise
    return name

def get_desc(args):
    if args.protocol=='pdeadd':
        desc = "{}_{}-gmm{}_{}_(C{}lr{}{}-D{}lr{})_e{}_b{}_aug-{}_augdiff-{}_atk-{}".format(
                args.backbone, args.protocol, args.use_gmm,
                args.desc, args.optimizerC, args.lrC, args.scheduler,
                args.optimizerDiff, args.lrDiff, 
                args.epoch, args.batch_size, args.aug_train, args.aug_train_diff, args.atk_train)
    elif args.protocol == 'standard'  or 'fixdiff' in args.protocol :
        desc = "{}_{}_{}_C{}lr{}{}_e{}_b{}_aug-{}_atk-{}".format(
                args.backbone, args.protocol,
                args.desc, args.optimizerC, args.lrC, args.scheduler,
                args.epoch, args.batch_size, args.aug_train, args.atk_train)
    else:
        raise  
    return desc

def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

class BestSaver():
    def __init__(self):
        self.best_acc=0
        self.best_epoch=0
    def apply(self, acc, epoch, model, optimizerC, scheduler, optimizerDiff=None, save_path=None):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            if epoch > 100:
                self.save_model(model, optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, save_path=save_path)
    
    def save_model(self, model, optimizerC, scheduler=None, optimizerDiff=None,  save_path=None):
        if scheduler:
            scheduler_state_dict = scheduler.state_dict()
        else:
            scheduler_state_dict = None
        
        if optimizerDiff:
            torch.save(
            {'model_state_dict': model.state_dict(),
            'optimizerC_state_dict': optimizerC.state_dict(),
            'optimizerDiff_state_dict': optimizerDiff.state_dict(),
            'scheduler':scheduler_state_dict
            }, save_path)
        else:
            torch.save(
            {'model_state_dict': model.state_dict(),
            'optimizerC_state_dict': optimizerC.state_dict(),
            'scheduler':scheduler_state_dict
            }, save_path)

def verbose_and_save(logger, epoch, start_epoch, eval_per_epoch, start, train_metric, eval_metric, writer):
    # save logs
    logger.info('\n[Epoch {}] - Time taken: {}'.format(epoch, format_time(time.time()-start)))
    logger.info('Train\t Acc: {:.2f}%, NLLLoss: {:.2f}, ClassLoss: {:.2f}'.format(
                train_metric['train_acc'],train_metric['train_loss_nll'],train_metric['train_loss_cla']))
    logger.info('Train\t Scale1: {:.4f}, Scale2: {:.4f}, Scale3: {:.4f}, Scale4: {:.4f}'.format(
                train_metric['scales_l1'],train_metric['scales_l2'],train_metric['scales_l3'],train_metric['scales_l4']))
    if (epoch == start_epoch) or (epoch % eval_per_epoch == 0):
        logger.info('Eval Nature Samples\t Acc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_metric["nat"]['eval_acc'],eval_metric["nat"]['eval_loss']))     
        logger.info('Eval O.O.D. Samples\t Acc: {:.2f}%'.format(eval_metric["ood"]['eval_acc']))
    logger.info("\n")
    # logger.info('Eval O.O.D. Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}\nendiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
    #     eval_metric["ood"]['eval_acc'],eval_metric["ood"]['eval_loss'],
    #     eval_metric["ood_diff"]['eval_acc'],eval_metric["ood_diff"]['eval_loss']))
    
    # save tensorboard
    writer.add_scalar('train/lossDiff', train_metric['train_loss_nll'], epoch)
    writer.add_scalar('train/lossC', train_metric['train_loss_cla'], epoch)
    writer.add_scalar('train/acc', train_metric['train_acc'], epoch)

    writer.add_scalar('scales/layer1', train_metric['scales_l1'], epoch)
    writer.add_scalar('scales/layer2', train_metric['scales_l2'], epoch)
    writer.add_scalar('scales/layer3', train_metric['scales_l3'], epoch)
    writer.add_scalar('scales/layer4', train_metric['scales_l4'], epoch)
    
    writer.add_scalar('evalnat/loss', eval_metric["nat"]['eval_loss'], epoch)
    writer.add_scalar('evalnat/acc', eval_metric["nat"]['eval_acc'], epoch)
    
    # writer_eval_diff.add_scalar('evalood/loss', eval_metric["ood"]['eval_loss'], epoch)
    writer.add_scalar('evalood/acc', eval_metric["ood"]['eval_acc'], epoch)
    # writer_eval_diff.add_scalar('evalood/loss', eval_metric["ood_diff"]['eval_loss'], epoch)
    # writer_eval_diff.add_scalar('evalood/acc', eval_metric["ood_diff"]['eval_acc'], epoch)


def eval_epoch(epoch):
    if epoch < 5:
        eval_per_epoch = 1
    elif epoch < 100:
        eval_per_epoch = 20
    elif epoch < 160:
        eval_per_epoch = 10
    else:
        eval_per_epoch = 1
    return eval_per_epoch

def vis(x_ori, x_aug, save_path):
    x=torch.cat([x_ori[:8],x_aug[:8]])
    save_image(x.cpu(), os.path.join(save_path,'samples.png'), nrow=8,padding=0, value_range=(0, 1), pad_value=0)
