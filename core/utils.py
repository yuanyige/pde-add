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
        desc = "{}_{}-{}_ntr{}_(C{}lr{}{}-D{}lr{})_e{}_b{}_aug-{}_augdiff-{}_atk-{}".format(
                args.backbone, args.protocol, 
                args.desc, args.npc_train,  
                args.optimizerC, args.lrC, args.scheduler,
                args.optimizerDiff, args.lrDiff, 
                args.epoch, args.batch_size, args.aug_train, args.aug_train_diff, args.atk_train)
    elif args.protocol == 'standard'  or 'fixdiff' in args.protocol :
        desc = "{}_{}-{}_ntr{}_C{}lr{}{}_e{}_b{}_aug-{}_atk-{}".format(
                args.backbone, args.protocol, 
                args.desc, args.npc_train, 
                args.optimizerC, args.lrC, args.scheduler,
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

def verbose_and_save(logger, epoch, start, train_metric, eval_metric, writer_train, writer_eval, writer_eval_diff, total_metrics, args):
    # save logs
    logger.info('\n[Epoch {}] - Time taken: {}'.format(epoch, format_time(time.time()-start)))
    logger.info('Train\t Acc: {:.2f}%, NLLLoss: {:.2f}, ClassLoss: {:.2f}'.format(
                train_metric['train_acc'],train_metric['train_loss_nll'],train_metric['train_loss_cla']))
    logger.info('Train\t Scale1: {:.2f}, Scale2: {:.2f}, Scale3: {:.2f}, Scale4: {:.2f}'.format(
                train_metric['scales_l1'],train_metric['scales_l2'],train_metric['scales_l3'],train_metric['scales_l4']))
    logger.info('Eval Nature Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
                eval_metric["nat"]['eval_acc'],eval_metric["nat"]['eval_loss']))     
    logger.info('Eval O.O.D. Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}\nendiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
            eval_metric["ood"]['eval_acc'],eval_metric["ood"]['eval_loss'],
            eval_metric["ood_diff"]['eval_acc'],eval_metric["ood_diff"]['eval_loss']))
    
    # save tensorboard
    writer_train.add_scalar('train/lossDiff', train_metric['train_loss_nll'], epoch)
    writer_train.add_scalar('train/lossC', train_metric['train_loss_cla'], epoch)
    writer_train.add_scalar('train/acc', train_metric['train_acc'], epoch)

    writer_train.add_scalar('scales/layer1', train_metric['scales_l1'], epoch)
    writer_train.add_scalar('scales/layer2', train_metric['scales_l2'], epoch)
    writer_train.add_scalar('scales/layer3', train_metric['scales_l3'], epoch)
    writer_train.add_scalar('scales/layer4', train_metric['scales_l4'], epoch)
    
    writer_eval.add_scalar('evalnat/loss', eval_metric["nat"]['eval_loss'], epoch)
    writer_eval.add_scalar('evalnat/acc', eval_metric["nat"]['eval_acc'], epoch)
    
    writer_eval_diff.add_scalar('evalood/loss', eval_metric["ood"]['eval_loss'], epoch)
    writer_eval_diff.add_scalar('evalood/acc', eval_metric["ood"]['eval_acc'], epoch)
    writer_eval_diff.add_scalar('evalood/loss', eval_metric["ood_diff"]['eval_loss'], epoch)
    writer_eval_diff.add_scalar('evalood/acc', eval_metric["ood_diff"]['eval_acc'], epoch)

    # save csv
    metric = pd.concat(
        [pd.DataFrame(train_metric,index=[epoch]), 
        pd.DataFrame(eval_metric["nat"],index=[epoch]),
        pd.DataFrame(eval_metric["ood"],index=[epoch]),
        pd.DataFrame(eval_metric["ood_diff"],index=[epoch])
        ], axis=1)
    total_metrics = pd.concat([total_metrics, metric], ignore_index=True)
    total_metrics.to_csv(os.path.join(args.save_dir, 'stats.csv'), index=True)
    return total_metrics

def eval_epoch(epoch):
    if epoch < 100:
        eval_per_epoch = 20
    elif epoch < 150:
        eval_per_epoch = 10
    else:
        eval_per_epoch = 1
    return eval_per_epoch

def vis(x_ori, x_aug, save_path):
    x=torch.cat([x_ori[:8],x_aug[:8]])
    save_image(x.cpu(), os.path.join(save_path,'ood.png'), nrow=8,padding=0, value_range=(0, 1), pad_value=0)
