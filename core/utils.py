import logging
import random
import numpy as np 
import torch 
import datetime
import os
import re

def get_logger(logpath, filepath=os.path.abspath(__file__),  displaying=True, saving=True):
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
    logger.info(filepath)
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
        name = os.path.join(save_dir, 'test-{}-s{}{}.log'.format(task, severity, postfix))
    elif task == 'adv':
        name = os.path.join(save_dir, 'test-{}-{}{}.log'.format(task, threat, postfix))
    else:
        raise
    #logger = get_logger(logpath=os.path.join(save_dir, 'test-{}-s{}{}.log'.format(task, severity, postfix)), filepath=os.path.abspath(__file__))
    return name #logger


def get_desc(args):
    if 'ladiff' in args.protocol:
        desc = "{}_{}-{}_ntr{}_(C{}lr{}{}-D{}lr{})_e{}_b{}_aug-{}_atk-{}".format(
                args.backbone, args.protocol, 
                args.desc, args.npc_train,  
                args.optimizerC, args.lrC, args.scheduler,
                args.optimizerDiff, args.lrDiff, 
                args.epoch, args.batch_size, args.aug_train, args.atk_train)
    elif args.protocol == 'standard'  or 'fixdiff' in args.protocol :
        desc = "{}_{}-{}_ntr{}_C{}lr{}{}_e{}_b{}_aug-{}_atk-{}".format(
                args.backbone, args.protocol, 
                args.desc, args.npc_train, 
                args.optimizerC, args.lrC, args.scheduler,
                args.epoch, args.batch_size, args.aug_train, args.atk_train)
    else:
        raise
    
    return desc



def set_seed(seed=2333):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
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
            #if epoch > 100:
            save_model(model, optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, save_path=save_path)



def save_model(model, optimizerC, scheduler=None, optimizerDiff=None,  save_path=None):
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



