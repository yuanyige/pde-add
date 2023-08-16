import os
import time
import json
from collections import defaultdict

import pandas as pd 
import torch
from torch.utils.tensorboard import SummaryWriter

from core.models import create_model
from core.trainfn import train_standard, train_pdeadd
from core.testfn import test
from core.parse import parser_train
from core.data import load_dataloader, load_corr_dataloader
from core.scheduler import WarmUpLR, get_scheduler
from core.utils import BestSaver, get_logger, set_seed, get_desc, eval_epoch, verbose_and_save

# load args
args = parser_train()
set_seed(args.seed)

if args.resume_file:
    resume_file = args.resume_file
    resume_epoch = args.epoch
    path = "/".join(args.resume_file.split('/')[:-2])
    with open(os.path.join(path, 'train', 'args.txt'), 'r') as f:
        old = json.load(f)
        args.__dict__ = dict(vars(args), **old)
    args.save_dir = os.path.join(path, 'resume')
    args.resume_file = resume_file
    args.epoch = resume_epoch
    args.scheduler ='none'
else:
    args.desc = get_desc(args)
    args.save_dir = os.path.join(args.save_dir, args.desc, 'train')

# set logs
os.makedirs(args.save_dir, exist_ok=True)
logger = get_logger(logpath=os.path.join(args.save_dir, 'verbose.log'))

# save args
with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# dataloaders
dataloader_train, dataloader_train_diff, dataloader_test = load_dataloader(args)
if args.data == 'tin200':
    dataloader_test_ood = dataloader_test
else:
    dataloader_test_ood = load_corr_dataloader(args.data, args.data_dir, args.batch_size_validation, dnum=60, severity=args.severity_eval, norm=args.norm)
    
logger.info('Using train dataset: {} with augment: {}'.format(args.data, args.aug_train))
logger.info('Using train diffusion guidance dataset: {} with augment: {}'.format(args.data_diff, args.aug_train_diff))
logger.info('Using test dataset: {} with its corrs'.format(args.data))

if args.data in ['tinyin200']:
    logger.info('Train data shape: {}, label shape: {}'.format(
        len(dataloader_train.dataset.samples), len(dataloader_train.dataset.targets)))
    logger.info('Test data shape: {}, label shape: {}'.format(
        len(dataloader_test.dataset.samples), len(dataloader_test.dataset.targets)))
    logger.info('Test ood data shape: {}, label shape: {}'.format(
        len(dataloader_test_ood.dataset.samples), len(dataloader_test_ood.dataset.targets)))
else:
    logger.info('Train data shape: {}, label shape: {}'.format(
        dataloader_train.dataset.data.shape, len(dataloader_train.dataset.targets)))
    logger.info('Test data shape: {}, label shape: {}'.format(
        dataloader_test.dataset.data.shape, len(dataloader_test.dataset.targets)))
    logger.info('Test ood data shape: {}, label shape: {}'.format(
        dataloader_test_ood.dataset.data.shape, len(dataloader_test_ood.dataset.targets)))  

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
logger.info('using device: {}'.format(device))

# create model
model = create_model(args.data, args.backbone, args.protocol)
model = model.to(device)
logger.info("using model: {}".format(args.backbone))
logger.info("using protocol: {}".format(args.protocol))

# attackers
attack_train = None
attack_eval = None
logger.info('using training attacker: {}'.format(args.atk_train))
logger.info('using evaluating attacker: {}'.format(args.atk_eval))

# optimizers
optimizerC = torch.optim.SGD(model.parameters(), lr=args.lrC, momentum=0.9, weight_decay=args.weight_decay)
optimizerDiff = None
if args.protocol == 'pdeadd':
    diffusion_params = []
    for name, param in model.named_parameters():
        if 'diff' in name:
            diffusion_params.append(param)
    optimizerDiff = torch.optim.Adam(diffusion_params,lr=args.lrDiff)

# schedulers
scheduler = get_scheduler(args, opt=optimizerC)
if args.warm:
    iter_per_epoch = len(dataloader_train)
    warmup_scheduler = WarmUpLR(optimizerC, iter_per_epoch * args.warm)
logger.info('using scheduler: {}, warmup {}'.format(args.scheduler, args.warm))

# resume
start_epoch=1
if args.resume_file:
    checkpoint = torch.load(args.resume_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizerC.load_state_dict(checkpoint['optimizerC_state_dict'])
    if 'ladiff' in args.protocol:
        optimizerDiff.load_state_dict(checkpoint['optimizerDiff_state_dict'])
    last_cla_lr = checkpoint['scheduler']["_last_lr"][0]
    start_epoch = checkpoint['scheduler']["last_epoch"]
    for param_group in optimizerC.param_groups:
        param_group["lr"] = last_cla_lr
    del checkpoint

# writer
writer_train = SummaryWriter(os.path.join(args.save_dir), comment='train', filename_suffix='train')
writer_eval = SummaryWriter(os.path.join(args.save_dir), comment='eval', filename_suffix='eval')
writer_eval_diff = SummaryWriter(os.path.join(args.save_dir), comment='eval_diff', filename_suffix='eval_diff')

# start training
total_metrics = pd.DataFrame()
saver, diff_saver = BestSaver(), BestSaver()
eval_metric, eval_metric["ood"],eval_metric["ood_diff"] = defaultdict(float),defaultdict(float),defaultdict(float)

for epoch in range(start_epoch, args.epoch + start_epoch):
    start = time.time()
    
    if args.protocol == 'pdeadd':
        train_metric = train_pdeadd(dataloader_train,  dataloader_train_diff, model, optimizerDiff, optimizerC, 
                                attacker=attack_train, device=device, visualize=True if epoch==start_epoch else False, epoch=epoch)
    else:
        train_metric = train_standard(dataloader_train, model, optimizerC, attacker=attack_train, 
                                    device=device, visualize=True if epoch==start_epoch else False, epoch=epoch)
    
    if (args.scheduler != 'none') and (epoch > args.warm):
        scheduler.step()  
    if (args.warm) and (epoch <= args.warm):
        warmup_scheduler.step()

    # test for nat
    eval_metric["nat"] = test(dataloader_test, model, use_diffusion=False, device=device)
    # test for ood
    eval_per_epoch = eval_epoch(epoch)
    if (epoch == start_epoch) or (epoch % eval_per_epoch == 0):
        eval_metric["ood"] = test(dataloader_test_ood, model, use_diffusion=False, device=device)
        eval_metric["ood_diff"] = test(dataloader_test_ood, model, use_diffusion=True, device=device)  

    total_metrics = verbose_and_save(logger, epoch, start, 
                        train_metric, eval_metric, 
                        writer_train, writer_eval, writer_eval_diff, 
                        total_metrics, args)

    # save model
    saver.apply(eval_metric["ood"]['eval_acc'], epoch, 
                        model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
                        save_path=os.path.join(args.save_dir,'model-best-wodiff.pt'))
    diff_saver.apply(eval_metric["ood_diff"]['eval_acc'], epoch, 
                        model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
                        save_path=os.path.join(args.save_dir,'model-best-endiff.pt'))
    if (epoch!=0) and (epoch % args.save_freq==0):
        saver.save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                    save_path=os.path.join(args.save_dir,'model-e{}.pt'.format(epoch)))
    saver.save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                save_path=os.path.join(args.save_dir,'model-last.pt'))

logger.info("\n[Final]\t Best wo-diff acc: {:.2f}% in Epoch: {}".format(saver.best_acc, saver.best_epoch))
logger.info("\n[Final]\t Best en-diff acc: {:.2f}% in Epoch: {}".format(diff_saver.best_acc, diff_saver.best_epoch))
