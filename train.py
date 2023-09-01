import os
import time
import json
from collections import defaultdict

import optuna
import pandas as pd 
import torch
from torch.utils.tensorboard import SummaryWriter

from core.models import create_model
from core.trainfn import train_standard, train_pdeadd
from core.testfn import test, eval_ood
from core.parse import parser_train
from core.data import load_dataloader,load_dg_dataloader, corruption_15, pacs_4
from core.scheduler import WarmUpLR, get_scheduler
from core.utils import BestSaver, get_logger, set_seed, get_desc, eval_epoch, verbose_and_save

# load args
def objective(trial=None): 
    global logger
    args = parser_train()
    set_seed(args.seed)

    if use_optuna:
        args.lrC = trial.suggest_float("lrC", 0.05, 0.1, step=0.01) 
        args.lrDiff = trial.suggest_float("lrDiff", 0.0002, 0.001, step=0.0001) 
        #args.batch_size=trial.suggest_int("bs", 64, 256, step=64)
        args.ls=trial.suggest_float("ls",0.1, 0.2, step=0.01)
        args.save_dir = optuna_save_dir

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
    if not use_optuna:
        logger = get_logger(logpath=os.path.join(args.save_dir, 'verbose.log'))
    else:
        logger.info('\nOptuna number of trial: '+str(trial.number))
        logger.info('Optuna number of trial: '+str(trial.params))

    if args.protocol in ['standard','fixdiff']:
        args.data_diff = None
    elif args.protocol == 'pdeadd':
        if args.data in ['cifar10','cifar100','tin200']:
            args.data_diff = args.data
    else:
        raise

    # save args
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # dataloaders
    if args.data in ['cifar10','cifar100','tin200'] :
        ood_data = corruption_15
    elif 'pacs' in args.data:
        ood_data = pacs_4
        if args.data.split("-")[1] in ood_data:
            ood_data.remove(args.data.split("-")[1])
        if (args.data_diff):
            if (args.data_diff.split("-")[1] in ood_data) and (args.use_gmm):
                ood_data.remove(args.data_diff.split("-")[1])
        

    if ('pacs' in args.data) and (args.data != args.data_diff) and (args.data_diff is not None):
        dataloader_train, dataloader_train_diff, dataloader_test = load_dg_dataloader(args) 
    else:
        dataloader_train, dataloader_train_diff, dataloader_test = load_dataloader(args) 
        

    logger.info('Using train dataset: {} with augment: {}'.format(args.data, args.aug_train))
    if ('pacs' in args.data) or (args.data in ['tin200']):
        logger.info('[+] data shape: {}, label shape: {}'.format(
            len(dataloader_train.dataset.samples), len(dataloader_train.dataset.targets)))
    elif ('cifar' in args.data):
        logger.info('[+] data shape: {}, label shape: {}'.format(
            dataloader_train.dataset.data.shape, len(dataloader_train.dataset.targets)))
    else:
        raise
    
    if args.data_diff:
        logger.info('Using train diffusion guidance dataset: {} with augment: {}'.format(args.data_diff, args.aug_train_diff))
        if ('pacs' in args.data) or (args.data in ['tin200']):
            logger.info('[+] data shape: {}, label shape: {}'.format(
                len(dataloader_train_diff.dataset.samples), len(dataloader_train_diff.dataset.targets)))
        elif ('cifar' in args.data) :
            logger.info('[+] data shape: {}, label shape: {}'.format(
                dataloader_train_diff.dataset.data.shape, len(dataloader_train_diff.dataset.targets)))
        else:
            raise
    else:
        logger.info('Not using train diffusion guidance dataset')

    logger.info('Using test dataset: {}'.format(ood_data))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger.info('using device: {}'.format(device))

    # create model
    model = create_model(args.backbone, args.protocol, num_classes=len(dataloader_train.dataset.classes))
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
        if 'pdeadd' in args.protocol:
            optimizerDiff.load_state_dict(checkpoint['optimizerDiff_state_dict'])
        last_cla_lr = checkpoint['scheduler']["_last_lr"][0]
        start_epoch = checkpoint['scheduler']["last_epoch"]
        for param_group in optimizerC.param_groups:
            param_group["lr"] = last_cla_lr
        del checkpoint

    # writer
    if not use_optuna:
        writer = SummaryWriter(os.path.join(args.save_dir), comment='train', filename_suffix='train')
        #writer_eval = SummaryWriter(os.path.join(args.save_dir), comment='eval', filename_suffix='eval')
        #writer_eval_diff = SummaryWriter(os.path.join(args.save_dir), comment='eval_diff', filename_suffix='eval_diff')

    # start training
    total_metrics = pd.DataFrame()
    saver = BestSaver()
    eval_metric, eval_metric["ood"],eval_metric["nat"] = defaultdict(float),defaultdict(float),defaultdict(float)

    for epoch in range(start_epoch, args.epoch + start_epoch):
        start = time.time()
        
        if args.protocol == 'pdeadd':
            train_metric = train_pdeadd(dataloader_train,  dataloader_train_diff, model, optimizerDiff, optimizerC, label_smooth=args.ls,
                                    attacker=attack_train, device=device, use_gmm=args.use_gmm, save_path=args.save_dir)
        else:
            train_metric = train_standard(dataloader_train, model, optimizerC, attacker=attack_train, 
                                        device=device, visualize=True if epoch==start_epoch else False, epoch=epoch)
        
        if (args.scheduler != 'none') and (epoch > args.warm):
            scheduler.step()  
        if (args.warm) and (epoch <= args.warm):
            warmup_scheduler.step()

        # test for nat
        with torch.no_grad(): 
            eval_per_epoch = eval_epoch(epoch)
            if (epoch == start_epoch) or (epoch % eval_per_epoch == 0):
                eval_metric["ood"] = eval_ood(ood_data=ood_data, args=args, model=model, use_diffusion=False, logger=logger, device=device) 
                if dataloader_test:
                    eval_metric["nat"] = test(dataloader_test, model, use_diffusion=False, device=device)
                
                
            if use_optuna:
                trial.report(eval_metric["ood"]["eval_acc"], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    return accuracy
            else:
                verbose_and_save(logger, epoch, start_epoch, eval_per_epoch, start, train_metric, eval_metric, writer)
        
        # save csv
        metric = pd.concat(
            [pd.DataFrame(train_metric,index=[epoch]), 
            pd.DataFrame(eval_metric["nat"],index=[epoch]),
            pd.DataFrame(eval_metric["ood"],index=[epoch]),
            #pd.DataFrame(eval_metric["ood_diff"],index=[epoch])
            ], axis=1)
        total_metrics = pd.concat([total_metrics, metric], ignore_index=True)
        total_metrics.to_csv(os.path.join(args.save_dir, 'stats.csv'), index=True)

        # save model
        saver.apply(eval_metric["ood"]['eval_acc'], epoch, 
                            model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
                            save_path=os.path.join(args.save_dir,'model-best-wodiff.pt'))
        # diff_saver.apply(eval_metric["ood_diff"]['eval_acc'], epoch, 
        #                     model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
        #                     save_path=os.path.join(args.save_dir,'model-best-endiff.pt'))
        if (epoch!=0) and (epoch % args.save_freq==0):
            saver.save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                        save_path=os.path.join(args.save_dir,'model-e{}.pt'.format(epoch)))
        saver.save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                    save_path=os.path.join(args.save_dir,'model-last.pt'))

    logger.info("[Final]\t Best wo-diff acc: {:.2f}% in Epoch: {}".format(saver.best_acc, saver.best_epoch))
    #logger.info("[Final]\t Best en-diff acc: {:.2f}% in Epoch: {}".format(diff_saver.best_acc, diff_saver.best_epoch))
    
    return eval_metric["ood"]["eval_acc"]

use_optuna = False
optuna_save_dir = '/home/yuanyige/Ladiff_nll/save_optuna_200'
if use_optuna:
    os.makedirs(optuna_save_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(optuna_save_dir, 'verbose.log'))
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('\n\nbest value',study.best_value) 
    print('best param',study.best_params) 
else:
    objective()