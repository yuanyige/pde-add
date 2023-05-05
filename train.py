import os
import time
import json

import pandas as pd 
import torch
from torch.utils.tensorboard import SummaryWriter

from core.models import create_model
from core.trainfn import train_standard, train_ladiff
from core.testfn import test
from core.parse import parser_train
from core.data import DataAugmentor, load_data, load_cifar_c
from core.utils import BestSaver, get_logger, set_seed, format_time, save_model, get_desc
from core.scheduler import WarmUpLR, get_scheduler
from core.attacks import create_attack

def run_ladiff(model):  

    # writer
    writer = SummaryWriter(os.path.join(args.save_dir), comment='train', filename_suffix='train')
    writer_wodiff = SummaryWriter(os.path.join(args.save_dir), comment='test_wodiff', filename_suffix='test_wodiff')
    writer_endiff = SummaryWriter(os.path.join(args.save_dir), comment='test_endiff', filename_suffix='test_endiff')

    # start training
    total_metrics = pd.DataFrame()
    wodiff_saver = BestSaver()
    endiff_saver = BestSaver()

    for epoch in range(start_epoch, args.epoch + start_epoch):

        start = time.time()
        
        # train
        train_metric = train_ladiff(args.protocol)(dataloader_train, model, optimizerDiff, optimizerC, 
                        augmentor=augmentor, attacker=attack_train, device=device, visualize=True if epoch==start_epoch else False, epoch=epoch)

        if (args.scheduler != 'none') and (epoch > args.warm):
            scheduler.step()  
        if (args.warm) and (epoch <= args.warm):
            warmup_scheduler.step()

        # test for nat
        eval_nat_wodiff_metric = test(dataloader_test, model, use_diffusion=False, device=device)
        # test for ood
        if epoch < 100:
            eval_per_epoch = 20
        elif epoch < 160:
            eval_per_epoch = 10
        else:
            eval_per_epoch = 1
            
        if (epoch == start_epoch) or (epoch % eval_per_epoch == 0):
            eval_ood_wodiff_metric = test(dataloader_test_ood, model, use_diffusion=False, device=device)
            eval_ood_endiff_metric = test(dataloader_test_ood, model, use_diffusion=True, device=device)
        
        # test for adv
        # eval_ood_wodiff_metric = test(dataloader_test, model, use_diffusion=False, attacker=attack_eval, device=device)
        #eval_ood_endiff_metric = test_ensemble(dataloader_test, model, ensemble_iter=args.ensemble_iter_eval, attacker=attack_eval, device=device)
        
        # save tensorboard
        writer.add_scalar('train/lossDiff', train_metric['train_loss_nll'], epoch)
        writer.add_scalar('train/lossC', train_metric['train_loss_cla'], epoch)
        writer.add_scalar('train/acc', train_metric['train_acc'], epoch)

        writer.add_scalar('scales/layer1', train_metric['scales1'], epoch)
        writer.add_scalar('scales/layer2', train_metric['scales2'], epoch)
        writer.add_scalar('scales/layer3', train_metric['scales3'], epoch)
        writer.add_scalar('scales/layer4', train_metric['scales4'], epoch)
        
        writer_wodiff.add_scalar('evalnat/loss', eval_nat_wodiff_metric['eval_loss'], epoch)
        writer_wodiff.add_scalar('evalnat/acc', eval_nat_wodiff_metric['eval_acc'], epoch)
        
        writer_wodiff.add_scalar('evalood/loss', eval_ood_wodiff_metric['eval_loss'], epoch)
        writer_wodiff.add_scalar('evalood/acc', eval_ood_wodiff_metric['eval_acc'], epoch)
        writer_endiff.add_scalar('evalood/loss', eval_ood_endiff_metric['eval_loss'], epoch)
        writer_endiff.add_scalar('evalood/acc', eval_ood_endiff_metric['eval_acc'], epoch)

        # save csv
        metric = pd.concat(
            [pd.DataFrame(train_metric,index=[epoch]), 
            pd.DataFrame(eval_nat_wodiff_metric,index=[epoch]),
            pd.DataFrame(eval_ood_wodiff_metric,index=[epoch]),
            pd.DataFrame(eval_ood_endiff_metric,index=[epoch])
            ], axis=1)
        total_metrics = pd.concat([total_metrics, metric], ignore_index=True)
        total_metrics.to_csv(os.path.join(args.save_dir, 'stats.csv'), index=True)
        
        # verbose
        logger.info('\n[Epoch {}] - Time taken: {}'.format(epoch, format_time(time.time()-start)))
        logger.info('Train\t Acc: {:.2f}%, NLLLoss: {:.2f}, ClassLoss: {:.2f}'.format(
                    train_metric['train_acc'],train_metric['train_loss_nll'],train_metric['train_loss_cla']))
        logger.info('Train\t Scale1: {:.2f}, Scale2: {:.2f}, Scale3: {:.2f}, Scale4: {:.2f}'.format(
                    train_metric['scales1'],train_metric['scales2'],train_metric['scales3'],train_metric['scales4']))
        logger.info('Eval Nature Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_nat_wodiff_metric['eval_acc'],eval_nat_wodiff_metric['eval_loss']))        
        logger.info('Eval O.O.D. Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}\nendiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_ood_wodiff_metric['eval_acc'],eval_ood_wodiff_metric['eval_loss'],
                    eval_ood_endiff_metric['eval_acc'],eval_ood_endiff_metric['eval_loss']))
        # logger.info('Eval O.O.D. Samples\nwodiff\t Acc: {:.2f}%, Loss: {:.2f}'.format(
        #     eval_ood_wodiff_metric['eval_acc'],eval_ood_wodiff_metric['eval_loss']))
        

        # save model
        wodiff_saver.apply(eval_ood_wodiff_metric['eval_acc'], epoch, 
                           model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
                           save_path=os.path.join(args.save_dir,'model-best-wodiff.pt'))
        endiff_saver.apply(eval_ood_endiff_metric['eval_acc'], epoch, 
                           model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff,
                           save_path=os.path.join(args.save_dir,'model-best-endiff.pt'))
        if (epoch!=0) and (epoch % args.save_freq==0):
            save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                       save_path=os.path.join(args.save_dir,'model-e{}.pt'.format(epoch)))
        save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, optimizerDiff=optimizerDiff, 
                   save_path=os.path.join(args.save_dir,'model-last.pt'))

    logger.info("\n[Final]\t Best wo-diff acc: {:.2f}% in Epoch: {}".format(wodiff_saver.best_acc, wodiff_saver.best_epoch))
    logger.info("\n[Final]\t Best en-diff acc: {:.2f}% in Epoch: {}".format(endiff_saver.best_acc, endiff_saver.best_epoch))




def run_standard(model):

    writer = SummaryWriter(args.save_dir)

    # start training
    total_metrics = pd.DataFrame()
    saver = BestSaver()
    
    for epoch in range(start_epoch, args.epoch + start_epoch):

        start = time.time()
        
        # train
        train_metric = train_standard(dataloader_train, model, optimizerC, 
                                      augmentor=augmentor, attacker=attack_train, 
                                      device=device, visualize=True if epoch==start_epoch else False, epoch=epoch)

        if (args.scheduler != 'none') and (epoch > args.warm):
            scheduler.step()  
        if (args.warm) and (epoch <= args.warm):
            warmup_scheduler.step()
        
        
        # test for nat
        eval_nat_metric = test(dataloader_test, model, use_diffusion=False, device=device)
        
        # test for ood
        eval_ood_metric = test(dataloader_test_ood, model, use_diffusion=False, device=device)

        # test for adv
        #eval_ood_metric = test_ensemble(dataloader_test, model, ensemble_iter=args.ensemble_iter_eval, attacker=attack_eval, device=device)


        # save tensorboard
        writer.add_scalar('train/lossC', train_metric['train_loss'], epoch)
        writer.add_scalar('train/acc', train_metric['train_acc'], epoch)
        writer.add_scalar('evalnat/loss', eval_nat_metric['eval_loss'], epoch)
        writer.add_scalar('evalnat/acc', eval_nat_metric['eval_acc'], epoch)
        writer.add_scalar('evalood/loss', eval_ood_metric['eval_loss'], epoch)
        writer.add_scalar('evalood/acc', eval_ood_metric['eval_acc'], epoch)

        # save csv
        metric = pd.concat([pd.DataFrame(train_metric,index=[epoch]), 
                            pd.DataFrame(eval_nat_metric,index=[epoch]),
                            pd.DataFrame(eval_ood_metric,index=[epoch])], axis=1)
        total_metrics = pd.concat([total_metrics, metric], ignore_index=True)
        total_metrics.to_csv(os.path.join(args.save_dir, 'stats.csv'), index=True)

        # verbose
        logger.info('\n[Epoch {}] - Time taken: {}'.format(epoch, format_time(time.time()-start)))
        logger.info('Train\tAcc: {:.2f}%, ClassLoss: {:.2f}'.format(
                    train_metric['train_acc'],train_metric['train_loss']))
        logger.info('Eval Nature Samples\n\tAcc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_nat_metric['eval_acc'],eval_nat_metric['eval_loss']))
        logger.info('Eval O.O.D. Samples\n\tAcc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_ood_metric['eval_acc'],eval_ood_metric['eval_loss']))

        # save model
        saver.apply(eval_ood_metric["eval_acc"], epoch, model=model, optimizerC=optimizerC, scheduler=scheduler,
                    save_path=os.path.join(args.save_dir,'model-best.pt'))
        if epoch % args.save_freq==0:
            save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, save_path=os.path.join(args.save_dir,'model-e{}.pt'.format(epoch)))
        save_model(model=model, optimizerC=optimizerC, scheduler=scheduler, save_path=os.path.join(args.save_dir,'model-last.pt'))

    logger.info("\n[Final]\t Best acc: {:.2f}% in Epoch: {}".format(saver.best_acc, saver.best_epoch))





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
logger = get_logger(logpath=os.path.join(args.save_dir, 'verbose.log'), filepath=os.path.abspath(__file__)) # wandb.init(project="test", config = args, name=args.desc)

# save args
with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# dataloaders
dataloader_train, dataloader_test = load_data(args)
dataloader_test_ood = load_cifar_c(args.data, args.data_dir, args.batch_size_validation, dnum=60, severity=args.severity_eval, norm=args.norm)
logger.info('Using dataset: {}'.format(args.data))
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


# augmentor
if args.aug_train == 'none':
    augmentor=None
else:
    augmentor = DataAugmentor(args.aug_train, args.save_dir)
logger.info('using training augmentors: {}'.format(args.aug_train))


# attackers
attack_train = create_attack(model, attack_type=args.atk_train, attack_eps=args.attack_eps, attack_iter=args.attack_iter, attack_step=args.attack_step, rand_init_type='uniform', save_path= args.save_dir)
if args.atk_train =='none':
    attack_eval = create_attack(model, attack_type=args.atk_eval, attack_eps=args.attack_eps, attack_iter=2*args.attack_iter, attack_step=args.attack_step)
else:
    attack_eval = create_attack(model, attack_type=args.atk_train, attack_eps=args.attack_eps, attack_iter=2*args.attack_iter, attack_step=args.attack_step)
logger.info('using training attacker: {}'.format(args.atk_train))
logger.info('using evaluating attacker: {}'.format(args.atk_eval))
        

# optimizers
optimizerC = torch.optim.SGD(model.parameters(), lr=args.lrC, momentum=0.9, weight_decay=args.weight_decay)
if 'ladiff' in args.protocol:
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
    

# main train
if 'ladiff' in args.protocol:
    run_ladiff(model = model)
elif 'fixdiff' in args.protocol:
    run_standard(model = model)
elif args.protocol == 'standard':
    run_standard(model = model)

