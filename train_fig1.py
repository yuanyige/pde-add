import os
import time
import json

import pandas as pd 
import torch
from torch.utils.tensorboard import SummaryWriter

from core.models import create_model
from core.trainfn import train_standard, train_ladiff
from core.testfn import test_ensemble
from core.parse import parser_train
from core.data import DataAugmentor, load_data, load_cifar10c
from core.utils import BestSaver, get_logger, set_seed, format_time, save_model, get_desc



def run_standard(model):

    writer = SummaryWriter(args.save_dir)

    # models
    model = model.to(device)
    logger.info("using model: {}".format(args.backbone))
    logger.info("using protocol: {}".format(args.protocol))
    logger.info(model)

    # optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lrC, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # start training
    total_metrics = pd.DataFrame()
    saver = BestSaver()
    
    for epoch in range(args.epoch):

        start = time.time()
        
        # train
        train_metric = train_standard(dataloader_train, model, optimizer, 
                                      augmentor=augmentor_train, device=device, 
                                      visualize=True if epoch==0 else False, epoch=epoch)
        scheduler.step()
        
        # test for nat
        eval_nat_metric = test_ensemble(dataloader_test, model, 
                                        ensemble_iter=args.ensemble_iter_eval, 
                                        augmentor=augmentor_eval, device=device)
        
        # test for ood
        eval_ood_metric = test_ensemble(dataloader_test_ood, model, 
                                        ensemble_iter=args.ensemble_iter_eval, 
                                        augmentor=augmentor_eval, device=device)


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

        # save tensorboard
        # wandb.log(train_metric, step=epoch)
        # wandb.log({"epoch": epoch,
        #            "evalnat_acc":eval_nat_metric['eval_acc'],
        #            "evalood_acc":eval_ood_metric['eval_acc'],
        #            "evalnat_loss":eval_nat_metric['eval_loss'],
        #            "evalood_loss":eval_ood_metric['eval_loss']}, step=epoch)

        # verbose
        logger.info('\n[Epoch {}] - Time taken: {}'.format(epoch, format_time(time.time()-start)))
        logger.info('Train\tAcc: {:.2f}%, ClassLoss: {:.2f}'.format(
                    train_metric['train_acc'],train_metric['train_loss']))
        logger.info('Eval Nature Samples\n\tAcc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_nat_metric['eval_acc'],eval_nat_metric['eval_loss']))
        logger.info('Eval O.O.D. Samples\n\tAcc: {:.2f}%, Loss: {:.2f}'.format(
                    eval_ood_metric['eval_acc'],eval_ood_metric['eval_loss']))

        # save model
        saver.apply(eval_ood_metric["eval_acc"], epoch, model=model, optimizerC=optimizer, 
                    save_path=os.path.join(args.save_dir,'model-best.pt'))
        if (epoch!=0) and (epoch % args.save_freq==0):
            save_model(model=model, optimizerC=optimizer, save_path=os.path.join(args.save_dir,'model-e{}.pt'.format(epoch)))
        save_model(model=model, optimizerC=optimizer, save_path=os.path.join(args.save_dir,'model-last.pt'))

    logger.info("\n[Final]\t Best acc: {:.2f}% in Epoch: {}".format(saver.best_acc, saver.best_epoch))






args = parser_train()
set_seed(args.seed)

# logs
args.desc = get_desc(args)
args.save_dir = os.path.join(args.save_dir, args.desc, 'train')
os.makedirs(args.save_dir, exist_ok=True)
logger = get_logger(logpath=os.path.join(args.save_dir, 'verbose.log'), filepath=os.path.abspath(__file__))

#wandb.init(project="test", config = args, name=args.desc)


# save args
with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# dataloaders
dataloader_train, dataloader_test = load_data(args)
dataloader_test_ood = load_cifar10c(args.data, args.data_dir, args.batch_size_validation, dnum=60, severity=args.severity_eval)

logger.info('Using dataset: {}'.format(args.data))
logger.info('Train data shape: {}, label shape: {}'.format(
    dataloader_train.dataset.data.shape, len(dataloader_train.dataset.targets)))
logger.info('Test data shape: {}, label shape: {}'.format(
    dataloader_test.dataset.data.shape, len(dataloader_test.dataset.targets)))
logger.info('Test ood data shape: {}, label shape: {}'.format(
    dataloader_test_ood.dataset.data.shape, len(dataloader_test_ood.dataset.targets)))

# augmentor
eval_aug = "color-1-0-0-0" #contrast
augmentor_train = DataAugmentor(args.aug_train, args.save_dir)
augmentor_eval = DataAugmentor(eval_aug, args.save_dir,name="-eval")
logger.info('using training augmentors: {}'.format(args.aug_train))
logger.info('using testing augmentors: {}'.format(eval_aug))

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
logger.info('using device: {}'.format(device))



run_standard(model = create_model(args.backbone, args.protocol))

#wandb.finish()


