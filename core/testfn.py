import math
import numpy as np
import pandas as pd 
from autoattack import AutoAttack

import torch
import torch.nn.functional as F
from collections import defaultdict
from core.context import ctx_noparamgrad_and_eval
from core.data import load_dataloader, load_corr_dataloader
from torchmetrics.functional.classification import multiclass_calibration_error
from torchvision.transforms.functional import convert_image_dtype
import math

def test(dataloader_test, model, use_diffusion=False, augmentor=None, attacker=None, device=None):

    metrics = pd.DataFrame()
    model.eval()
    
    
    for x, y in dataloader_test:
        batch_metric = defaultdict(float)
        x, y = x.to(device), y.to(device)
        
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                x, _ = attacker.perturb(x, y)
        
        with torch.no_grad():
            if augmentor:
                # x_aug = augmentor(convert_image_dtype(x, torch.uint8))
                # x_aug = convert_image_dtype(x_aug, torch.float)
                x_aug = augmentor(x)
            
            if use_diffusion:
                proba = 0 
                for _ in range(10):  
                    out = model(x, use_diffusion=True)
                    proba = proba + out
                out = proba/10
            else:
                out = model(x, use_diffusion = use_diffusion)
            # out_aug = model(x_aug, use_diffusion = use_diffusion)
            # distance = torch.abs(out_aug-out)

            batch_metric["eval_loss"] = F.cross_entropy(out, y).data.item()
            batch_metric["eval_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()

            # batch_metric["eval_acc_aug"] = (torch.softmax(out_aug.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
            # batch_metric["distance"] = distance.data.mean().item()
            # batch_metric["distance_min"] = distance.data.mean().item()
            # batch_metric["distance_max"] = distance.data.mean().item()
            # batch_metric["distance_std"] = distance.data.mean().item()

            metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

    return dict(metrics.agg({
            "eval_loss":"mean",
            "eval_acc":lambda x: 100*sum(x)/len(dataloader_test.dataset),
            # "eval_acc_aug":lambda x: 100*sum(x)/len(dataloader_test.dataset),
            # "distance":"mean",
            # "distance_min":"min",
            # "distance_max":"max",
            # "distance_std":"std"
            }))


def eval_ood(ood_data, args, model, use_diffusion, logger, device):
    if 'pacs' in args.data:
        res = [] 
        for c in range(len(ood_data)):
            dataloader = load_corr_dataloader(args.data, args.data_dir, args.batch_size, cname=ood_data[c])
            dict = test(dataloader, model, use_diffusion =use_diffusion, device=device)
            del dataloader
            res.append(dict["eval_acc"])
            logger.info("{}-{}".format(ood_data[c],dict["eval_acc"]))
        ret = np.array(res).mean()
         
    else:
        res = np.zeros((5, len(ood_data)))
        for c in range(len(ood_data)):
            for s in range(1, 6):
                dataloader = load_corr_dataloader(args.data, args.data_dir, args.batch_size, cname=ood_data[c], severity=s)
                dict = test(dataloader, model, use_diffusion =use_diffusion, device=device)
                del dataloader
                res[s-1, c] = dict["eval_acc"]
        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=ood_data)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
        frame['avg'] = frame[list(range(1, 6))].mean(axis=1)
        logger.info(frame)
        ret = frame["avg"]["average"]
    return {"eval_acc":ret}


def clean_accuracy(model: torch.nn.Module,
                   use_diffusion: bool,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr, use_diffusion=use_diffusion)
            
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def compute_ece (model: torch.nn.Module,
                   use_diffusion: bool,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    ece = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr, use_diffusion=use_diffusion)
            
            ece += multiclass_calibration_error(output, y_curr, num_classes=10, n_bins=15, norm='l1')

    return ece.item() / n_batches


def compute_mce(corruption_accs, baseline_acc):
  """Compute mCE (mean Corruption Error) normalized by Baseline performance."""
  mce = 0.
  for i in range(15):
    ce = (1-corruption_accs[i]) / (1-baseline_acc[i])
    mce += ce 
  return mce / 15


def compute_rmce(nat_acc, corr_accs, baseline_nat_acc, baseline_corr_accs):
  """Compute rmCE (relative mean Corruption Error) normalized by Baseline performance."""
  mce = 0.
  for i in range(15):
    ce = ((1-corr_accs[i])-(1-nat_acc)) / ((1-baseline_corr_accs[i])-(1-baseline_nat_acc))
    mce += ce 
  return mce / 15


def final_corr_eval(x_corrs, y_corrs, model, use_diffusion, corruptions, baseline_accs, logger):
    l = len(corruptions)
    
    model.eval()
    nat_acc = clean_accuracy(model, use_diffusion, x_corrs[0].to(list(model.parameters())[0].device), y_corrs[0].to(list(model.parameters())[0].device))
    logger.info('nat_acc: {}'.format(nat_acc))

    
    res = np.zeros((5, l))
    for i in range(1, 6):
        for j, c in enumerate(corruptions):
            res[i-1, j] = clean_accuracy(model, use_diffusion, x_corrs[i][j].to(list(model.parameters())[0].device), y_corrs[i][j].to(list(model.parameters())[0].device))
            print(c, i, res[i-1, j])

    frame = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=corruptions)
    frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
    frame['avg'] = frame[list(range(1,6))].mean(axis=1)
    logger.info(frame)

    baseline_acc_nat=baseline_accs[0]
    baseline_acc_s0=baseline_accs[1]
    baseline_acc_s5=baseline_accs[2]

    s0 = list(frame['avg'])
    s5 = list(frame[5])

    mce_s0 = compute_mce(s0, baseline_acc_s0)
    logger.info('mce_s0: {}'.format(mce_s0))

    mce_s5 = compute_mce(s5, baseline_acc_s5)
    logger.info('mce_s5: {}'.format(mce_s5))

    rmce_s0 = compute_rmce(nat_acc, s0, baseline_acc_nat, baseline_acc_s0)
    logger.info('rmce_s0: {}'.format(rmce_s0))

    rmce_s5 = compute_rmce(nat_acc, s5, baseline_acc_nat, baseline_acc_s5)
    logger.info('rmce_s5: {}'.format(rmce_s5))


def run_final_test_autoattack(model, args_test, logger, device):
    
    _, loader_test = load_dataloader(args_test)

    l = [x for (x, y) in loader_test]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader_test]
    y_test = torch.cat(l, 0)

    if args_test.threat =='linf':
        epsilon = 8 / 255.
    elif args_test.threat =='l2':
        epsilon = 0.5
    adversary = AutoAttack(model, norm=args_test.threat, eps=epsilon, version='standard', log_path=logger, seed=args_test.seed)
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
