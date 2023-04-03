import pandas as pd 
import torch
import torch.nn.functional as F
from collections import defaultdict
from core.utils import set_seed, get_logger, get_logger_name
from core.context import ctx_noparamgrad_and_eval
from core.data import DataAugmentor, load_cifar_c, load_data
import os
from autoattack import AutoAttack

def test(dataloader_test, model, use_diffusion=True, augmentor=None, attacker=None, device=None):

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
                x = augmentor.apply(x, True)

            out = model(x, use_diffusion = use_diffusion)

            batch_metric["eval_loss"] = F.cross_entropy(out, y).data.item()
            batch_metric["eval_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()

            metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

    return dict(metrics.agg({
            "eval_loss":"mean",
            "eval_acc":lambda x: 100*sum(x)/len(dataloader_test.dataset)}))



def test_ensemble(dataloader_test, model, ensemble_iter=10, augmentor=None, attacker=None, device=None):

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
                x = augmentor.apply(x, True)
            
            proba = 0 
            for k in range(ensemble_iter):  
                out = model(x, use_diffusion = True)
                p = F.softmax(out, dim=1)
                proba = proba + p
            out = ((proba/ensemble_iter)+1e-20).log() # next nll
            
            batch_metric["eval_loss"] = F.nll_loss(out, y).data.item()
            batch_metric["eval_acc"] = (out.data.argmax(dim=1) == y.data).sum().data.item()

            metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

    return dict(metrics.agg({
            "eval_loss":"mean",
            "eval_acc":lambda x: 100*sum(x)/len(dataloader_test.dataset)}))


def final_test(protocol, loader, model, task,  logger, augmentor=None, attacker=None, device=None):
    if ('ladiff' in protocol):
        ood_wodiff_metric = test(loader, model,  use_diffusion=False, augmentor=augmentor, attacker=attacker, device=device)
        ood_endiff_metric = test_ensemble(loader, model, ensemble_iter=10, augmentor=augmentor, attacker=attacker, device=device)
        logger.info('{}, {:.2f}%, {:.2f}, {:.2f}%, {:.2f}'.format(
                    task, ood_wodiff_metric['eval_acc'],ood_wodiff_metric['eval_loss'],
                    ood_endiff_metric['eval_acc'],ood_endiff_metric['eval_loss']))
    elif (protocol == 'standard') or ('fixdiff' in protocol):
        ood_metric = test(loader, model, augmentor=augmentor, attacker=attacker, device=device)
        ood_endiff_metric = test_ensemble(loader, model, ensemble_iter=10, augmentor=augmentor, attacker=attacker, device=device)
        logger.info('{}, {:.2f}%, {:.2f}, {:.2f}%, {:.2f}'.format(
                    task, ood_metric['eval_acc'],ood_metric['eval_loss'],
                    ood_endiff_metric['eval_acc'],ood_endiff_metric['eval_loss']))
    else:
        raise


def run_final_test_selfmade(model, args_test, device): 
    set_seed(args_test.seed)
    save_path = os.path.join(args_test.ckpt_path, 'test')
    logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

    _, loader_test = load_data(args_test)
    #final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)

    augmentor = DataAugmentor('color-0-0-0-0.5', save_path=save_path, name='-hue')
    final_test(args_test.protocol, loader_test, model, task='hue', logger=logger, augmentor=augmentor, device=device)

    augmentor = DataAugmentor('rotation-20', save_path=save_path) #45
    final_test(args_test.protocol, loader_test, model, task='rotation', logger=logger, augmentor=augmentor, device=device)

    augmentor = DataAugmentor('color-1-0-0-0', save_path=save_path, name='-brightness')
    final_test(args_test.protocol, loader_test, model, task='brightness', logger=logger, augmentor=augmentor, device=device)

    augmentor = DataAugmentor('color-0-1-0-0', save_path=save_path, name='-contrast')
    final_test(args_test.protocol, loader_test, model, task='contrast', logger=logger, augmentor=augmentor, device=device)

    augmentor = DataAugmentor('color-0-0-2-0', save_path=save_path, name="-saturation")
    final_test(args_test.protocol, loader_test, model, task='saturation', logger=logger, augmentor=augmentor, device=device)

    augmentor = DataAugmentor('gaublur-3-2',save_path=save_path)
    final_test(args_test.protocol, loader_test, model, task='gaublur', logger=logger, augmentor=augmentor, device=device)

    logger.info("Evalution done.")




def run_final_test_cifarc(model, args_test, device): 
    set_seed(args_test.seed)
    logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

    _, loader_test = load_data(args_test)
    final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)

    cname_list = ['fog','snow','frost',
                'zoom_blur','defocus_blur','glass_blur','gaussian_blur','motion_blur',
                'speckle_noise','shot_noise','impulse_noise','gaussian_noise',
                'jpeg_compression','pixelate','spatter',
                'elastic_transform','brightness','saturate','contrast']
    
    for ci, cname in enumerate(cname_list):
        # Load data
        loader_c = load_cifar_c(args_test.data, args_test.data_dir, args_test.batch_size_validation, cname, severity=args_test.severity, norm=args_test.norm)
        final_test(args_test.protocol, loader_c, model, task=cname,  logger=logger, device=device)

    logger.info("Evalution done.")



def run_final_test_select_cifarc(model, args_test, device): 
    set_seed(args_test.seed)
    logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

    _, loader_test = load_data(args_test)
    final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)
    
    cname_list = ['gaussian_blur','brightness','contrast']
    
    for ci, cname in enumerate(cname_list):
        # Load data
        loader_c = load_cifar_c(args_test.data, args_test.data_dir, args_test.batch_size_validation, cname, severity=args_test.severity, norm=args_test.norm)
        final_test(args_test.protocol, loader_c, model, task=cname,  logger=logger, device=device)

    logger.info("Evalution done.")



def run_final_test_autoattack(model, args_test, device):
    set_seed(args_test.seed)
    logger_path = get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, threat=args_test.threat)

    _, loader_test = load_data(args_test)

    l = [x for (x, y) in loader_test]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader_test]
    y_test = torch.cat(l, 0)

    if args_test.threat =='linf':
        epsilon = 8 / 255.
    elif args_test.threat =='l2':
        epsilon = 0.5
    adversary = AutoAttack(model, norm=args_test.threat, eps=epsilon, version='standard', log_path=logger_path, seed=args_test.seed)
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
