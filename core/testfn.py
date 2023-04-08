import pandas as pd 
import torch
import torch.nn.functional as F
from collections import defaultdict
from core.context import ctx_noparamgrad_and_eval
from core.data import load_data
from autoattack import AutoAttack
import numpy as np
import math
from core.utils import set_seed

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

            #batch_metric["eval_loss"] = F.cross_entropy(out, y).data.item()
            #batch_metric["eval_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
            batch_metric["eval_loss"] = F.nll_loss(out, y).data.item() 
            batch_metric["eval_acc"] = (out.data.argmax(dim=1) == y.data).sum().data.item()

            metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

    return dict(metrics.agg({
            "eval_loss":"mean",
            "eval_acc":lambda x: 100*sum(x)/len(dataloader_test.dataset)}))


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



def final_corr_eval(x_corrs, y_corrs, model, use_diffusion, corruptions, logger, n_runs=2):
    model.eval()
    clean_acc=np.zeros((1, n_runs))
    for k in range(n_runs):
        clean_acc[0, k] = clean_accuracy(model, use_diffusion, x_corrs[0].to(list(model.parameters())[0].device), y_corrs[0].to(list(model.parameters())[0].device))
    logger.info("Clean accuracy: {:.2%}+-{:.2%} ".format(clean_acc.mean(),clean_acc.std()))

    res = np.zeros((5, 15, n_runs))
    for i in range(1, 6):
        for j, c in enumerate(corruptions):
            for k in range(n_runs):
                res[i-1, j, k] = clean_accuracy(model, use_diffusion, x_corrs[i][j].to(list(model.parameters())[0].device), y_corrs[i][j].to(list(model.parameters())[0].device))
                print(f"{c} {i} {res[i-1, j, k]}")
    
    mean_acc = np.mean(res, axis=2)
    std_acc = np.std(res, axis=2)
    
    frame = pd.DataFrame({i+1: [f"{mean_acc[i, j]:.2%}+-{std_acc[i, j]:.2%}" for j in range(15)] for i in range(0, 5)}, index=corruptions)
    frame.loc['average'] = {i+1: f"{np.mean(mean_acc, axis=1)[i]:.2%}+-{np.mean(std_acc, axis=1)[i]:.2%}" for i in range(0, 5)}
    frame['avg'] = [f"{np.mean(mean_acc[:, i]):.2%}+-{np.mean(std_acc[:, i]):.2%}" for i in range(15)] + [f"{np.mean(mean_acc):.2%}+-{np.mean(std_acc):.2%}"]
    logger.info(frame)
    return frame



# def final_corr_eval(x_corrs, y_corrs, model, use_diffusion, corruptions, logger):
#     model.eval()
#     res = np.zeros((5, 15))
#     for i in range(1, 6):
#         for j, c in enumerate(corruptions):
#             res[i-1, j] = clean_accuracy(model, use_diffusion, x_corrs[i][j].to(list(model.parameters())[0].device), y_corrs[i][j].to(list(model.parameters())[0].device))
#             print(c, i, res[i-1, j])

#     frame = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=corruptions)
#     frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
#     frame['avg'] = frame[list(range(1,6))].mean(axis=1)
#     logger.info(frame)


# def test_ensemble(dataloader_test, model, ensemble_iter=10, augmentor=None, attacker=None, device=None):

#     metrics = pd.DataFrame()
#     model.eval()
    
#     for x, y in dataloader_test:
#         batch_metric = defaultdict(float)
#         x, y = x.to(device), y.to(device)

#         if attacker:
#             with ctx_noparamgrad_and_eval(model):
#                 x, _ = attacker.perturb(x, y)

#         with torch.no_grad():
#             if augmentor:
#                 x = augmentor.apply(x, True)
            
#             proba = 0 
#             for k in range(ensemble_iter):  
#                 out = model(x, use_diffusion = True)
#                 p = F.softmax(out, dim=1)
#                 proba = proba + p
#             out = ((proba/ensemble_iter)+1e-20).log() # next nll
            
#             batch_metric["eval_loss"] = F.nll_loss(out, y).data.item()
#             batch_metric["eval_acc"] = (out.data.argmax(dim=1) == y.data).sum().data.item()

#             metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

#     return dict(metrics.agg({
#             "eval_loss":"mean",
#             "eval_acc":lambda x: 100*sum(x)/len(dataloader_test.dataset)}))




# def final_test(protocol, loader, model, task,  logger, augmentor=None, attacker=None, device=None):
#     if ('ladiff' in protocol):
#         ood_wodiff_metric = test(loader, model,  use_diffusion=False, augmentor=augmentor, attacker=attacker, device=device)
#         ood_endiff_metric = test_ensemble(loader, model, ensemble_iter=10, augmentor=augmentor, attacker=attacker, device=device)
#         logger.info('{}, {:.2f}%, {:.2f}, {:.2f}%, {:.2f}'.format(
#                     task, ood_wodiff_metric['eval_acc'],ood_wodiff_metric['eval_loss'],
#                     ood_endiff_metric['eval_acc'],ood_endiff_metric['eval_loss']))
#     elif (protocol == 'standard') or ('fixdiff' in protocol):
#         ood_metric = test(loader, model, augmentor=augmentor, attacker=attacker, device=device)
#         ood_endiff_metric = test_ensemble(loader, model, ensemble_iter=10, augmentor=augmentor, attacker=attacker, device=device)
#         logger.info('{}, {:.2f}%, {:.2f}, {:.2f}%, {:.2f}'.format(
#                     task, ood_metric['eval_acc'],ood_metric['eval_loss'],
#                     ood_endiff_metric['eval_acc'],ood_endiff_metric['eval_loss']))
#     else:
#         raise


# def run_final_test_selfmade(model, args_test, device): 
    
#     save_path = os.path.join(args_test.ckpt_path, 'test')
#     logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

#     _, loader_test = load_data(args_test)
#     #final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)

#     augmentor = DataAugmentor('color-0-0-0-0.5', save_path=save_path, name='-hue')
#     final_test(args_test.protocol, loader_test, model, task='hue', logger=logger, augmentor=augmentor, device=device)

#     augmentor = DataAugmentor('rotation-20', save_path=save_path) #45
#     final_test(args_test.protocol, loader_test, model, task='rotation', logger=logger, augmentor=augmentor, device=device)

#     augmentor = DataAugmentor('color-1-0-0-0', save_path=save_path, name='-brightness')
#     final_test(args_test.protocol, loader_test, model, task='brightness', logger=logger, augmentor=augmentor, device=device)

#     augmentor = DataAugmentor('color-0-1-0-0', save_path=save_path, name='-contrast')
#     final_test(args_test.protocol, loader_test, model, task='contrast', logger=logger, augmentor=augmentor, device=device)

#     augmentor = DataAugmentor('color-0-0-2-0', save_path=save_path, name="-saturation")
#     final_test(args_test.protocol, loader_test, model, task='saturation', logger=logger, augmentor=augmentor, device=device)

#     augmentor = DataAugmentor('gaublur-3-2',save_path=save_path)
#     final_test(args_test.protocol, loader_test, model, task='gaublur', logger=logger, augmentor=augmentor, device=device)

#     logger.info("Evalution done.")




# def run_final_test_cifarc(model, args_test, device): 
#     set_seed(args_test.seed)
#     logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

#     _, loader_test = load_data(args_test)
#     final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)

#     cname_list = ['fog','snow','frost',
#                 'zoom_blur','defocus_blur','glass_blur','gaussian_blur','motion_blur',
#                 'speckle_noise','shot_noise','impulse_noise','gaussian_noise',
#                 'jpeg_compression','pixelate','spatter',
#                 'elastic_transform','brightness','saturate','contrast']
    
#     for ci, cname in enumerate(cname_list):
#         # Load data
#         loader_c = load_cifar_c(args_test.data, args_test.data_dir, args_test.batch_size_validation, cname, severity=args_test.severity, norm=args_test.norm)
#         final_test(args_test.protocol, loader_c, model, task=cname,  logger=logger, device=device)

#     logger.info("Evalution done.")



# def run_final_test_select_cifarc(model, args_test, device): 
#     set_seed(args_test.seed)
#     logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, severity=args_test.severity))

#     _, loader_test = load_data(args_test)
#     final_test(args_test.protocol, loader_test, model, task='iid',  logger=logger, device=device)
    
#     #cname_list = ['gaussian_blur','brightness','contrast']
#     cname_list = ['shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
#                'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
#                'jpeg_compression', 'elastic_transform']
    
#     for ci, cname in enumerate(cname_list):
#         # Load data
#         loader_c = load_cifar_c(args_test.data, args_test.data_dir, args_test.batch_size_validation, cname, severity=args_test.severity, norm=args_test.norm)
#         final_test(args_test.protocol, loader_c, model, task=cname,  logger=logger, device=device)

#     logger.info("Evalution done.")



def run_final_test_autoattack(model, args_test, logger, device):
    #set_seed(args_test.seed)
    #logger_path = get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task, threat=args_test.threat)

    _, loader_test = load_data(args_test)

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
