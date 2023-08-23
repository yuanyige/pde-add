import pandas as pd 
from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.context import ctx_noparamgrad_and_eval

nll_loss = nn.GaussianNLLLoss()

#augmentor_ood = DataAugmentor('rotation-20')

def get_ratio(mu_aug_ood, mu, sigma, f):
    ratio1 = ((mu_aug_ood-mu).abs()/(sigma+1e-8)).mean().item()
    distance = (mu_aug_ood-mu).abs().mean().item()
    sigma = sigma.mean().item()
    ratio2 = distance/sigma
    f.write("{},{},{},{}\n".format(distance,sigma,ratio2,ratio1))

def train_pdeadd(dataloader_train, dataloader_train_diff, model,
                 optimizerDiff, optimizerC, label_smooth=0.1, attacker=None,
                 device=None, visualize=False, epoch=None, save_path=None, use_gmm=True):
        
    print('use_gmm',use_gmm)
    
    metrics = pd.DataFrame()
    batch_index = 0
    model.train()
    

    for (x, y), (x_ood, y_ood) in tqdm(zip(dataloader_train,dataloader_train_diff)):
        
        if_visualize = (True*visualize) if batch_index==0 else (False*visualize)
        batch_metric = defaultdict(float)
        x, y = x.to(device), y.to(device)
        x_ood, y_ood = x_ood.to(device), y_ood.to(device)

        if (y - y_ood).sum().cpu().detach().item():
            raise
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                x_adv, _ = attacker.perturb(x, y,  visualize=if_visualize) 
        
        _ = model(x, use_diffusion = True)
        mus = model.mus
        sigmas = model.sigmas

        _ = model(x_ood, use_diffusion = False)
        mus_aug = model.mus
        
        # x_aug_ood = augmentor_ood.apply(x,  visualize=False)
        # _ = model(x_aug_ood, use_diffusion = False)
        # mus_aug_ood = model.mus

        # f1 =  open(os.path.join(save_path,'Q2_1.csv'), 'a')
        # f2 =  open(os.path.join(save_path,'Q2_2.csv'), 'a')
        # f3 =  open(os.path.join(save_path,'Q2_3.csv'), 'a')
        # f4 =  open(os.path.join(save_path,'Q2_4.csv'), 'a')

        # get_ratio(mus_aug_ood[0], mus[0], sigmas[0], f1)
        # get_ratio(mus_aug_ood[1], mus[1], sigmas[1], f2)
        # get_ratio(mus_aug_ood[2], mus[2], sigmas[2], f3)
        # get_ratio(mus_aug_ood[3], mus[3], sigmas[3], f4)

        # f1.close()
        # f2.close()
        # f3.close()
        # f4.close()

        lossDiff = 0
        for mu_aug, mu, sigma in zip(mus_aug, mus, sigmas):
            lossDiff += nll_loss(mu_aug.view(x.shape[0],-1), mu.view(x.shape[0],-1), sigma.view(x.shape[0],-1))
        lossDiff = lossDiff/len(mus)
        
        optimizerDiff.zero_grad()
        lossDiff.backward()
        optimizerDiff.step()
        
        # out_aug = model(x_aug, use_diffusion = True)
        # out = model(x, use_diffusion = True)
        # optimizerC.zero_grad()
        # lossC =  F.nll_loss(out_aug, y) + F.nll_loss(out, y)
        # lossC.backward()
        # optimizerC.step()

        if use_gmm:
            x_all = torch.cat((x, x_ood), dim=0)
            y_all = torch.cat((y, y), dim=0)
        else:
            x_all = x
            y_all = y
        out = model(x_all, use_diffusion = True)
        optimizerC.zero_grad()
        lossC =  F.cross_entropy(out, y_all, label_smoothing=label_smooth)
        lossC.backward()
        optimizerC.step()

        batch_metric["train_loss_nll"] = lossDiff.data.item()
        batch_metric["train_loss_cla"] = lossC.data.item()
        batch_metric["train_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y_all.data).sum().data.item()
        batch_metric["scales_l1"] =  model.scales[0]
        batch_metric["scales_l2"] =  model.scales[1]
        batch_metric["scales_l3"] =  model.scales[2]
        batch_metric["scales_l4"] =  model.scales[3]
        metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)
        batch_index += 1

        if use_gmm:
            length = 2*len(dataloader_train.dataset)
        else:
            length = len(dataloader_train.dataset)
    
    return dict(metrics.agg({
            "train_loss_nll":"mean",
            "train_loss_cla":"mean",
            "train_acc":lambda x:100*sum(x)/(length),
            "scales_l1":"mean","scales_l2":"mean","scales_l3":"mean","scales_l4":"mean"}))


def train_standard(dataloader_train, model, optimizer, 
                   augmentor=None, attacker=None, device=None, visualize=False, epoch=None):
        
    metrics = pd.DataFrame()
    batch_index = 0
    model.train()

    for x, y in tqdm(dataloader_train):

        batch_metric = defaultdict(float)
        x, y = x.to(device), y.to(device)

        # Update Classifier network  
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                x, _ = attacker.perturb(x, y, visualize=(True*visualize) if batch_index==0 else (False*visualize))
        if augmentor:
            x = augmentor.apply(x, visualize=(True*visualize) if batch_index==0 else (False*visualize))
        
        out = model(x)
        optimizer.zero_grad()
        loss =  F.cross_entropy(out, y) 
        loss.backward()
        optimizer.step()

        batch_metric["train_loss_cla"] = loss.data.item()
        batch_metric["train_loss_nll"] = 0
        batch_metric["train_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
        batch_metric["scales_l1"] =  0
        batch_metric["scales_l2"] =  0
        batch_metric["scales_l3"] =  0
        batch_metric["scales_l4"] =  0
        metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)
        batch_index += 1
    
    return dict(metrics.agg({
            "train_loss_nll":"mean",
            "train_loss_cla":"mean",
            "train_acc":lambda x:100*sum(x)/(len(dataloader_train.dataset)),
            "scales_l1":"mean","scales_l2":"mean","scales_l3":"mean","scales_l4":"mean"}))

