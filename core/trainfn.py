import pandas as pd 
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.context import ctx_noparamgrad_and_eval

nll_loss = nn.GaussianNLLLoss()

def train_ladiff_augdiff(dataloader_train, model,
                 optimizerDiff, optimizerC, augmentor=None, attacker=None,
                 device=None, visualize=False, epoch=None):

    print("ladiff_augdiff..")
        
    metrics = pd.DataFrame()
    batch_index = 0
    model.train()
    

    for x, y in dataloader_train:
        if_visualize = (True*visualize) if batch_index==0 else (False*visualize)
        batch_metric = defaultdict(float)
        x, y = x.to(device), y.to(device)
        
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                x_aug, _ = attacker.perturb(x, y,  visualize=if_visualize)
        else:
            x_aug = augmentor.apply(x,  visualize=if_visualize)

        
        out_fake = model(x, use_diffusion = True)
        mus = model.mus
        sigmas = model.sigmas

        out_aug = model(x_aug, use_diffusion = False)
        mus_aug = model.mus

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

        x_all = torch.cat((x, x_aug), dim=0)
        y_all = torch.cat((y, y), dim=0)
        out = model(x_all, use_diffusion = True)
        optimizerC.zero_grad()
        lossC =  F.cross_entropy(out, y_all, label_smoothing=0.1) #
        lossC.backward()
        optimizerC.step()


        batch_metric["train_loss_nll"] = lossDiff.data.item()
        batch_metric["train_loss_cla"] = lossC.data.item()
        batch_metric["train_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y_all.data).sum().data.item()
        batch_metric["scales1"] =  model.scales[0]
        batch_metric["scales2"] =  model.scales[1]
        batch_metric["scales3"] =  model.scales[2]
        batch_metric["scales4"] =  model.scales[3]

        metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

        batch_index += 1
    
    return dict(metrics.agg({
            "train_loss_nll":"mean",
            "train_loss_cla":"mean",
            "train_acc":lambda x:100*sum(x)/len(dataloader_train.dataset),
            "scales1":"mean","scales2":"mean","scales3":"mean","scales4":"mean"}))



def train_ladiff_oridiff(dataloader_train, model,
                 optimizerDiff, optimizerC, augmentor=None, attacker=None,
                 device=None, visualize=False, epoch=None):
        
    print("ladiff_oridiff..")

    metrics = pd.DataFrame()
    batch_index = 0
    model.train()
    

    for x, y in dataloader_train:

        batch_metric = defaultdict(float)
        x, y = x.to(device), y.to(device)
        
        x_aug = augmentor.apply(x, (True*visualize) if batch_index==0 else (False*visualize))

        out_fake = model(x, use_diffusion = True)
        mus = model.mus
        sigmas = model.sigmas

        out_aug = model(x_aug, use_diffusion = False)
        mus_aug = model.mus

        lossDiff = 0
        for mu_aug, mu, sigma in zip(mus_aug, mus, sigmas):
            lossDiff += nll_loss(mu_aug.view(x.shape[0],-1), mu.view(x.shape[0],-1), sigma.view(x.shape[0],-1))
        lossDiff = lossDiff/len(mus)
        
        optimizerDiff.zero_grad()
        lossDiff.backward()
        optimizerDiff.step()

        out_fake = model(x, use_diffusion = True)
        optimizerC.zero_grad()
        lossC = F.cross_entropy(out_fake, y, label_smoothing=0.1) #
        lossC.backward()
        optimizerC.step()

        batch_metric["train_loss_nll"] = lossDiff.data.item()
        batch_metric["train_loss_cla"] = lossC.data.item()
        batch_metric["train_acc"] = (torch.softmax(out_fake.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
        batch_metric["scales1"] =  model.scales[0]
        batch_metric["scales2"] =  model.scales[1]
        batch_metric["scales3"] =  model.scales[2]
        batch_metric["scales4"] =  model.scales[3]

        metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

        batch_index += 1
    
    return dict(metrics.agg({
            "train_loss_nll":"mean",
            "train_loss_cla":"mean",
            "train_acc":lambda x:100*sum(x)/len(dataloader_train.dataset),
            "scales1":"mean","scales2":"mean","scales3":"mean","scales4":"mean"}))


def train_ladiff(protocol):
    if 'augdiff' in protocol:
        return train_ladiff_augdiff
    elif 'oridiff' in protocol:
        return train_ladiff_oridiff
    else:
        raise


def train_standard(dataloader_train, model, optimizer, 
                   augmentor=None, attacker=None, device=None, visualize=False, epoch=None):
        
    metrics = pd.DataFrame()
    batch_index = 0
    model.train()

    for x, y in dataloader_train:

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

        batch_metric["train_loss"] = loss.data.item()
        batch_metric["train_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
        metrics = pd.concat([metrics, pd.DataFrame(batch_metric, index=[0])], ignore_index=True)

        batch_index += 1
    
    return dict(metrics.agg({
            "train_loss":"mean",
            "train_acc":lambda x:100*sum(x)/len(dataloader_train.dataset)}))

