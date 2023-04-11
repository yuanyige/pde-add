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
            
            if use_diffusion:
                out = 0 
                for k in range(10):  
                    o = model(x, use_diffusion=True)
                    out = out + o
                out = out/10
            else:
                out = model(x, use_diffusion = False)

            batch_metric["eval_loss"] = F.cross_entropy(out, y).data.item()
            batch_metric["eval_acc"] = (torch.softmax(out.data, dim=1).argmax(dim=1) == y.data).sum().data.item()
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
    loss = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            if use_diffusion:
                output = 0 
                for k in range(10):  
                    o = model(x, use_diffusion=True)
                    output = output + o
                output = output/10
                
            else:
                output = model(x_curr, use_diffusion=False)
            
            acc += (output.max(1)[1] == y_curr).float().sum()

            loss += F.cross_entropy(output,y_curr).float()
        

    return acc.item() / x.shape[0], loss.item() / n_batches



def final_corr_eval(x_corrs, y_corrs, model, use_diffusion, corruptions, logger, n_runs=1):
    model.eval()
    clean_acc=np.zeros((1, n_runs))
    clean_loss =np.zeros((1, n_runs))
    for k in range(n_runs):
        clean_acc[0, k], clean_loss[0, k] = clean_accuracy(model, use_diffusion, x_corrs[0].to(list(model.parameters())[0].device), y_corrs[0].to(list(model.parameters())[0].device))
    logger.info("Clean accuracy: {:.2%}+-{:.2} ".format(clean_acc.mean(),clean_acc.std()))
    logger.info("Clean loss: {:.2}+-{:.2} ".format(clean_loss.mean(),clean_loss.std()))

    res_acc = np.zeros((5, 15, n_runs))
    res_loss = np.zeros((5, 15, n_runs))
    for i in range(1, 6):
        for j, c in enumerate(corruptions):
            for k in range(n_runs):
                res_acc[i-1, j, k], res_loss[i-1, j, k] = clean_accuracy(model, use_diffusion, x_corrs[i][j].to(list(model.parameters())[0].device), y_corrs[i][j].to(list(model.parameters())[0].device))
                print(f"{c} {i} {res_acc[i-1, j, k]}")
                print(f"{c} {i} {res_loss[i-1, j, k]}")
    
    mean_acc = np.mean(res_acc, axis=2)
    std_acc = np.std(res_acc, axis=2)

    mean_loss = np.mean(res_loss, axis=2)
    std_loss = np.std(res_loss, axis=2)
    
    frame = pd.DataFrame({i+1: [f"{mean_acc[i, j]:.2%}+-{std_acc[i, j]:.2%}" for j in range(15)] for i in range(0, 5)}, index=corruptions)
    frame.loc['average'] = {i+1: f"{np.mean(mean_acc, axis=1)[i]:.2%}+-{np.mean(std_acc, axis=1)[i]:.2%}" for i in range(0, 5)}
    frame['avg'] = [f"{np.mean(mean_acc[:, i]):.2%}+-{np.mean(std_acc[:, i]):.2%}" for i in range(15)] + [f"{np.mean(mean_acc):.2%}+-{np.mean(std_acc):.2%}"]
    logger.info(frame)

    frame_loss = pd.DataFrame({i+1: [f"{mean_loss[i, j]:.2}+-{std_loss[i, j]:.2}" for j in range(15)] for i in range(0, 5)}, index=corruptions)
    frame_loss.loc['average'] = {i+1: f"{np.mean(mean_loss, axis=1)[i]:.2}+-{np.mean(std_loss, axis=1)[i]:.2}" for i in range(0, 5)}
    frame_loss['avg'] = [f"{np.mean(mean_loss[:, i]):.2}+-{np.mean(std_loss[:, i]):.2}" for i in range(15)] + [f"{np.mean(mean_loss):.2%}+-{np.mean(std_acc):.2%}"]
    logger.info(frame_loss)

    return frame, frame_loss



def run_final_test_autoattack(model, args_test, logger, device):

    _, loader_test = load_data(args_test)

    l = [x for (x, y) in loader_test]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader_test]
    y_test = torch.cat(l, 0)

    if args_test.threat =='Linf':
        epsilon = 8 / 255.
    elif args_test.threat =='L2':
        epsilon = 0.5
    adversary = AutoAttack(model, norm=args_test.threat, eps=epsilon, version='standard', log_path=logger, seed=args_test.seed)
    
    def get_logits_en(self, x):
        if not self.is_tf_model:
            proba = 0 
            for k in range(10):  
                o = self.model(x, use_diffusion=True)
                proba = proba + o
            out = proba/10
            return out
        else:
            return self.model.predict(x)
    adversary.get_logits = get_logits_en

    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
