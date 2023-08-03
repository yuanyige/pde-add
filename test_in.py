import os
import json
import torch
from core.models import create_model
from core.testfn import test
from core.parse import parser_test
from core.utils import get_logger, get_logger_name
from core.data import all_corruptions, corruptions, get_cifar10_numpy, get_cifar100_numpy
from robustbench.data import load_imagenetc
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
import pandas as pd
args_test = parser_test()

with open(os.path.join(args_test.ckpt_path,'train/args.txt'), 'r') as f:
    old = json.load(f)
    args_test.__dict__ = dict(vars(args_test), **old)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(args_test.data, args_test.backbone, args_test.protocol)
model = model.to(device)
checkpoint = torch.load(os.path.join(args_test.ckpt_path,'train',args_test.load_ckpt+'.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

if args_test.type == 'c15':
    corr = corruptions
elif args_test.type == 'c19':
    corr = all_corruptions
else:
    raise

transform_eval=[T.Resize(32), T.ToTensor()]
if args_test.norm:
    transform_eval.append(T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
transform_eval = T.Compose(transform_eval)
logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task))

res = np.zeros((5, len(corr)))
for c in range(len(corr)):
    for s in range(1, 6):
        data_test = datasets.ImageFolder(os.path.join(args_test.data_dir, 'Tiny-ImageNet-C', corr[c], str(s)), transform=transform_eval)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=args_test.batch_size, shuffle = False, num_workers=0, pin_memory=True)
        dict = test(dataloader_test, model,device=device)
        res[s-1, c] = dict["eval_acc"]
        log = "-".join([corr[c], str(s), str(res[s-1, c])])
        logger.info(log)
frame = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=corr)
frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
frame['avg'] = frame[list(range(1, 6))].mean(axis=1)
logger.info(frame)