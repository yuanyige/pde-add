import os
import json
import torch
import numpy as np
import pandas as pd

from core.models import create_model
from core.testfn import test
from core.parse import parser_test
from core.utils import get_logger, get_logger_name
from core.data import corruption_19, corruption_15, load_corr_dataloader, load_dataloader
import torchvision.transforms as T

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

corr = eval('_'.join(['corruption',args_test.type[1:]]))
if args_test.data == 'cifar10':
    baseline_accs = (0.9535, [0.832,0.89348,0.80054,0.55018,0.8274,0.78482,0.78786,0.48148,0.60628,0.53044,0.75678,0.94036,0.76344,0.78972,0.85054],[0.7814,0.7552,0.6652,0.4942,0.5429,0.661,0.6369,0.2826,0.3597,0.2354,0.4816,0.9156,0.3,0.712,0.7554])
elif args_test.data == 'cifar100':
    baseline_accs = (0.7771,[0.5540,0.6465,0.5049,0.2418,0.6003,0.5545,0.5360,0.2302,0.3144,0.2548,0.5245,0.7372,0.5538,0.5231,0.6150],[0.4611,0.4040,0.3477,0.2207,0.3371,0.4430,0.3998,0.1111,0.1277,0.0662,0.2394,0.6656,0.1988,0.4420,0.5123])
else:
    raise

logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task))
#augmentor = T.RandomRotation(360)
#augmentor = T.GaussianBlur(5,5)
#augmentor = T.ElasticTransform(150.0)
#augmentor = T.RandomInvert()
augmentor = T.ColorJitter(5,0,0,0)

_,_,dataloader_nat = load_dataloader(args_test)
dict = test(dataloader_nat, model,device=device, augmentor=augmentor)
# logger.info("nat-"+str(dict["eval_acc"]))
# logger.info("aug-"+str(dict["eval_acc_aug"]))
# logger.info("dis-"+str(dict["distance"]))
# logger.info("dismin-"+str(dict["distance_min"]))
# logger.info("dismax-"+str(dict["distance_max"]))
# logger.info("disstd-"+str(dict["distance_std"]))
# exit(0)
res = np.zeros((5, len(corr)))
for c in range(len(corr)):
    for s in range(1, 6):
        dataloader = load_corr_dataloader(args_test.data, args_test.data_dir, args_test.batch_size, cname=corr[c], dnum='all', severity=s)
        dict = test(dataloader, model,device=device, augmentor=augmentor)
        res[s-1, c] = dict["eval_acc"]
        log = "-".join([corr[c], str(s), str(res[s-1, c])])
        logger.info(log)
frame = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=corr)
frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
frame['avg'] = frame[list(range(1, 6))].mean(axis=1)
logger.info(frame)