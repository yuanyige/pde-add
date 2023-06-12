import os
import json
import torch
from core.models import create_model
from core.testfn import final_corr_eval
from core.parse import parser_test
from core.utils import get_logger, get_logger_name
from core.data import all_corruptions, corruptions, get_cifar10_numpy, get_cifar100_numpy

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

if args_test.type == 's15':
    corr = corruptions
else:
    corr = all_corruptions

if args_test.data == 'cifar10':
    x_corrs, y_corrs, _, _ = get_cifar10_numpy(corr)
    baseline_accs = (0.9535, [0.832,0.89348,0.80054,0.55018,0.8274,0.78482,0.78786,0.48148,0.60628,0.53044,0.75678,0.94036,0.76344,0.78972,0.85054],[0.7814,0.7552,0.6652,0.4942,0.5429,0.661,0.6369,0.2826,0.3597,0.2354,0.4816,0.9156,0.3,0.712,0.7554])
elif args_test.data == 'cifar100':
    x_corrs, y_corrs, _, _ = get_cifar100_numpy(corr)
    baseline_accs = (0.7771,[0.5540,0.6465,0.5049,0.2418,0.6003,0.5545,0.5360,0.2302,0.3144,0.2548,0.5245,0.7372,0.5538,0.5231,0.6150],[0.4611,0.4040,0.3477,0.2207,0.3371,0.4430,0.3998,0.1111,0.1277,0.0662,0.2394,0.6656,0.1988,0.4420,0.5123])
else:
    raise

logger = get_logger(get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task))
final_corr_eval(x_corrs, y_corrs, model, use_diffusion=True, corruptions=corr, baseline_accs=baseline_accs, logger=logger)