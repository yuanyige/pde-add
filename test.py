import os
import json
import torch
from core.utils import set_seed
from core.models import create_model
from core.testfn import final_corr_eval, run_final_test_autoattack
from core.parse import parser_test
from core.utils import set_seed, get_logger, get_logger_name
from core.data import corruptions, get_cifar10_numpy

args_test = parser_test()

with open(os.path.join(args_test.ckpt_path,'train/args.txt'), 'r') as f:
    old = json.load(f)
    args_test.__dict__ = dict(vars(args_test), **old)

set_seed(args_test.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(args_test.data, args_test.backbone, args_test.protocol)
model = model.to(device)
checkpoint = torch.load(os.path.join(args_test.ckpt_path,'train',args_test.load_ckpt+'.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



x_corrs, y_corrs, _, _ = get_cifar10_numpy()
logger_path = get_logger_name(args_test.ckpt_path, args_test.load_ckpt, args_test.main_task)
logger = get_logger(logger_path)

if args_test.main_task =='ood':
    logger.info("use diffusion..")
    final_corr_eval(x_corrs, y_corrs, model, use_diffusion=True, corruptions=corruptions, logger=logger)

    logger.info("not use diffusion..")
    final_corr_eval(x_corrs, y_corrs, model, use_diffusion=False, corruptions=corruptions, logger=logger)

elif args_test.main_task =='adv':
    run_final_test_autoattack(model, args_test, logger=logger_path, device=device )