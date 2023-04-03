import os
import json
import torch

from core.models import create_model
from core.testfn import run_final_test_select_cifarc, run_final_test_cifarc, run_final_test_selfmade
from core.parse import parser_test

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

run_final_test_cifarc(model, args_test, device=device)
