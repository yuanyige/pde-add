import os
import json
import torch
from torchvision import datasets
import torchvision.transforms as T

from core.data import load_cifar10c
from core.utils import get_logger, set_seed
from core.models import load_model

ckpt_path = "/home/yuanyige/Ladiff_nll/exps/d500/abla/resnet-18_ladiff-none_ntr500_(Csgdlr0.005-Dadamlr0.1)_e400_b128_atr-augmix"
task = "ood" # ood advlinf
os.environ['CUDA_VISIBLE_DEVICES']='3'
severity = 0
load_ckpt = 'model-best-endiff'

with open(os.path.join(ckpt_path,'train/args.txt'), 'r') as f:
    args = json.load(f)
    #print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args["seed"])
#save_dir = './exps/test'
save_dir = os.path.join(ckpt_path, 'test')
os.makedirs(save_dir, exist_ok=True)
logger = get_logger(logpath=os.path.join(save_dir, 'scale-{}-s{}.log'.format(task,severity)), filepath=os.path.abspath(__file__))


model = load_model(args["backbone"], args["protocol"])
model = model.to(device)
checkpoint = torch.load(os.path.join(ckpt_path,'train',load_ckpt+'.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out


def ladiff_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if 'diff.main.5' in n:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
    return fea_hooks

# def fixdiff_hook(model):
#     fea_hooks = []
#     for n, m in model.named_modules():
#         print('names',n)
#         if n == 'layer1.0.diff.main':
#             cur_hook = HookTool()
#             m.register_forward_hook(cur_hook.hook_fun)
#             fea_hooks.append(cur_hook)
#     return fea_hooks

# def final_test(loader, model, task,  device):
#     if args["protocol"] == 'ladiff':
#         scale_hooks = ladiff_hook(model)
#     # elif args["protocol"] == 'fixdiff' :
#     #     scale_hooks = fixdiff_hook(model)
#     test_hook(loader, model,  use_diffusion=True, device=device)
#     logger.info(task)
#     scale = 0
#     for i in scale_hooks:
#         print(i.fea.shape)
#         #scale+=i.fea
#     #logger.info(scale.fea.max().data.item())
        
def test_hook(dataloader_test, model, task,  device=None):
    model.eval()
    scale_hooks = ladiff_hook(model)
    with torch.no_grad():
        all = []
        for x, y in dataloader_test:
            x, y = x.to(device), y.to(device)
            model(x, use_diffusion = True)
            scales = 0
            for i in scale_hooks:
                scale = i.fea.view(x.shape[0],-1).mean(dim=1)
                scales += scale
            scales = scales/len(scale_hooks)
            all.append(scales)
        all_scale = torch.cat(all,dim=0)
    logger.info("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(task, all_scale.mean(), all_scale.max(), all_scale.min(), all_scale.median(),all_scale.std()))

data_test = datasets.CIFAR10(root=os.path.join(args["data_dir"], 'cifar10'), transform = T.Compose([T.ToTensor()]), train = False)
loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size= args["batch_size_validation"], shuffle = False)
test_hook(loader_test, model, task='iid',  device=device)

cname_list = ['fog','snow','frost',
              'zoom_blur','defocus_blur','glass_blur','gaussian_blur','motion_blur',
              'speckle_noise','shot_noise','impulse_noise','gaussian_noise',
              'jpeg_compression','pixelate','spatter',
              'elastic_transform','brightness','saturate','contrast']
for ci, cname in enumerate(cname_list):
    # Load data
    loader_c = load_cifar10c(args["data"], args["data_dir"], args["batch_size_validation"], cname, severity=severity)
    test_hook(loader_c, model, task=cname,  device=device)

logger.info("Evalution done.")

    
