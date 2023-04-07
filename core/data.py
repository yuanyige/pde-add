import os
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def split_small_dataset(num, tar, num_class):
    index = []
    for k in range(num_class):
        l = [i for i,j in enumerate(tar) if j == k][:num]
        index.extend(l)
    return index



def load_data(args):
    
    # if args.data == 'cifar10':
    #     mean, std = CIFAR10_TRAIN_MEAN,CIFAR10_TRAIN_STD
    # elif args.data == 'cifar100':
    #     mean, std = CIFAR100_TRAIN_MEAN,CIFAR100_TRAIN_STD

    transform_train = T.Compose([T.RandomCrop(32, padding=4), 
                            T.RandomHorizontalFlip(),
                            T.ToTensor()])
    transform_eval = T.Compose([T.ToTensor()])


    if args.data.lower() == 'mnist':
        data_train = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform=transform_train,train = True, download = True)
        data_test = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform = transform_eval,train = False)
    elif args.data.lower() == 'cifar10':
        data_train = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform=transform_train,train = True, download = True)
        data_test = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform = transform_eval,train = False)
    elif args.data.lower() == 'cifar100':
        data_train = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform=transform_train,train = True, download = True)
        data_test = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform = transform_eval,train = False)
    else:
        raise
    
    if args.npc_train != 'all':
        index = split_small_dataset(num=int(args.npc_train), tar = data_train.targets, num_class=len(data_train.classes))
        data_train.data = data_train.data[index]
        data_train.targets = np.array(data_train.targets)[index].tolist()

    dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle = True)
    dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size_validation, shuffle = False)

    return dataloader_train, dataloader_test




# class Identity(torch.nn.Module):
#     def __init__(self):
#         super().__init__()    
#     def forward(self, x):
#         return x



class DataAugmentor():
    def __init__(self, augments, save_path=None, name=''):
        augments = augments.split('-')
        self.aug_type = augments[0]
        self.aug_param = []
        if len(augments) > 1:
            for p in augments[1:]:
                if '.' in p:
                    p = float(p)
                else:
                    p = int(p)
                self.aug_param.append(p)
            #print(self.aug_param)
            #self.aug_param = [int(p) for p in augments[1:]]
        self.aug_mapping = {"gaublur":T.GaussianBlur, 
                            "elastic":T.ElasticTransform,
                            "contrast":T.RandomAutocontrast,
                            "invert":T.RandomInvert,
                            "color":T.ColorJitter,
                            "rotation":T.RandomRotation,
                            "augmix":T.AugMix,
                            "randaug":T.RandAugment}
        self.save_path = save_path
        self.name=name
    
    def apply(self, x, visualize=False):
        x_ori = x
        augmentor = self.aug_mapping[self.aug_type](*self.aug_param)
        x = augmentor(convert_image_dtype(x, torch.uint8))
        x = convert_image_dtype(x, torch.float)
        
        if visualize:
            self.vis(x_ori, x)
        return x
    
    def vis(self, x_ori, x_aug):
        x=torch.cat([x_ori[:8],x_aug[:8]])
        save_image(x.cpu(), os.path.join(self.save_path,'ood-{}{}.png'.format(self.aug_type, self.name)), nrow=8,
            padding=0, value_range=(0, 1), pad_value=0)






# all_ood=['snow','fog','frost','glass_blur',
#         'defocus_blur','gaussian_blur','motion_blur','zoom_blur',
#         'gaussian_noise','shot_noise','speckle_noise','impulse_noise',
#         'brightness','contrast','elastic_transform',
#         'pixelate','jpeg_compression','spatter','saturate']
corruptions = ['snow', 'fog', 'frost', 'glass_blur', 'defocus_blur', 'motion_blur','zoom_blur', 
               'gaussian_noise', 'shot_noise', 'impulse_noise',
               'pixelate', 'brightness', 'contrast','jpeg_compression', 'elastic_transform']

from robustbench.data import load_cifar10c, load_cifar10

def get_cifar10_numpy():
    x_clean, y_clean = load_cifar10(n_examples=10000, data_dir='./datasets/cifar10')
    x_corrs = []
    y_corrs = []
    x_corrs.append(x_clean)
    y_corrs.append(y_clean)
    for i in range(1, 6):
        x_corr = []
        y_corr = []
        for j, corr in enumerate(corruptions):
            x_, y_ = load_cifar10c(n_examples=10000, data_dir='./datasets/', severity=i, corruptions=(corr,))
            x_corr.append(x_)
            y_corr.append(y_)

        x_corrs.append(x_corr)
        y_corrs.append(y_corr)

    x_corrs_fast = []
    y_corrs_fast = []
    for i in range(1, 6):
        x_, y_ = load_cifar10c(n_examples=1000, data_dir='./datasets/', severity=i, shuffle=True)
        x_corrs_fast.append(x_)
        y_corrs_fast.append(y_)

    return x_corrs, y_corrs, x_corrs_fast, y_corrs_fast


class CIFARC(datasets.VisionDataset):
    def __init__(self, root :str, name=None,
                 transform=None, target_transform=None, dnum='all', severity=0):
        #assert name in corruptions
        super(CIFARC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        if dnum == 'all': 
            data_path = os.path.join(root, name + '.npy')
            target_path = os.path.join(root, 'labels.npy')
            
            self.data = np.load(data_path)
            self.targets = np.load(target_path)
            
            if severity:
                self.data=self.data[10000*(severity-1):10000*severity, :]
                self.targets=self.targets[10000*(severity-1):10000*severity]
              
        elif (dnum != 'all') and (name==None):    
            
            self.data = []
            self.targets = []

            for n in corruptions:   
                data_path = os.path.join(root, n + '.npy')
                target_path = os.path.join(root, 'labels.npy')
                
                adata = np.load(data_path)
                atargets = np.load(target_path)
                
                index = split_small_dataset(num=dnum, tar = atargets, num_class=atargets.max()+1)
                adata = adata[index]
                atargets = atargets[index]

                self.data.append(adata)
                self.targets.append(atargets)
            
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)



def load_cifar_c(data_name, data_dir, batch_size, cname=None, dnum='all', severity=0, norm=False):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    # if data_name == 'cifar10':
    #     mean, std = CIFAR10_TRAIN_MEAN,CIFAR10_TRAIN_STD
    # elif data_name == 'cifar100':
    #     mean, std = CIFAR100_TRAIN_MEAN,CIFAR100_TRAIN_STD
    
    transform = T.Compose([T.ToTensor()])

    if dnum =='all': 
        data_cifar10c = CIFARC(os.path.join(data_dir, data_name+'c'), cname, transform=transform, dnum='all', severity=severity)  
        dataloader_cifar10c = torch.utils.data.DataLoader(data_cifar10c, batch_size=batch_size, shuffle=False)
    else:
        transform = T.Compose([T.ToTensor()])
        data_cifar10c = CIFARC(os.path.join(data_dir, data_name+'c'), transform=transform, dnum=dnum, severity=severity)  
        dataloader_cifar10c = torch.utils.data.DataLoader(data_cifar10c, batch_size=batch_size, shuffle=False)
    
    return dataloader_cifar10c   




