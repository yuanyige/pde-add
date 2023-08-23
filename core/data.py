import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Sampler, DataLoader
#from robustbench.data import load_cifar10c, load_cifar10, load_cifar100c, load_cifar100

corruption_19=[ 'snow', 'fog', 'frost', 'glass_blur', 'defocus_blur','motion_blur','zoom_blur','gaussian_blur',
                  'gaussian_noise','shot_noise','impulse_noise','speckle_noise',
                  'pixelate','brightness','contrast','jpeg_compression','elastic_transform','spatter','saturate']

corruption_15 = ['snow', 'fog', 'frost', 'glass_blur', 'defocus_blur', 'motion_blur','zoom_blur', 
               'gaussian_noise', 'shot_noise', 'impulse_noise',
               'pixelate', 'brightness', 'contrast','jpeg_compression', 'elastic_transform']

def split_small_dataset(num, tar, num_class):
    index = []
    for k in range(num_class):
        l = [i for i,j in enumerate(tar) if j == k][:num]
        index.extend(l)
    return index

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class FixSampler(Sampler):
    def __init__(self, dataset):
        length = int(len(dataset))
        self.indices = list(range(length))
        random.Random(1).shuffle(self.indices) 
    def __iter__(self):
        return iter(self.indices)

class Augmentor(torch.nn.Module):
    def __init__(self, augments):
        super().__init__()
        self.augments=augments
        self.mapping = {
            "gaublur":T.GaussianBlur, 
            "elastic":T.ElasticTransform,
            "contrast":T.RandomAutocontrast,
            "invert":T.RandomInvert,
            "color":T.ColorJitter,
            "rotation":T.RandomRotation,
            "augmix":T.AugMix,
            "randaug":T.RandAugment,
            "autoaug":T.AutoAugment,
            "none":Identity}  

    def forward(self, img):
        type, param = self.split_string()
        return self.mapping[type](*param)(img)
    
    def split_string(self):
        augments = self.augments.split('-')
        type = augments[0]
        param = []
        if len(augments) > 1:
            for p in augments[1:]:
                if '.' in p:
                    p = float(p)
                else:
                    p = int(p)
                param.append(p)
        return type, param
    
def load_dataloader(args):
    # define transforms
    transform_train = [ T.Resize(32), 
                        T.RandomCrop(32, padding=4), 
                        T.RandomHorizontalFlip(),
                        Augmentor(args.aug_train),
                        T.ToTensor()]
    transform_train = T.Compose(transform_train)

    if args.data_diff is not None:
        transform_train_ood = [ T.Resize(32), 
                            T.RandomCrop(32, padding=4), 
                            T.RandomHorizontalFlip(),
                            Augmentor(args.aug_train_diff),
                            T.ToTensor()]
        transform_train_ood = T.Compose(transform_train_ood)
    
    transform_eval = [T.Resize(32), T.ToTensor()]
    transform_eval = T.Compose(transform_eval)

    # load train & test data
    if args.data.lower() == 'mnist':
        data_train = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform=transform_train,train = True, download = True)
        data_test = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform = transform_eval,train = False)
    elif args.data.lower() == 'cifar10':
        data_train = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform=transform_train, train = True, download = True)
        data_test = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform = transform_eval, train = False)
    elif args.data.lower() == 'cifar100':
        data_train = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform=transform_train,train = True, download = True)
        data_test = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform = transform_eval,train = False)
    elif args.data.lower() == 'tin200':
        data_train = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'train'), transform=transform_train)
        data_test = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'val'), transform=transform_eval)
    else:
        raise
    
    # load ood train data for pde-add
    if args.data_diff is not None:
        if args.data_diff.lower() == 'mnist':
            data_train_diff = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform=transform_train_ood, train = True, download = True)
        elif args.data_diff.lower() == 'cifar10':
            data_train_diff = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform=transform_train_ood, train = True, download = True)
        elif args.data_diff.lower() == 'cifar100':
            data_train_diff = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform=transform_train_ood, train = True, download = True)
        elif args.data_diff.lower() == 'tin200':
            data_train_diff = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'train'), transform=transform_train_ood)
        else:
            raise

    if args.npc_train != 'all':
        index = split_small_dataset(num=int(args.npc_train), tar = data_train.targets, num_class=len(data_train.classes))
        data_train.data = data_train.data[index]
        data_train.targets = np.array(data_train.targets)[index].tolist()
        if args.data_diff is not None:
            data_train_diff.data = data_train_diff.data[index]
            data_train_diff.targets = np.array(data_train_diff.targets)[index].tolist()

    sampler = FixSampler(data_train)
    dataloader_train =  DataLoader(dataset=data_train, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=False)
    if args.data_diff is not None:
        dataloader_train_diff =  DataLoader(dataset=data_train_diff, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=False)
    else:
        dataloader_train_diff = None
    dataloader_test =  DataLoader(dataset=data_test, batch_size=args.batch_size_validation, shuffle = False, num_workers=args.num_workers, pin_memory=False)

    return dataloader_train,  dataloader_train_diff, dataloader_test

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

            for n in corruption_15:   
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

def load_corr_dataloader(data_name, data_dir, batch_size, cname=None, dnum='all', severity=0, num_workers=2):
    transform = T.Compose([T.Resize(32),T.ToTensor()])
    if data_name in ['cifar10','cifar100']:
        filename = '-'.join(['CIFAR', data_name[5:], 'C'])   
        if dnum =='all': 
            data = CIFARC(os.path.join(data_dir, filename), cname, transform=transform, dnum='all', severity=severity)  
            dataloader =  DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        else:
            data = CIFARC(os.path.join(data_dir, filename), transform=transform, dnum=dnum, severity=severity)  
            dataloader =  DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    elif data_name == 'tin200':
        severity = str(severity)
        data = datasets.ImageFolder(os.path.join(data_dir, 'Tiny-ImageNet-C', cname, severity), transform=transform)
        dataloader =  DataLoader(dataset=data, batch_size=batch_size, shuffle = False, num_workers=num_workers, pin_memory=False)
    else:
        raise
    return dataloader   
