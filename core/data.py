import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict
from torch.utils.data import Dataset
#from robustbench.data import load_cifar10c, load_cifar10, load_cifar100c, load_cifar100

corruption_19=[ 'snow', 'fog', 'frost', 'glass_blur', 'defocus_blur','motion_blur','zoom_blur','gaussian_blur',
                  'gaussian_noise','shot_noise','impulse_noise','speckle_noise',
                  'pixelate','brightness','contrast','jpeg_compression','elastic_transform','spatter','saturate']

corruption_15 = ['snow', 'fog', 'frost', 'glass_blur', 'defocus_blur', 'motion_blur','zoom_blur', 
               'gaussian_noise', 'shot_noise', 'impulse_noise',
               'pixelate', 'brightness', 'contrast','jpeg_compression', 'elastic_transform']

pacs_4 = ['art', 'cartoon', 'photo', 'sketch']

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
    def __init__(self, length):
        #length = int(len(dataset))
        self.indices = list(range(length))
        random.Random(1).shuffle(self.indices) 
    def __iter__(self):
        return iter(self.indices)


class PairedDataset(Dataset):
    def __init__(self, data, classes, transform=None):
        self.samples = [x[0] for x in data]
        self.targets = [x[1] for x in data]
        self.classes = classes
        self.transform = transform

    def __getitem__(self, index):
        image = self.dasamplesta[index]
        if self.transform:
            image = self.transform(image)
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.samples)

def Balance(root, data1, data2, t1, t2):
    def balance_classes(data1, data2):
        if len(data1) < len(data2):
            while len(data1) < len(data2):
                data1.append(random.choice(data1))
        elif len(data1) > len(data2):
            while len(data2) < len(data1):
                data2.append(random.choice(data2))

    # Load the datasets
    domain1 = datasets.ImageFolder(os.path.join(root, 'pacs', data1.split('-')[1]))
    domain2 = datasets.ImageFolder(os.path.join(root, 'pacs', data2.split('-')[1]))
    #print('domain1',domain1.samples)

    # Group images by class
    data_dict1 = defaultdict(list)
    data_dict2 = defaultdict(list)

    for image, label in domain1:
        data_dict1[label].append((image, label))
    #print(data_dict1)

    for image, label in domain2:
        data_dict2[label].append((image, label))

    balanced_data1 = []
    balanced_data2 = []

    # Balance the number of images in each class
    for label in range(len(domain1.classes)):
        print('label',label)
        balance_classes(data_dict1[label], data_dict2[label])
        balanced_data1.extend(data_dict1[label])
        balanced_data2.extend(data_dict2[label])
    #print('balanced_data1balanced_data1',balanced_data1)

    # Create the final datasets with balanced class distribution 
    dataset1 = PairedDataset(balanced_data1, domain1.classes, transform=t1)
    dataset2 = PairedDataset(balanced_data2, domain2.classes, transform=t2)
    return dataset1, dataset2

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
        transform_train_diff = [ T.Resize(32), 
                            T.RandomCrop(32, padding=4), 
                            T.RandomHorizontalFlip(),
                            Augmentor(args.aug_train_diff),
                            T.ToTensor()]
        transform_train_diff = T.Compose(transform_train_diff)
    
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
        data_train = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform = transform_train,train = True, download = True)
        data_test = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform = transform_eval,train = False)
    elif args.data.lower() == 'tin200':
        data_train = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'train'), transform=transform_train)
        data_test = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'val'), transform=transform_eval)
    elif 'pacs' in args.data.lower():
        data_train = datasets.ImageFolder(os.path.join(args.data_dir, 'pacs', args.data.split('-')[1]), transform=transform_train)
        data_test = None
    else:
        raise
    
    # load ood train data for pde-add
    if args.data_diff is not None:
        if args.data_diff.lower() == 'mnist':
            data_train_diff = datasets.MNIST(root=os.path.join(args.data_dir, 'mnist') ,transform=transform_train_diff, train = True, download = True)
        elif args.data_diff.lower() == 'cifar10':
            data_train_diff = datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), transform=transform_train_diff, train = True, download = True)
        elif args.data_diff.lower() == 'cifar100':
            data_train_diff = datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), transform=transform_train_diff, train = True, download = True)
        elif args.data_diff.lower() == 'tin200':
            data_train_diff = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200', 'train'), transform=transform_train_diff)
        elif 'pacs' in args.data_diff.lower():
            data_train_diff = datasets.ImageFolder(os.path.join(args.data_dir, 'pacs', args.data_diff.split('-')[1]), transform=transform_train_diff)
        else:
            raise

    if args.npc_train != 'all':
        index = split_small_dataset(num=int(args.npc_train), tar = data_train.targets, num_class=len(data_train.classes))
        data_train.data = data_train.data[index]
        data_train.targets = np.array(data_train.targets)[index].tolist()
        if args.data_diff is not None:
            data_train_diff.data = data_train_diff.data[index]
            data_train_diff.targets = np.array(data_train_diff.targets)[index].tolist()

    sampler = FixSampler(int(len(data_train)))
    dataloader_train =  DataLoader(dataset=data_train, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=True)
    
    if args.data_diff is not None:
        dataloader_train_diff =  DataLoader(dataset=data_train_diff, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=True)
    else:
        dataloader_train_diff = None
    
    if data_test is not None:
        dataloader_test =  DataLoader(dataset=data_test, batch_size=args.batch_size_validation, shuffle = False, num_workers=args.num_workers, pin_memory=True)
    else:
         dataloader_test = None
    
    return dataloader_train,  dataloader_train_diff, dataloader_test

def load_dg_dataloader(args):
    transform_train = [ T.Resize(32), 
                        T.RandomCrop(32, padding=4), 
                        T.RandomHorizontalFlip(),
                        Augmentor(args.aug_train),
                        T.ToTensor()]
    transform_train = T.Compose(transform_train)

    if args.data_diff is not None:
        transform_train_diff = [ T.Resize(32), 
                            T.RandomCrop(32, padding=4), 
                            T.RandomHorizontalFlip(),
                            Augmentor(args.aug_train_diff),
                            T.ToTensor()]
        transform_train_diff = T.Compose(transform_train_diff)
    
    
    dataset_train, dataset_train_diff = Balance(args.data_dir, args.data, args.data_diff, transform_train, transform_train_diff)
    sampler = FixSampler(int(len(dataset_train)))
    dataloader_train =  DataLoader(dataset=dataset_train, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=True)
    dataloader_train_diff =  DataLoader(dataset=dataset_train_diff, sampler=sampler, batch_size=args.batch_size, shuffle = False, num_workers=args.num_workers, pin_memory=True)
    return dataloader_train,  dataloader_train_diff, None


class CIFARC(datasets.VisionDataset):
    def __init__(self, root :str, name=None,
                 transform=None, target_transform=None, dnum='all', severity=0):
        #assert name in corruptions
        super(CIFARC, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
        if severity:
            self.data=self.data[10000*(severity-1):10000*severity, :]
            self.targets=self.targets[10000*(severity-1):10000*severity]
        
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

def load_corr_dataloader(data_name, data_dir, batch_size, cname=None, severity=None, num_workers=2):
    transform = T.Compose([T.Resize(32), T.ToTensor()])
    if data_name in ['cifar10','cifar100']:
        filename = '-'.join(['CIFAR', data_name[5:], 'C'])   
        data = CIFARC(os.path.join(data_dir, filename), cname, transform=transform, severity=severity)  
        dataloader =  DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif data_name == 'tin200':
        severity = str(severity)
        data = datasets.ImageFolder(os.path.join(data_dir, 'Tiny-ImageNet-C', cname, severity), transform=transform)
        dataloader =  DataLoader(dataset=data, batch_size=batch_size, shuffle = False, num_workers=num_workers, pin_memory=True)
    elif 'pacs' in data_name:
        data = datasets.ImageFolder(os.path.join(data_dir, 'pacs', cname), transform=transform)
        dataloader =  DataLoader(dataset=data, batch_size=batch_size, shuffle = False, num_workers=num_workers, pin_memory=True)
    else:
        raise
    return dataloader   
