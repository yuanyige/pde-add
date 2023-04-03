'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion(nn.Module):
    def __init__(self, planes):
        super(Diffusion, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(planes, 2*planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(2*planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out =  self.main(input)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

        self.diff = Diffusion(planes)

    def forward(self, x):
        x, use_diffusion, _ = x 
        
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        #return out

        if use_diffusion:
            sigma = self.diff(out)
            out = out + sigma * torch.randn_like(out)
            return (out, use_diffusion, sigma)
        else:
            return (out, use_diffusion, 0)


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward(self, x, use_diffusion=True):
        
        self.mus = []
        self.sigmas = [] 
        self.scales = []

        out = self.conv1(x)
        out = self.layer1((out, use_diffusion, 0))
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())

        out = self.layer2(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())

        out = self.layer3(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())

        out = self.layer4(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())

        out = out[0]

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def ladiff_preresnet(name, num_classes=10, pretrained=False, device='cpu'):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'preresnet-18':
        return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == 'preresnet-34':
        return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)
    
    raise ValueError('Only preresnet18, preresnet34, resnet50 and resnet101 are supported!')

