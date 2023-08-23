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


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

        



class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on. 
    """
    def __init__(self, block, num_blocks, num_classes=10, device='cpu'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.diff1 = Diffusion(64)
        self.diff2 = Diffusion(128)
        self.diff3 = Diffusion(256)
        self.diff4 = Diffusion(512)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def net(self, x, use_diffusion=True):
        
        self.mus = []
        self.sigmas = [] 
        self.scales = []

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        if use_diffusion:
            sigma = self.diff1(out)
        else:
            sigma = torch.zeros_like(out)
        out_diff = out + sigma * torch.randn_like(out)
        self.mus.append(out)
        self.sigmas.append(sigma)
        self.scales.append(sigma.mean().detach().data.item())

        out = self.layer2(out_diff)
        if use_diffusion:
            sigma = self.diff2(out)
        else:
            sigma = torch.zeros_like(out)
        out_diff = out + sigma * torch.randn_like(out)
        self.mus.append(out)
        self.sigmas.append(sigma)
        self.scales.append(sigma.mean().detach().data.item())

        out = self.layer3(out_diff)
        if use_diffusion:
            sigma = self.diff3(out)
        else:
            sigma = torch.zeros_like(out)
        out_diff = out + sigma * torch.randn_like(out)
        self.mus.append(out)
        self.sigmas.append(sigma)
        self.scales.append(sigma.mean().detach().data.item())

        out = self.layer4(out_diff)
        if use_diffusion:
            sigma = self.diff4(out)
        else:
            sigma = torch.zeros_like(out)
        out_diff = out + sigma * torch.randn_like(out)
        self.mus.append(out)
        self.sigmas.append(sigma)
        self.scales.append(sigma.mean().detach().data.item())

        out = F.avg_pool2d(out_diff, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
    def forward(self, x, use_diffusion=True):
        #if self.training:
        out = self.net(x, use_diffusion=use_diffusion)
        # else:
        #     if use_diffusion:
        #         proba = 0 
        #         for _ in range(10):  
        #             out = self.net(x, use_diffusion=True)
        #             proba = proba + out
        #         out = proba/10
        #     else:
        #         out = self.net(x, use_diffusion=False)
        return out



def pdeadd_resnet(name, num_classes=10, pretrained=False, device='cpu'):
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
    if name == 'resnet-18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, device=device)
    elif name == 'resnet-34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, device=device)
    
    raise ValueError('Only resnet18, resnet34, resnet50 and resnet101 are supported!')


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Diffusion(nn.Module):
#     def __init__(self, planes):
#         super(Diffusion, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(planes, 2*planes, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(2*planes),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True))

#     def forward(self, input):
#         out =  self.main(input)
#         return out


# class BasicBlock(nn.Module):
#     """
#     Implements a basic block module for Resnets.
#     Arguments:
#         in_planes (int): number of input planes.
#         out_planes (int): number of output filters.
#         stride (int): stride of convolution.
#     """
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes))

#         self.diff = Diffusion(planes)


#     def forward(self, x):
#         x, use_diffusion, _ = x 

#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
        
#         if use_diffusion:
#             sigma = self.diff(out)
#             out = out + sigma * torch.randn_like(out)
#             return (out, use_diffusion, sigma)
#         else:
#             return (out, use_diffusion, 0)

        



# class ResNet(nn.Module):
#     """
#     ResNet model
#     Arguments:
#         block (BasicBlock or Bottleneck): type of basic block to be used.
#         num_blocks (list): number of blocks in each sub-module.
#         num_classes (int): number of output classes.
#         device (torch.device or str): device to work on. 
#     """
#     def __init__(self, block, num_blocks, num_classes=10, device='cpu'):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def net(self, x, use_diffusion=True):
        
#         self.mus = []
#         self.sigmas = [] 
#         self.scales = []

#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1((out, use_diffusion, 0))
#         self.mus.append(out[0])
#         self.sigmas.append(out[2])
#         if use_diffusion:
#             self.scales.append(out[2].max().detach().data.item())
#         else:
#             self.scales.append(0)

#         out = self.layer2(out)
#         self.mus.append(out[0])
#         self.sigmas.append(out[2])
#         if use_diffusion:
#             self.scales.append(out[2].max().detach().data.item())
#         else:
#             self.scales.append(0)

#         out = self.layer3(out)
#         self.mus.append(out[0])
#         self.sigmas.append(out[2])
#         if use_diffusion:
#             self.scales.append(out[2].max().detach().data.item())
#         else:
#             self.scales.append(0)

#         out = self.layer4(out)
#         self.mus.append(out[0])
#         self.sigmas.append(out[2])
#         if use_diffusion:
#             self.scales.append(out[2].max().detach().data.item())
#         else:
#             self.scales.append(0)

#         out = out[0]
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)

#         return out
    
#     def forward(self, x, use_diffusion=True):
#         if self.training:
#             #print('training........')
#             out = self.net(x, use_diffusion=use_diffusion)
#             out = F.log_softmax(out, dim=1)
#         else:
#             #print('evaling........')
#             if use_diffusion:
#                 proba = 0 
#                 for k in range(10):  
#                     out = self.net(x, use_diffusion=True)
#                     p = F.softmax(out, dim=1)
#                     proba = proba + p
#                 out = ((proba/10)+1e-20).log() # next nll
#             else:
#                 out = self.net(x, use_diffusion=False)
#                 out = F.log_softmax(out, dim=1)
#         return out

# def pdeadd_resnet(name, num_classes=10, pretrained=False, device='cpu'):
#     """
#     Returns suitable Resnet model from its name.
#     Arguments:
#         name (str): name of resnet architecture.
#         num_classes (int): number of target classes.
#         pretrained (bool): whether to use a pretrained model.
#         device (str or torch.device): device to work on.
#     Returns:
#         torch.nn.Module.
#     """
#     if name == 'resnet-18':
#         return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, device=device)
#     elif name == 'resnet-34':
#         return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, device=device)
    
#     raise ValueError('Only resnet18, resnet34, resnet50 and resnet101 are supported!')

