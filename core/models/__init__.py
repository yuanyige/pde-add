import torch
import torch.nn as nn
import torch.nn.functional as F

from .cifar10.standard_resnet import standard_resnet as standard_resnet_cifar10
from .cifar10.fixdiff_resnet import fixdiff_resnet as fixdiff_resnet_cifar10
from .cifar10.ladiff_resnet import ladiff_resnet as ladiff_resnet_cifar10

from .cifar10.ladiff_preresnet import ladiff_preresnet as ladiff_preresnet_cifar10

from .cifar10.standard_wideresnet import standard_wideresnet as standard_wideresnet_cifar10
from .cifar10.fixdiff_wideresnet import fixdiff_wideresnet as fixdiff_wideresnet_cifar10
from .cifar10.ladiff_wideresnet import ladiff_wideresnet as ladiff_wideresnet_cifar10

from .cifar100.standard_resnet import standard_resnet as standard_resnet_cifar100

def create_model(data, backbone, protocol):

    net = backbone.split('-')[0]
    if 'fixdiff' in protocol: 
        protocol, sigma = protocol.split('-')
    if 'ladiff' in protocol: 
        protocol, _ = protocol.split('-')

    model_name = "{}_{}_{}".format(protocol, net, data)
    print("using name: {}..".format(model_name))
    func = eval(model_name)

    if 'fixdiff' in protocol: 
        return func(backbone, float(sigma))
    else:
        return func(backbone)
    

#print(load_model('wideresnet-16-4','standard'))