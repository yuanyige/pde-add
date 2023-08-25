import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard_resnet import standard_resnet 
from .pdeadd_resnet import pdeadd_resnet

def create_model(backbone, protocol, num_classes):
    net = backbone.split('-')[0]
    model_name = "{}_{}".format(protocol, net)
    print("using name: {}..".format(model_name))
    func = eval(model_name)
    return func(name=backbone, num_classes=num_classes)
