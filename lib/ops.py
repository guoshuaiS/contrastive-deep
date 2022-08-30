import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math
from lib import contrastive_loss
from lib import norm_sigma
#input:输入图像
#target:重建图像
class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return 0.5 * torch.mean((input-target)**2)

class BCELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(target*torch.log(torch.clamp(input, min=1e-10))+
            (1-target)*torch.log(torch.clamp(1-input, min=1e-10)), 1))
class my_loss(nn.Module):
    def __init__(self,alpha=0,beta=0.1):
        super(self.__class__, self).__init__()
        self.alpha=alpha#低秩的配比系数
        self.beta=beta#对比损失的配比系数

    def forward(self, input, target):
        #引入对比损失
        instance_temperature = 0.5  # 这个可以调，有的文章说0.1好，有的设置0.5
        if torch.cuda.is_available():
            loss_device = torch.device("cuda")
        else:
            loss_device = torch.device("cpu")
        criterion_instance = contrastive_loss.InstanceLoss(instance_temperature, loss_device).to(
            loss_device)
        #loss1 = norm_sigma.norm_sigma(input - target, 0.1)
        n=input.size(0)#获取样本个数
        loss1 = torch.mean((input-target)**2)
        #loss2 = torch.nuclear_norm(target)/n
        loss3 = criterion_instance(input, target)  #
        return loss1+self.beta * loss3
        #return 0.01*loss3

