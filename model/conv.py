from collections import OrderedDict

import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import random

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class stjm_block(nn.Module):
    def __init__(self, in_Channel, out_Channel, graph,temporalSlide=7,dilation=1,TStride=1,residual=True,adaptive=True):
        super().__init__()
        self.out_Channel = out_Channel
        self.graph = graph
        self.adaptive=adaptive
        self.activate=nn.ReLU()

        self.conv=nn.ModuleList()
        padding = int(temporalSlide / 2)
        self.global_conv=Depthwise_Separable_Conv(in_Channel,out_Channel,temporalSlide,dilation,TStride)
        # self.global_conv= nn.Conv2d(in_Channel, out_Channel, kernel_size=(temporalSlide, 3), stride=(TStride, 3),
        #                               padding=(padding,0), padding_mode="zeros")
        for i in range(graph.STJMArray.shape[0]):
            self.conv.append(
                Depthwise_Separable_Conv(in_Channel, out_Channel, temporalSlide, dilation, TStride)
                # nn.Conv2d(in_Channel, out_Channel, kernel_size=(temporalSlide, 3), stride=(TStride, 3),
                #                       padding=(padding,0), padding_mode="zeros")
            )

        self.bn = nn.BatchNorm2d(out_Channel)

        self.adaptive=adaptive
        if adaptive:
            self.STJMArray = nn.Parameter(graph.STJMArray, requires_grad=False)+1e-4
            self.globalGraph=nn.Parameter(torch.zeros(graph.STJMArray[0].shape),requires_grad=True)+1e-2
        else:
            self.STJMArray=Variable(torch.from_numpy(graph.STJMArray.astype(np.float32)), requires_grad=False)
        self.weight=nn.Parameter(torch.ones(self.STJMArray.shape[0]+1),requires_grad=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_Channel == out_Channel and TStride==1:
            self.residual= nn.Sequential(
                nn.Identity(),
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_Channel, out_Channel, kernel_size=1,stride=(TStride,1)),
            )

    def forward(self, input):  # N C T V
        device=input.device
        residual=self.residual(input)

        globalGraph=self.globalGraph.to(device)
        STJMArray=self.STJMArray.to(device)

        if self.adaptive:
            output = self.global_conv(torch.matmul(input, globalGraph))*self.weight[0]
            for i in range(self.STJMArray.shape[0]):
                output+=self.conv[i](torch.matmul(input,STJMArray[i]))*self.weight[i+1]
        else:
            input=torch.matmul(input,self.STJMArray)
            output = self.conv(input)

        output = self.activate(self.bn(output+ residual)).to(device)
        return output

class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, temporalSlide=7,dilation=1,TStride=1,*args, **kwargs):
        super().__init__()
        self.padding = int(temporalSlide / 2) * dilation
        self.Depthwise_Conv = nn.Conv2d(in_channels, in_channels, kernel_size=(temporalSlide, 3), stride=(TStride, 3),
                                        padding=(self.padding, 0), padding_mode="zeros", groups=in_channels,
                                        dilation=(dilation, 1))
        self.Pointwise_Conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, input):
        out = self.Depthwise_Conv(input)
        out = self.Pointwise_Conv(out)
        return out

class Basic_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, graph, temporalSlide=7, dilation=1, TStride=1, residual=True, adaptive=True):
        super(Basic_Unit, self).__init__()
        self.gcn = unit_gcn(in_channels, out_channels, graph.A, adaptive=adaptive)
        self.stjm=stjm_block(in_channels, out_channels, graph, temporalSlide, dilation)
        self.tcn = unit_tcn(out_channels, out_channels, stride=TStride)
        self.activate = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (TStride == 1):
            self.residual = nn.Sequential(
                nn.Identity(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Sequential(
                unit_tcn(in_channels, out_channels, kernel_size=1, stride=TStride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        device=x.device
        y = self.activate(self.tcn(self.stjm(x)+self.gcn(x)) + self.residual(x)).to(device)
        return y

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        if adaptive:
            self.PA = torch.from_numpy(A.astype(np.float32))
            order,subset,V,V=self.PA.shape
            self.PA = torch.reshape(self.PA,(order*subset,V,V))
            self.PA=nn.Parameter(self.PA,requires_grad=False)
            self.globalA=torch.zeros(A.shape[0],requires_grad=True)
        else:
            self.PA = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.PA.shape[0]+1):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.weight=nn.Parameter(torch.ones(self.PA.shape[0]+1),requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.PA.cuda(x.get_device())
        for i in range(self.PA.shape[0]):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            if i <A.shape[0]+1:
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            else:
                z = self.conv_d[i](torch.matmul(A2, self.globalA).view(N, C, T, V))
            y = z*self.weight[i] + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.activate(y)
        return y
