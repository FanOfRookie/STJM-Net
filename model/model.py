from collections import OrderedDict

import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from .conv import *
import math
import random

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0.3, num_set=3,residual=True):
        super(Model, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_channels=in_channels
        self.num_set=num_set

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.graph.STJMArray=Variable(torch.from_numpy(self.graph.STJMArray.astype(np.float32)).to(device), requires_grad=False)
        # self.graph.STJMArray=self.graph.STJMArray
        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.layers=nn.Sequential(OrderedDict([
            ('layer0', Basic_Unit(in_channels=3, out_channels=64, graph=self.graph)),
            ('layer1', Basic_Unit(in_channels=64, out_channels=64,graph=self.graph,dilation=2)),
            ('layer2', Basic_Unit(in_channels=64, out_channels=64,graph=self.graph)),
            ('layer3', Basic_Unit(in_channels=64, out_channels=64,graph=self.graph,dilation=3)),
            ('layer4', Basic_Unit(in_channels=64, out_channels=128,TStride=2,graph=self.graph)),
            ('layer5', Basic_Unit(in_channels=128, out_channels=128,graph=self.graph,dilation=4)),
            ('layer6', Basic_Unit(in_channels=128, out_channels=128, graph=self.graph)),
            ('layer7', Basic_Unit(in_channels=128, out_channels=256,TStride=2, graph=self.graph, dilation=5)),
            ('layer8', Basic_Unit(in_channels=256, out_channels=256, graph=self.graph)),
            ('layer9', Basic_Unit(in_channels=256, out_channels=256, graph=self.graph)),
        ]))

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x=self.layers(x)

        # N*M,C,T,V
        c_new = x.shape[1]
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
