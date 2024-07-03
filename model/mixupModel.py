import torch
import torch.nn as nn
import numpy as np

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot

index_spine=[4,3,21,2,1]
index_spine=[ i-1 for  i in index_spine]
mask_spine=torch.zeros(25)
mask_spine[index_spine]=1

index_lefthand=[24,25,12,11,10,9]
index_lefthand=[ i-1 for  i in index_lefthand]
mask_lefthand=torch.zeros(25)
mask_lefthand[index_lefthand]=1

index_righthand=[22,23,8,7,6,5]
index_righthand=[ i-1 for  i in index_righthand]
mask_righthand=torch.zeros(25)
mask_righthand[index_righthand]=1

index_leftfoot=[20,19,18,17]
index_leftfoot=[ i-1 for  i in index_leftfoot]
mask_leftfoot=torch.zeros(25)
mask_leftfoot[index_leftfoot]=1

index_rightfoot=[16,15,14,13]
index_rightfoot=[ i-1 for  i in index_rightfoot]
mask_rightfoot=torch.zeros(25)
mask_rightfoot[index_rightfoot]=1

mask=torch.stack((mask_spine,mask_lefthand,mask_righthand,mask_leftfoot,mask_rightfoot)).cuda()

CELoss = nn.CrossEntropyLoss()

class ManifoldMixupModel(nn.Module):
    def __init__(self, model, criterion,num_classes=10,persons=2 ,alpha=1):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        self.persons=persons
        ##选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self.module_list = []
        for n, m in self.model.named_modules():
            # if 'conv' in n:
            if n[:-1] == 'layer':
                self.module_list.append(m)
        self.criterion=criterion

    def forward(self, x, target=None):
        if target == None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                self.lam = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()

            loss=self.lam*self.criterion(out,target)+(1-self.lam)*self.criterion(out,target[self.indices])
            return out, loss

    def hook_modify(self, module, input, output):
        N,C,T,V=output.shape
        output=output.view(-1,self.persons,C,T,V)
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        output = output.view(-1, C, T, V)
        return output
