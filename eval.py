import torch
from model.conv import Model
from thop import profile
from graph.ntu_rgb_d import Graph
dict = {"mode": "Partitioning", "K": 2}
model = Model(num_class=60, num_point=25, num_person=2, graph=Graph, graph_args=dict, in_channels=3,
                 drop_out=0.3, num_set=3,residual=True)
checkpoint=torch.load('/root/autodl-tmp/CTR-GCN/CTR-GCN-main/work_dir/ntu/xsub/GGCNMixGCN_T64_mixup_Separable_order2_modify/joint/runs-39-24414.pt')
model.load_state_dict(checkpoint)
# model.train()

input = torch.randn(1, 3, 64, 25,2)
flops, params = profile(model, inputs=(input, ))
print('flops:{}'.format(flops))
print('params:{}'.format(params))