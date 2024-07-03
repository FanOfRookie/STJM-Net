import matplotlib.pyplot as plt
import numpy as np

stjm_net_dir = '../work_dir/ntu/xsub/GGCNMixGCN_T64_mixup_Separable_order2_modify/joint/runs-80-50080_right.txt'
baseline_dir = '../work_dir/ntu60/xsub/STGCN_ADP/joint/runs-41-12833_right.txt'

actions_dir = '../data/ntu/actions.txt'
actions = []
with open(actions_dir, "r") as file:
    for line in file:
        line = line.rstrip('\n')
        actions.append(line)

classes = actions.__len__()

# 读取数据
with open(stjm_net_dir, "r") as file:
    class_count = np.zeros([classes])
    right_count = np.zeros([classes])
    # 逐行读取文件内容
    for line in file:
        # 对每一行进行处理，这里只是打印出来，你可以根据需要进行其他操作
        data = line.strip().split(',')
        class_count[int(data[1]) - 1] += 1
        if(int(data[1]) == int(data[0])):
            right_count[int(data[1]) - 1] += 1

stjm_per_class_acc = np.zeros([classes])
for i in range(classes):
    stjm_per_class_acc[i] = right_count[i] / class_count[i]


with open(baseline_dir, "r") as file:
    class_count = np.zeros([classes])
    right_count = np.zeros([classes])
    # 逐行读取文件内容
    for line in file:
        # 对每一行进行处理，这里只是打印出来，你可以根据需要进行其他操作
        data = line.strip().split(',')
        class_count[int(data[1]) - 1] += 1
        if(int(data[1]) == int(data[0])):
            right_count[int(data[1]) - 1] += 1
baseline_per_class_acc = np.zeros([classes])
for i in range(classes):
    baseline_per_class_acc[i] = right_count[i] / class_count[i]


# 文件读取完毕后，自动关闭文件

gruop_label = ['Diff Acc']
gruop_color = ['blue']
y_list = [stjm_per_class_acc - baseline_per_class_acc]

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 1,figsize=(40,20))
# 画三次，每次都在x轴上偏斜一定距离画
width = 0.8
x = np.arange(1,61)
for index in range(len(y_list)):
    bar = ax.bar(x + index * width, y_list[index], edgecolor="black", label=gruop_label[index], width=width,
                 color=gruop_color[index])

    # ax.bar_label(bar, label_type="edge")
ax.legend(fontsize=24,ncol = gruop_label.__len__())
plt.xticks(x,actions)
# 设置x轴刻度文字竖排显示
plt.xticks(fontsize=30)
plt.setp(plt.gca().get_xticklabels(), rotation=70, ha='right')
# plt.tick_params(axis='x', width=2, length=10)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.savefig('diff_acc.png')
plt.show()