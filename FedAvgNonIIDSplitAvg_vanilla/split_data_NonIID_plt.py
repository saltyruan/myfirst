import torch
from torchvision import datasets
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt
import time
import split_noniid
import utils

torch.manual_seed(42)

if __name__ == "__main__":

    N_CLIENTS = 10 
    DIRICHLET_ALPHA = 1.0

    dir = "../data/"

#     train_data = datasets.EMNIST(root="../data/", split="byclass", download=True, train=True)
#     test_data = datasets.EMNIST(root="../data/", split="byclass", download=True, train=False)
    
    transform_train = transform.Compose([  # 数据增强操作，训练集的预处理
            transform.RandomCrop(32, padding=4),  # 随机剪裁，大小为32*32，添加4个像素的填充内容
            transform.RandomHorizontalFlip(),  # 随机垂直方向的翻转
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化操作，数值是抽样得到的，无需考虑太多，分别是均值和标准差
        ])
    transform_test = transform.Compose([  # 对测试集进行预处理
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    train_data = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)  # 获得训练集
    eval_data = datasets.CIFAR10(dir, train=False, transform=transform_test)  # 获得测试集


    n_channels = 1


    input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes)


    train_labels = np.array(train_data.targets)

    # # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    # client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    client_idcs = split_noniid.pathological_non_iid_split(train_data, 10, 10, 2)

    # 

    a_ndarray = np.array(train_data.targets)

    # 每个 client 数量
    for i in range(len(client_idcs)):
        print(i)
        print(len(client_idcs[i]))
        # 统计每个client类别数量
        utils.count_every_sum(a_ndarray[client_idcs[i]])

    # 展示不同client的不同label的数据分布
    plt.figure(figsize=(20,3))
    plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), train_data.classes)
    curTime = time.strftime('%Y%m%d_%H%M%S');
    plt.savefig("./CIFAR10_label_distribution_{}.jpg".format(curTime))
    plt.savefig
    plt.show()