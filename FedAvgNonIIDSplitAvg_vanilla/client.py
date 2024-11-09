import torch
import copy


class Client(object):
    def __init__(self, conf, model, train_dataset, train_dataset_idcs, id=1):
        self.conf = conf  # 配置信息
        self.local_model = copy.deepcopy(model)  # 本地模型
        self.client_id = id  # 客户端id
        self.train_dataset = train_dataset  # 训练集
        # all_range = list(range(len(self.train_dataset)))  # 获得整个未分割训练数据集的下标
        # data_len = int(len(self.train_dataset) / self.conf['no_models'])  # 计算本地数据集长度
        # indices = all_range[id * data_len : (id + 1) * data_len]  # 切分该客户端对应的数据部分
        # self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf['batch_size'],sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        # 训练集下标
        self.train_dataset_idcs = train_dataset_idcs
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf['batch_size'],sampler=torch.utils.data.sampler.SubsetRandomSampler(self.train_dataset_idcs))

        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
        # 训练数据加载器

    def local_train(self, model):
        # print("100000000000000000 原localmodel")
        # print(self.local_model.state_dict()['conv1.weight'][0][0][0])
        # print("200000000000000000 原globalmodel")
        # print(model.state_dict()['conv1.weight'][0][0][0])
        for name, param in model.state_dict().items():  # 获得全局模型参数
            self.local_model.state_dict()[name].copy_(param.clone())  # 复制全局参数到本地
        # print("11111111111111111 复制 global 后的 localmodel")
        # print(self.local_model.state_dict()['conv1.weight'][0][0][0])
        # 定义优化器
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.local_model.train()  # 标记为训练模式，参数可以改变

        for e in range(self.conf['local_epochs']):  # 本地轮数
            for batch_id, bach in enumerate(self.train_loader):  # 按batch加载训练数据
                data, target = bach  # 获得本batch数据和标签
                if torch.cuda.is_available():  # 如果GPU可用
                    data, target = data.cuda(), target.cuda()  # 放在GPU计算
                optimizer.zero_grad()  # 优化器置零
                output = self.local_model(data)  # 获得预测结果
                loss = torch.nn.functional.cross_entropy(output, target)  # 获得预测损失
                loss.backward()  # 进行反向传播
                optimizer.step()
            print('本地模型{}完成第{}轮训练'.format(self.client_id, e))  # 打印目前训练进度
        # print("2222222222222222 训练后的 localmodel")
        # print(self.local_model.state_dict()['conv1.weight'][0][0][0])
        # print("2222222222222222 训练后查看 globalmodel")
        # print(model.state_dict()['conv1.weight'][0][0][0])
        # print("作差")
        # print(self.local_model.state_dict()['conv1.weight'][0][0][0] - model.state_dict()['conv1.weight'][0][0][0])
        diff = dict()  # 计算参数差异的容器
        for name, data in self.local_model.state_dict().items():  # 给原值
            diff[name] = data
        # print("==========================")

        return diff

if __name__=="__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
