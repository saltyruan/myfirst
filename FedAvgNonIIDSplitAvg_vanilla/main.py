import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import random
import time

import torch
import dataset
from server import Server
from client import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import numpy as np 
import copy

#按照interval取list,[)
def batch_list(len_data_list, interval):
    res_list = []
    for i in range(0,len_data_list,interval):
        if(i % interval == 0):
            begin,end = i, min(i + interval, len_data_list)
            res_list.append((begin,end))
    return res_list


# 获取位于[begin, end)区间的子列表
def get_block_i(data_list, interval):
    begin, end = interval
    return data_list[begin: end]

def search(all_weights,all_choose_k,random_index_k,start_index,now_path,res,server,cal_num):
    if(start_index>=len(random_index_k)):
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        #计算client的中心_1 求和
        for i in range(len(all_weights)):
            if(len(all_weights[i]))!=0:#如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    weight_accumulator[name] += params
        #计算client的中心_2 除数量
        for name, params in weight_accumulator.items():
            weight_accumulator[name]  = weight_accumulator[name]/cal_num
        #计算client中心到各client的距离
        temp_res = 0
        for i in range(len(all_weights)):
            if(len(all_weights[i]))!=0:#如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    temp_res += np.linalg.norm((weight_accumulator[name]-params).cpu())#tensor转numpy        
        if(temp_res<res):
            print("出现")
            for i in range(len(now_path)):
                all_choose_k[i] = now_path[i] 
            #all_choose_k = copy.copy(now_path) #注意这里不能这样写
            res = temp_res
        return 
    for every in range(len(all_weights[random_index_k[start_index]])):
        now_path[random_index_k[start_index]] = every
        search(all_weights,all_choose_k,random_index_k,start_index+1,now_path,res,server,cal_num)
    

if __name__ == '__main__':
    # 存储容器，用于绘制图像
    accs = []  # 存放准确率
    losses = []  # 存放损失

    # 载入配置文件
    with open('config.json', 'r') as f  :
        conf = json.load(f)

    train_datasets, eval_datasets, client_idcs = dataset.get_dataset('../data/', conf['type'])  # 获取训练数据和测试数据
    server = Server(conf, eval_datasets)  # 创建服务器
    clients = []  # 客户端列表
    all_client_num = conf['no_models']
    memory_k = conf['memory_k']
    warmup  = conf['warmup']
    batch_client_num = conf['batch_client_num']
    all_weights = [[] for i in range(all_client_num)] #全局缓存表，缓存的是client的memory_k个绝对参数
    
    all_choose_k = [0 for i in range(all_client_num)] #默认选择的是第一个
    
    for c in range(all_client_num):  # 创建客户端
        clients.append(Client(conf, server.global_model, train_datasets, client_idcs[c], c))
    
    oldTime = time.strftime('%Y%m%d_%H%M%S')
    print("starttime")
    print(oldTime)
    for e in tqdm(range(conf['global_rounds'])):  # 进行全局轮数
        cal_num = all_client_num #每轮次要计算的client的个数，从最大值开始减回去
        random_index = [i for i in range(all_client_num)]# 从0开始
        random_index_k = random.sample(random_index, conf['k'])  # 随机选取k个index
        candidates = [clients[index] for index in random_index_k] # 随机选取k个客户端
        #candidates = random.sample(clients, conf['k'])  # 随机选取k个客户端
        round_all_weights = [{} for i in range(all_client_num)] #本轮缓存表，缓存的是相对参数
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # 初始化上面的参数字典，大小和全局模型相同
            
        for c_index in range(all_client_num):
            if c_index in random_index_k:
                c = clients[c_index]
                client_res = c.local_train(server.global_model)  # 进行本地训练并计算差值字典
                weight_temp = {} #该本地模型变化量
                for name, params in server.global_model.state_dict().items():
                    weight_temp[name] = client_res[name] - params
                client_temp = copy.deepcopy(client_res)
                #记录全局
                if(len(all_weights[c_index])>=memory_k):
                    del all_weights[c_index][0] #LRU删除
                    all_weights[c_index].append({}) #添加当前
                    for name, data in server.global_model.state_dict().items():
                        #最后一个应该是memory_k-1
                        all_weights[c_index][memory_k-1][name] = client_temp[name] 
                else:
                    all_weights[c_index].append({}) #添加当前
                    for name, data in server.global_model.state_dict().items():  
                        all_weights[c_index][len(all_weights[c_index])-1][name] = client_temp[name]  # 记录全局缓存表
                round_all_weights[c_index] = weight_temp #记录本地缓存表模型变换量，这时候只是给一个初值
            else:
                if(len(all_weights[c_index])!=0):
                    #记录未选中的部分
                    for name, data in server.global_model.state_dict().items():
                        weight_accumulator[name].add_(all_weights[c_index][all_choose_k[c_index]][name] - data)
                else:
                    cal_num = cal_num - 1 
                # for name, data in server.global_model.state_dict().items():  # 计算差异
                #     if(len(all_weights[c_index])!=0): #此时该client被选中过，但本轮没有被选中
                #         diff[name] = all_weights[c_index][all_choose_k[c_index]][name] - data
                #     else:
                #         #此时该client被选中过，但本轮没有被选中
                #         #此时还没算到这个client,把它减去
                #         cal_num = cal_num - 1
                #         diff[name] = torch.zeros_like(data)
                # round_all_weights[c_index] = diff
                # #记录没有选中的部分
                # for name, params in server.global_model.state_dict().items():
                #     weight_accumulator[name].add_(round_all_weights[c_index][name])
        if e < warmup:
            #此时warmup阶段，用策略一
            #这里可以只改被选中的
            for i in range(all_client_num):
                if(len(all_weights[i])!=0):
                    all_choose_k[i] = len(all_weights[i])-1
        else:
            #更新:分段搜索
            #搜索函数，寻找这个时候的被选中的client应该选中的model,即最后返回的结果在all_choose_k中体现
            random_split = batch_list(len(random_index_k),batch_client_num)

            for i in range(len(random_split)):
                # 获取第i块batch
                b, e = random_split[i]
                batch_i = get_block_i(random_index_k, (b, e))
                now_path = copy.deepcopy(all_choose_k)
                result = 99999999999999999
                #search(all_weights,all_choose_k,random_index_k,0,now_path,result,server,cal_num)
                search(all_weights,all_choose_k,batch_i,0,now_path,result,server,cal_num)
        
        #random 搜索
        for random_c_index in random_index_k:
            for name, data in server.global_model.state_dict().items():  
                round_all_weights[random_c_index][name] = all_weights[random_c_index][all_choose_k[random_c_index]][name] - server.global_model.state_dict()[name] # 记录本地缓存表模型变换量
            #记录选中的部分
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(round_all_weights[random_c_index][name])
                
        #server.model_aggregrate(weight_accumulator)  # 模型聚合
        server.model_aggregrate_new(weight_accumulator,cal_num)
        acc, loss = server.model_eval()  # 进行全局模型测试
        accs.append(acc)
        losses.append(loss)
        print('全局模型：第{}轮完成！准确率：{:.2f} loss: {:.2f}'.format(e, acc, loss))

    # 将准确率信息存储在txt文件中用于绘图
    curTime = time.strftime('%Y%m%d_%H%M%S');
    print("finish time")
    print(curTime)
    plt.plot([i for i in range(len(accs))], accs, label='Acc')
    plt.legend()
    plt.xlabel('Global Rounds')
    plt.ylabel('Accuracy')
    plt.title("Our_method_2 random_search warmup50 0.0001,Drichlet alpha=0.1 10/10")
    plt.savefig("./results/{}.jpg".format(curTime))
    plt.show()
    df = pd.DataFrame([accs, losses])  # 计入表格
    df.to_csv("./results/data_{}.csv".format(curTime))  # 
    print("持续时间:")
    print(curTime-oldTime)
