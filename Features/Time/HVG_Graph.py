import numpy as np
import os
from Time_feature.Graph_Al import *
import math
from scipy.optimize import leastsq

# 查找文件夹下所有dat文件的路径
def fileListFunc(filePath):
    fileList = []
    for top, dirs, nondirs in os.walk(filePath):
        for item in nondirs:
            if item.endswith(".dat"):
                fileList.append(os.path.join(top, item))
    return fileList


# 根据文件路径解析EHG信号,去除前后3分钟的信号点
def load_origin_data(filename, channel):
    signal = np.fromfile(filename, dtype=np.int16)
    N = len(signal) // 12
    signal = signal.astype(float) / 13107.0
    signal = np.reshape(signal, [N, 12])
    sig = signal[:, channel]
    sig = sig[6000:]
    sig = sig[:-6000]
    return sig


# signal表示原始信号，step表示步长
def max_by_step(signal, step):
    re = []
    remainder = len(signal) % step
    for idx in range(0, len(signal[0:len(signal) - remainder]), step):
        signal_EHG = []
        for i in range(step):
            signal_EHG.append(signal[idx + i])
        re.append(signal_EHG)
    result_max=[]
    for i in re:
        result_max.append(max(i))
    return result_max


# 水平视觉图算法time_series表示的是时间序列
def HVG(Fre_val):
    pair_Node = []
    for i in range(len(Fre_val) - 2):
        pair_Node.append([i, i + 1])
        if Fre_val[i + 1] <min(Fre_val[i], Fre_val[i + 2]):
            pair_Node.append([i, i + 2])
        j = i + 3
        if j < len(Fre_val):
            max_index = np.argmax(np.array(Fre_val[j:])) + j
            for h in range(j, max_index + 1):
                if max(Fre_val[i + 1:h]) < min(Fre_val[i], Fre_val[h]):
                    pair_Node.append([i, h])
    return pair_Node


'''
输入的是无向图中的边，
dict_Node输出的是每个节点对应的边的另一个节点，
count_dict输出的是每个节点边的个数，
total_Edge输出的是无向图中每个节点对应的变得总数，
Degree_count表示的是无向图中度的分布。
'''


def sum_by_step(signal, step):
    re = []
    all_avg = []
    remainder = len(signal) % step
    for idx in range(0, len(signal[0:len(signal) - remainder]), step):
        signal_EHG = []
        for i in range(step):
            signal_EHG.append(signal[idx + i])
        re.append(signal_EHG)
    for i in re:
        row_sum = 0
        for j in i:
            row_sum = row_sum + j
        all_avg.append(row_sum / len(i))
    return all_avg

'''
输入的是无向图中的边，
dict_Node输出的是每个节点对应的边的另一个节点，
count_dict输出的是每个节点边的个数，
total_Edge输出的是无向图中每个节点对应的变得总数，
Degree_count表示的是无向图中度的分布。
'''


def Edge_Count(list_Node):
    dict_Node = {}
    for i in list_Node:
        if i[0] not in dict_Node.keys():
            dict_Node[i[0]] = []
            dict_Node[i[0]].append(i[1])
        else:
            dict_Node[i[0]].append(i[1])
        if i[1] not in dict_Node.keys():
            dict_Node[i[1]] = []
            dict_Node[i[1]].append(i[0])
        else:
            dict_Node[i[1]].append(i[0])
    count_dict = {}
    total_Edge = 0
    for j in dict_Node:
        if j not in count_dict:
            count_dict[j] = len(dict_Node[j])
            total_Edge += len(dict_Node[j])
    Degree_count = {}
    for k in count_dict:
        if count_dict[k] not in Degree_count:
            Degree_count[count_dict[k]] = 1
        else:
            Degree_count[count_dict[k]] += 1
    return dict_Node, count_dict, total_Edge, Degree_count


def Degree_Rate(Degree_count, total_Edge):
    log_value = {}
    for i in Degree_count:
        if i != 1:
            log_value[i] = math.log(i * Degree_count[i] / total_Edge, np.e)
    return log_value


# 需要拟合的函数func :指定函数的形状
def func(p, x):
    k, b = p
    return k * x + b


# 偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p, x, y):
    return func(p, x) - y


def slop_feature(Node_pair):
    dict_Node, count_dict, total_Edge, Degree_count = Edge_Count(Node_pair)
    Xi, Yi = [], []
    log_Data = Degree_Rate(Degree_count, total_Edge)
    if len(log_Data) > 1:
        for i in log_Data:
            Xi.append(i)
            Yi.append(log_Data[i])
        Xi = np.array(Xi)
        Yi = np.array(Yi)
        # k,b的初始值，可以任意设定,经过几次试验：Para[1]
        p0 = [1, 20]
        # 把error函数中除了p0以外的参数打包到args中(使用要求)
        Para = leastsq(error, p0, args=(Xi, Yi))
        k, b = Para[0]
        return k


# filename表示每一个样本的路径和名称
fileList = fileListFunc("D:\\Kinpeng_Zhang\\EHG_Signal\\tpehgdb")
f = open("E:\\Fre_domian\\Time\\Thrid_Channel_Graph_Features" + ".txt", "a+")
for filename in fileList:
    preprocessing.MinMaxScaler()
    signal = min_max_scaler.fit_transform(load_origin_data(filename, 10)，5)
    signal = np.array(signal))
    # 选取三通道的过滤信号0.3HZ-3HZ信号
    li = np.zeros((len(sum_signal), len(sum_signal)))
    Node_pair = HVG(sum_signal)
    slop=slop_feature(Node_pair)
    for i in Node_pair:
        li[i[0]][i[1]] = 1
        li[i[1]][i[0]] = 1
    density = density_und(li)
    clustering_coef = clustering_coef_bu(np.array(li))
    assortativity = assortativity_bin(np.array(li))
    f.write(filename.split("\\")[-1].split(".")[0] +"     " + str('%.9f' % slop)+"     " + str('%.9f' % density) + "     " + str(
        '%.9f' % clustering_coef) + "     " + str('%.9f' % assortativity) + "\n")
    f.flush()
f.close()
