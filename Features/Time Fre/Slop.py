from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
from scipy.optimize import leastsq
import os


# 查找文件夹下所有dat文件的路径
def fileListFunc(filePath):
    fileList = []
    for top, dirs, nondirs in os.walk(filePath):
        for item in nondirs:
            if item.endswith(".dat"):
                fileList.append(os.path.join(top, item))
    return fileList


def load_origin_data(filename, channel):
    signal = np.fromfile(filename, dtype=np.int16)
    N = len(signal) // 12
    signal = signal.astype(float) / 13107.0
    signal = np.reshape(signal, [N, 12])
    signal_channel = signal[:, channel]
    return signal_channel


def FSY(single):
    # 读取EHG信号
    f, t, Zxx = signal.stft(single,nperseg=300)

    return f, t, Zxx


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


# 水平视觉图算法time_series表示的是时间序列
def HVG(Fre_val):
    pair_Node = []
    for i in range(len(Fre_val) - 2):
        pair_Node.append([i, i + 1])
        if Fre_val[i + 1] < min(Fre_val[i], Fre_val[i + 2]):
            pair_Node.append([i, i + 2])
        j = i + 3
        if j < len(Fre_val):
            max_index = np.argmax(np.array(Fre_val[j:])) + j
            for h in range(j, max_index + 1):
                if max(Fre_val[i + 1:h]) < min(Fre_val[i], Fre_val[h]):
                    pair_Node.append([i, h])
    return pair_Node


# 计算并绘制STFT的大小
# fs:采样频率；
# nperseg： 每个段的长度，默认为256
# 最好选用821样例
fileList = fileListFunc("D:\\Kinpeng_Zhang\\EHG_Signal\\tpehgdb")
file_write = open("E:\\Fre_domian\\Third_Channel_Fre_Slop" + ".txt", "a")
for filename in fileList:
    file_write.write(filename.split("\\")[-1].split(".")[0] + "    ")
    single = load_origin_data(filename, 10)
    f, t, Zxx = FSY(single)
    for i in range(len(f)):
        Node_pair = HVG(np.abs(Zxx[i]))
        dict_Node, count_dict, total_Edge, Degree_count = Edge_Count(Node_pair)
        Xi, Yi = [], []
        log_Data = Degree_Rate(Degree_count, total_Edge)
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
        file_write.write(str('%.9f' % k) + "    ")
    file_write.write("\n")
