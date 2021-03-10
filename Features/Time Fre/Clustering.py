from scipy import signal
import numpy as np
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
    f, t, Zxx = signal.stft(single, nperseg=300)

    return f, t, Zxx


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


def clustering_coef_bu(G):  # 聚类系数
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    n = len(G)
    C = np.zeros((n,))

    for u in range(n):
        V, = np.where(G[u, :])
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)]
            C[u] = np.sum(S) / (k * k - k)
    average = np.average(C)
    return average


# 计算并绘制STFT的大小
# fs:采样频率；
# nperseg： 每个段的长度，默认为256
# 最好选用821样例
fileList = fileListFunc("D:\\Kinpeng_Zhang\\EHG_Signal\\tpehgdb")
file_write = open("E:\\Fre_domian\\Third_Channel_Fre_clustering" + ".txt", "a")
Fre_list = []
for filename in fileList:
    file_write.write(filename.split("\\")[-1].split(".")[0] + "    ")
    single = load_origin_data(filename, 10)
    f, t, Zxx = FSY(single)
    for i in range(len(f)):
        Fre_list.append(f[i])
        li = np.zeros((len(Zxx[i]), len(Zxx[i])))
        Node_pair = HVG(np.abs(Zxx[i]))
        for i in Node_pair:
            li[i[0]][i[1]] = 1
            li[i[1]][i[0]] = 1
        va = clustering_coef_bu(li)
        file_write.write(str('%.9f' % va) + "    ")
    file_write.write("\n")
