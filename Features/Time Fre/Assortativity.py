from scipy import signal
import numpy as np
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
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


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def degrees_und(CIJ):
    '''
    Node degree is the number of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    deg : Nx1 np.ndarray
        node degree

    Notes
    -----
    Weight information is discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    return np.sum(CIJ, axis=0)


def degrees_dir(CIJ):
    '''
    Node degree is the number of links connected to the node. The indegree
    is the number of inward links and the outdegree is the number of
    outward links.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed binary/weighted connection matrix

    Returns
    -------
    id : Nx1 np.ndarray
        node in-degree
    od : Nx1 np.ndarray
        node out-degree
    deg : Nx1 np.ndarray
        node degree (in-degree + out-degree)

    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
           Weight information is discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    id = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    od = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ
    deg = id + od  # degree = indegree+outdegree
    return id, od, deg


def assortativity_bin(CIJ, flag=0):  # 同配系数
    '''
    The assortativity coefficient is a correlation coefficient between the
    degrees of all nodes on two opposite ends of a link. A positive
    assortativity coefficient indicates that nodes tend to link to other
    nodes with the same or similar degree.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary directed/undirected connection matrix
    flag : int
        0 : undirected graph; degree/degree correlation
    Returns
    -------
    r : float
        assortativity coefficient

    Notes
    -----
    The function accepts weighted networks, but all connection
    weights are ignored. The main diagonal should be empty. For flag 1
    the function computes the directed assortativity described in Rubinov
    and Sporns (2010) NeuroImage.
    '''
    if flag == 0:  # undirected version
        deg = degrees_und(CIJ)
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        degi = deg[i]
        degj = deg[j]
    else:
        print("error")

    # compute assortativity
    term1 = np.sum(degi * degj) / K
    term2 = np.square(np.sum(.5 * (degi + degj)) / K)
    term3 = np.sum(.5 * (degi * degi + degj * degj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r


# 计算并绘制STFT的大小
# fs:采样频率；
# nperseg： 每个段的长度，默认为256
# 最好选用821样例
fileList = fileListFunc("D:\\Kinpeng_Zhang\\EHG_Signal\\tpehgdb")
file_write = open("E:\\Fre_domian\\Third_Channel_Fre_assortativity" + ".txt", "a")
for filename in fileList:
    file_write.write(filename.split("\\")[-1].split(".")[0] + "    ")
    single = load_origin_data(filename, 10)
    f, t, Zxx = FSY(single)
    for i in range(len(f)):
        li = np.zeros((len(Zxx[i]), len(Zxx[i])))
        Node_pair = HVG(np.abs(Zxx[i]))
        for i in Node_pair:
            li[i[0]][i[1]] = 1
            li[i[1]][i[0]] = 1
        va = assortativity_bin(li)
        file_write.write(str('%.9f' % va) + "    ")
    file_write.write("\n")
