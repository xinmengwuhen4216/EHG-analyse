import numpy as np


def density_und(CIJ):  # 无向网络的密度
    '''
    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected (weighted/binary) connection matrix

    Returns
    -------
    kden : float
        density
    N : int
        number of vertices
    k : int
        number of edges

    Notes
    -----
    Assumes CIJ is undirected and has no self-connections.
            Weight information is discarded.
    '''
    n = len(CIJ)
    k = np.size(np.where(np.triu(CIJ).flatten()))
    kden = k / ((n * n - n) / 2)
    return kden   # 返回的是无向网络的密度


def clustering_coef_bu(G): # 聚类系数
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
    average=np.average(C)
    return average
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

def assortativity_bin(CIJ, flag=0):   # 同配系数
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
        1 : directed graph; out-degree/in-degree correlation
        2 : directed graph; in-degree/out-degree correlation
        3 : directed graph; out-degree/out-degree correlation
        4 : directed graph; in-degree/in-degreen correlation

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
    else:  # directed version
        id, od, deg = degrees_dir(CIJ)
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            degi = od[i]
            degj = id[j]
        elif flag == 2:
            degi = id[i]
            degj = od[j]
        elif flag == 3:
            degi = od[i]
            degj = od[j]
        elif flag == 4:
            degi = id[i]
            degj = id[j]
        else:
            raise ValueError('Flag must be 0-4')

    # compute assortativity
    term1 = np.sum(degi * degj) / K
    term2 = np.square(np.sum(.5 * (degi + degj)) / K)
    term3 = np.sum(.5 * (degi * degi + degj * degj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r
# G = [[0, 1, 1, 0, 1],
#      [1, 0, 0, 1, 0],
#      [1, 0, 0, 0, 1],
#      [0, 1, 0, 0, 0],
#      [1, 0, 1, 0, 0]]
# print("网络密度",density_und(G))
# print("这是聚类系数",clustering_coef_bu(np.array(G)))
# print("同配系数",assortativity_bin(np.array(G)))
