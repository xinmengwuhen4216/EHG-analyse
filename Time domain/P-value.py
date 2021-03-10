import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from scipy import stats

f_slop = open("D:\\New_Feature\\Thrid_Channel_Graph_Features.txt", 'r')


def Smote_Algorithm(lists):
    Feature_list = np.array(lists)[:, :-1]
    Label = np.array(lists)[:, -1]
    smo = SMOTE(sampling_strategy={1.0: 262}, random_state=1337, k_neighbors=3)
    X_smo, y_smo = smo.fit_sample(Feature_list, Label)
    list_x_smo = X_smo.tolist()
    for i in range(len(list_x_smo)):
        list_x_smo[i].append(y_smo[i])
    return list_x_smo


# 读取斜率数据
lines_slop = f_slop.readlines()
L_slop = np.array([i.strip().split() for i in lines_slop])
features_to_float = []
for each in L_slop:
    feature_all = list(map(float, each))
    features_to_float.append(feature_all)

# 使用smote算法平衡样本
smote_list = Smote_Algorithm(features_to_float)

# 将所有的数据进行归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(np.array(smote_list)[:, :-1])
# 将归一化的数据添加标签
after_std = np.hstack((X_minMax, [[j] for j in np.array(smote_list)[:, -1]]))
Term_List = []
Preterm_List = []
# 将数据分成早产与非早产样本
for i in after_std:
    if (i[-1] == 1):
        Preterm_List.append(i)
    elif (i[-1] == 0):
        Term_List.append(i)

for i in range(np.array(features_to_float).shape[1]-1):
    # 求出每个特征的P-value值
    t, p1 = stats.ranksums(np.array(Preterm_List)[:, i], np.array(Term_List)[:, i])
    print(p1)
