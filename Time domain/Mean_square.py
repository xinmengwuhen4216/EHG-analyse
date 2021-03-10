import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from Time_feature.basic_units import cm, inch
import matplotlib.pyplot as plt

f_slop = open("D:\\New_Feature\\Thrid_Channel_Graph_Features.txt", 'r')
plt.rcParams.update({'font.sans-serif': 'Arial', 'xtick.labelsize': 13, 'ytick.labelsize': 16})


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
# (x-min(x))/(max(x)-min(x))
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(np.array(smote_list)[:, :-1])
# X_minMax=preprocessing.scale(np.array(smote_list)[:, :-1])
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
mean_Term_list = []
mean_Preterm_list = []
std_Term_list = []
std_Preterm_list = []
for i in range(np.array(features_to_float).shape[1] - 1):
    # 求均值
    mean_Term_list.append(np.mean(np.array(Term_List)[:, i]))
    mean_Preterm_list.append(np.mean(np.array(Preterm_List)[:, i]))
    # 求方差
    std_Term_list.append(np.std(np.array(Term_List)[:, i]))
    std_Preterm_list.append(np.std(np.array(Preterm_List)[:, i]))

N = 8

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars
print("Preterm",std_Preterm_list)
print("Term",std_Term_list)
ax.bar(ind, np.array(mean_Term_list)+0.1, width, bottom=0*cm, yerr=std_Term_list, label='Term')
ax.bar(ind + width, np.array(mean_Preterm_list)+0.1, width, bottom=0*cm, yerr=std_Preterm_list,
       label='Preterm')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Density', 'Clu_coef', 'Ass_coef', 'Slop',"RMS","M_Fre","P_Fre","SamEn"))

ax.legend(loc=2)
ax.set_ylim(0,1)
ax.yaxis.set_units("")
ax.autoscale_view()
plt.xticks(rotation=-20)

plt.savefig('Fig05.eps', dpi=400, format='eps')

plt.show()