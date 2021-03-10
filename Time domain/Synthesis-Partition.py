import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


warnings.filterwarnings("ignore")


# smote算法list表示的是特征值的集合，最后一个为分类标记
def Smote_Algorithm(lists):
    Feature_list = np.array(lists)[:, :-1]
    Label = np.array(lists)[:, -1]
    smo = SMOTE(sampling_strategy={1.0: 262},random_state=1337,k_neighbors=3)
    X_smo, y_smo = smo.fit_sample(Feature_list, Label)
    list_x_smo = X_smo.tolist()
    for i in range(len(list_x_smo)):
        list_x_smo[i].append(y_smo[i])
    return list_x_smo


# TODO svc算法
def pred_svc(X_train, X_test, y_train, y_test):
    # training
    svr_rbf10 = SVC(C=1, kernel='rbf', gamma=4.1)
    svr_rbf10.fit(X_train, y_train)
    # TODO 计算AOC面积
    y_pred = svr_rbf10.decision_function(X_test)
    y_p = svr_rbf10.predict(X_test)
    fpr_kb, tpr_kb, threshold = roc_curve(y_test, y_pred)  # 计算真正率和假正率
    Area = auc(fpr_kb, tpr_kb)

    # check performance
    pred_preterm = y_p
    real_preterm = y_test
    Accuracy = np.sum(pred_preterm == real_preterm) / len(pred_preterm)

    """
                               TN          FP             TP         FN
     pred_preterm = np.array([False,      True,          True,      False])
     real_preterm = np.array([False,      False,         True,      True])
    """
    TP = (pred_preterm == real_preterm) & (real_preterm == 1)  # True positive
    FP = (pred_preterm != real_preterm) & (real_preterm == 0)  # prediction is positive, while real is negative
    TN = (pred_preterm == real_preterm) & (real_preterm == 0)  # True negative
    FN = (pred_preterm != real_preterm) & (real_preterm == 1)
    # TPR即为敏感度（sensitivity）TNR即为特异度（specificity）
    sensitivity = np.sum(TP) / (np.sum(TP) + np.sum(FN))  # True Positive Rate
    specificity = np.sum(TN) / (np.sum(TN) + np.sum(FP))  # False Positive Rate
    F_measure = (2 * specificity * sensitivity) / (specificity + sensitivity)
    Gmean = np.sqrt(sensitivity * specificity)
    return sensitivity, specificity, Gmean, F_measure, Area, Accuracy


# 将数据分成5份
def list_split(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def random_list(feature_to_float, m):
    preterm_list = []
    term_list = []
    for i in feature_to_float:
        if i[-1] == 0:
            term_list.append(i)
        elif i[-1] == 1:
            preterm_list.append(i)
    term_list_five = list_split(term_list, int(len(term_list) / m + 1))
    preterm_list_five = list_split(preterm_list, int(len(preterm_list) / m + 1))
    preterm_term_list = []
    for i in range(m):
        a = np.vstack((term_list_five[i], preterm_list_five[i]))
        preterm_term_list.extend(a)
    return preterm_term_list

# 读取文件数据
f = open("D:\\New_Feature\\Thrid_Channel_Graph_Features.txt", 'r')
lines = f.readlines()
L = np.array([i.strip().split() for i in lines])[:, :]
features = L[:, [1,2,3,8]]
feature_to_float = []
for each in features:
    feature_all = list(map(float, each))
    feature_to_float.append(list(feature_all))
list_sensitivity = []
list_specificity = []
list_Gmean = []
list_F_measure = []
list_Area = []
list_Accuracy = []
for i in range(200):
    sensitivity_list = []
    specificity_list = []
    Gmean_list = []
    F_measure_list = []
    Area_list = []
    Accuracy_list = []
    kf = KFold(n_splits=5)

    # 使用smote算法生成1:1的数据
    smote_list = np.array(Smote_Algorithm(feature_to_float))
    # 将生成的数据按照行进行打乱
    np.random.shuffle(smote_list)
    # 训练集与测试集中的早产与非早产样本进行5:1的比例进行分配
    rangdom_preterm_term = random_list(smote_list, 5)
    for train_index, test_index in kf.split(rangdom_preterm_term):
        train_list = np.array(rangdom_preterm_term)[train_index]
        test_list = np.array(rangdom_preterm_term)[test_index]
        train_X = train_list[:, :-1]
        train_Y = train_list[:, -1]
        test_X = test_list[:, :-1]
        test_Y = test_list[:, -1]
        # 将特征进行标准化
        sc = StandardScaler()
        train_X_1 = sc.fit_transform(train_X)
        test_X_1 = sc.fit_transform(test_X)
        # 使用SVM进行早产预测
        sensitivity, specificity, Gmean, F_measure, Area, Accuracy = pred_svc(train_X_1, test_X_1, train_Y, test_Y)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        Gmean_list.append(Gmean)
        F_measure_list.append(F_measure)
        Area_list.append(Area)
        Accuracy_list.append(Accuracy)
    list_sensitivity.append(np.average(sensitivity_list))
    list_specificity.append(np.average(specificity_list))
    list_Gmean.append(np.average(Gmean_list))
    list_F_measure.append(np.average(F_measure_list))
    list_Area.append(np.average(Area_list))
    list_Accuracy.append(np.average(Accuracy_list))

data = {
'sensitivity': list_sensitivity,
'specificity': list_specificity,
'gmean': list_Gmean,
"auc": list_Area,
  "accuracy":list_Accuracy
}

df = pd.DataFrame(data)
df.plot.box(title="Consumer spending in each country")
plt.grid(linestyle="--", alpha=0.3)
plt.show()
print("Graph_sensitivity=", list_sensitivity)
print("Graph_specificity=", list_specificity)
print("Graph_Area=", list_Area)
print("Graph_Accuracy=", list_Accuracy)
