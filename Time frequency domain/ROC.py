import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from scipy import interp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.sans-serif': 'Times New Roman', 'xtick.labelsize': 16, 'ytick.labelsize': 16})
plt.rcParams.update({'font.size': 13})     #设置图例字体大小
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



# smote算法list表示的是特征值的集合，最后一个为分类标记
def Smote_Algorithm(lists):
    Feature_list = lists[:, :-1]
    Label = lists[:, -1]
    smo = SMOTE(random_state=1337, k_neighbors=3)
    X_smo, y_smo = smo.fit_sample(Feature_list, Label)
    list_x_smo = X_smo.tolist()
    for i in range(len(list_x_smo)):
        list_x_smo[i].append(y_smo[i])
    return list_x_smo


# TODO svc算法
def pred_svc(X_train, X_test, y_train, y_test):
    # training
    svr_rbf10 = SVC(C=1, gamma=0.005, kernel='rbf')
    svr_rbf10.fit(X_train, y_train)
    # TODO 计算AOC面积
    y_pred = svr_rbf10.decision_function(X_test)
    y_p = svr_rbf10.predict(X_test)
    fpr_kb, tpr_kb, threshold = roc_curve(y_test, y_pred)  # 计算真正率和假正率
    area = auc(fpr_kb, tpr_kb)

    return fpr_kb, tpr_kb,area


f = open("E:\\Fre_domian\Third_Channel_Fre_assortativity_label.txt", 'r')
lines = f.readlines()
L = np.array([i.strip().split() for i in lines])[:, 1:]
feature_to_float = []
for each in L:
    feature_all = np.abs(list(map(float, each)))
    feature_to_float.append(feature_all)
X_new = SelectKBest(chi2, k=55).fit_transform(np.array(feature_to_float)[:, :-1], np.array(feature_to_float)[:, -1])
features = np.array(np.hstack((X_new, [[j] for j in np.array(feature_to_float)[:, -1]])))
kf = KFold(n_splits=5)
tprs=[]
aucs=[]
i=1

mean_fpr=np.linspace(0,1,100)
np.random.shuffle(features)
rangdom_preterm_term=random_list(features,5)
for train_index, test_index in kf.split(rangdom_preterm_term):
    train_list = np.array(rangdom_preterm_term)[train_index]
    test_list = np.array(rangdom_preterm_term)[test_index]
    smote_train_list = np.array(Smote_Algorithm(train_list))
    smote_test_list = np.array(Smote_Algorithm(test_list))
    train_X = smote_train_list[:, :-1]
    train_Y = smote_train_list[:, -1]
    test_X = smote_test_list[:, :-1]
    test_Y = smote_test_list[:, -1]
    # 将特征进行标准化
    sc = StandardScaler()
    train_X_1 = sc.fit_transform(train_X)
    test_X_1 = sc.fit_transform(test_X)
    # 使用SVM进行早产预测
    fpr, tpr, area = pred_svc(train_X_1, test_X_1, train_Y, test_Y)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)


    # interp:插值 把结果添加到tprs列表中
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i=i+1



#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='royalblue',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='royalblue',label=r'r ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')





f = open("E:\\Fre_domian\Third_Channel_Fre_slop_label.txt", 'r')
lines = f.readlines()
L = np.array([i.strip().split() for i in lines])[:, 1:]
feature_to_float = []
for each in L:
    feature_all = np.abs(list(map(float, each)))
    feature_to_float.append(feature_all)
X_new = SelectKBest(chi2, k=55).fit_transform(np.array(feature_to_float)[:, :-1], np.array(feature_to_float)[:, -1])
features = np.array(np.hstack((X_new, [[j] for j in np.array(feature_to_float)[:, -1]])))
kf = KFold(n_splits=5)
tprs=[]
aucs=[]
i=1

mean_fpr=np.linspace(0,1,100)
np.random.shuffle(features)
rangdom_preterm_term=random_list(features,5)
for train_index, test_index in kf.split(rangdom_preterm_term):
    train_list = np.array(rangdom_preterm_term)[train_index]
    test_list = np.array(rangdom_preterm_term)[test_index]
    smote_train_list = np.array(Smote_Algorithm(train_list))
    smote_test_list = np.array(Smote_Algorithm(test_list))
    train_X = smote_train_list[:, :-1]
    train_Y = smote_train_list[:, -1]
    test_X = smote_test_list[:, :-1]
    test_Y = smote_test_list[:, -1]
    # 将特征进行标准化
    sc = StandardScaler()
    train_X_1 = sc.fit_transform(train_X)
    test_X_1 = sc.fit_transform(test_X)
    # 使用SVM进行早产预测
    fpr, tpr, area = pred_svc(train_X_1, test_X_1, train_Y, test_Y)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)


    # interp:插值 把结果添加到tprs列表中
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i=i+1



#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='lightblue',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='lightblue',label=r'k ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.legend()




f = open("E:\\Fre_domian\Third_Channel_Fre_clustering_label.txt", 'r')
lines = f.readlines()
L = np.array([i.strip().split() for i in lines])[:, 1:]
feature_to_float = []
for each in L:
    feature_all = np.abs(list(map(float, each)))
    feature_to_float.append(feature_all)
X_new = SelectKBest(chi2, k=55).fit_transform(np.array(feature_to_float)[:, :-1], np.array(feature_to_float)[:, -1])
features = np.array(np.hstack((X_new, [[j] for j in np.array(feature_to_float)[:, -1]])))
kf = KFold(n_splits=5)
tprs=[]
aucs=[]
i=1

mean_fpr=np.linspace(0,1,100)
np.random.shuffle(features)
rangdom_preterm_term=random_list(features,5)
for train_index, test_index in kf.split(rangdom_preterm_term):
    train_list = np.array(rangdom_preterm_term)[train_index]
    test_list = np.array(rangdom_preterm_term)[test_index]
    smote_train_list = np.array(Smote_Algorithm(train_list))
    smote_test_list = np.array(Smote_Algorithm(test_list))
    train_X = smote_train_list[:, :-1]
    train_Y = smote_train_list[:, -1]
    test_X = smote_test_list[:, :-1]
    test_Y = smote_test_list[:, -1]
    # 将特征进行标准化
    sc = StandardScaler()
    train_X_1 = sc.fit_transform(train_X)
    test_X_1 = sc.fit_transform(test_X)
    # 使用SVM进行早产预测
    fpr, tpr, area = pred_svc(train_X_1, test_X_1, train_Y, test_Y)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)


    # interp:插值 把结果添加到tprs列表中
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i=i+1



#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='pink',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='pink',label=r'$C_g$ ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.legend()


f = open("E:\\Fre_domian\Third_Channel_Fre_density_label.txt", 'r')
lines = f.readlines()
L = np.array([i.strip().split() for i in lines])[:, 1:]
feature_to_float = []
for each in L:
    feature_all = np.abs(list(map(float, each)))
    feature_to_float.append(feature_all)
X_new = SelectKBest(chi2, k=55).fit_transform(np.array(feature_to_float)[:, :-1], np.array(feature_to_float)[:, -1])
features = np.array(np.hstack((X_new, [[j] for j in np.array(feature_to_float)[:, -1]])))
kf = KFold(n_splits=5)
tprs=[]
aucs=[]
i=1

mean_fpr=np.linspace(0,1,100)
np.random.shuffle(features)
rangdom_preterm_term=random_list(features,5)
for train_index, test_index in kf.split(rangdom_preterm_term):
    train_list = np.array(rangdom_preterm_term)[train_index]
    test_list = np.array(rangdom_preterm_term)[test_index]
    smote_train_list = np.array(Smote_Algorithm(train_list))
    smote_test_list = np.array(Smote_Algorithm(test_list))
    train_X = smote_train_list[:, :-1]
    train_Y = smote_train_list[:, -1]
    test_X = smote_test_list[:, :-1]
    test_Y = smote_test_list[:, -1]
    # 将特征进行标准化
    sc = StandardScaler()
    train_X_1 = sc.fit_transform(train_X)
    test_X_1 = sc.fit_transform(test_X)
    # 使用SVM进行早产预测
    fpr, tpr, area = pred_svc(train_X_1, test_X_1, train_Y, test_Y)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)


    # interp:插值 把结果添加到tprs列表中
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i=i+1



#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='lawngreen',label=r'D ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.legend()


plt.savefig('ROC.eps', dpi=400, format='eps')

plt.show()