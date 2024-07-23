from collections import Counter
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score



# Data load
data = pd.read_csv(r'file path', index_col=0)

data = data.dropna(subset=['HBA1C_x'], axis=0)
data = data.dropna(axis=0)
data.reset_index(drop=True, inplace=True)
print(data)

# X is manual features
X = data.loc[:, ['MBG', 'SDBG', 'BGmax', 'TIR13.9', 'TIR10',
 'TIR3.9-10', 'LAGE', 'JINDEX', 'CONGA4',
 'HBGI', 'GRADE', 'GRADEEu', 'GRADEHyper', 'MValue', 'AUC', 'AUCHyper',
 'AUC_hypo', 'ratio', 'TRAS', 'Cid']]
print(X)

# Y is HbA1c values
Y = data.loc[:, 'HBA1C']


# HbA1c levels
def classify(x):
    if x > 7.0:
        return 1
    else:
        return 0
Y = Y.apply(classify)


#  manual feature Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #实例化
X = scaler.fit_transform(X)
Y = Y.ravel()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=420)
print('The sample labels are distributed as %s' % Counter(Ytest))


LR = LogisticRegression()
# Parameters range
parameters = {
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']
}

# GridSearchCV
clf = GridSearchCV(LR, parameters, scoring='accuracy', cv=5, n_jobs=-1)
clf.fit(Xtrain, Ytrain)

# Best parameters
print("Best parameters found: ", clf.best_params_)

# 创建逻辑回归模型
logistic_regression_model = LogisticRegression(penalty=clf.best_params_['penalty'],
                                               solver=clf.best_params_['solver'])

# train
logistic_regression_model.fit(Xtrain, Ytrain)

# predict
y_pred = logistic_regression_model.predict(Xtest)
y_prob = logistic_regression_model.predict_proba(Xtest)[:, 1]



# ROC
fpr, tpr, _ = roc_curve(Ytest, y_prob)
roc_auc = auc(fpr, tpr)
df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
# print(f'fpr:{fpr}')
# print(f'tpr:{tpr}')
print(f'AUC:{roc_auc}')
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




accuracy = accuracy_score(Ytest, y_pred)
F1score = f1_score(Ytest, y_pred)
print("Model Accuracy: %.2f%%" % (accuracy * 100.0))
print('LR' + f'的F1分数为{F1score}')



matrix = confusion_matrix(Ytest, y_pred)
dataframe = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
import seaborn as sns
sns.heatmap(dataframe, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion_Matrix'), plt.tight_layout
plt.show()



# # 输出分类报告，包括精确度、召回率和F1分数等
# report = classification_report(y_test, y_pred)
# print("Classification Report:\n", report)
