from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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



rfc = RandomForestClassifier(n_jobs=-1)

# Parameters range
parameters = {
              'max_depth': [5, 10, 15, 20, 30, 40, 50, 80, 100],
              'n_estimators': [5, 10, 15, 20, 30, 40, 50, 80, 100]
}

# GridSearchCV
clf = GridSearchCV(rfc, parameters, scoring='accuracy', cv=5, n_jobs=-1)
clf.fit(Xtrain, Ytrain)

# Best parameters
print("Best parameters found: ", clf.best_params_)

# Model
rfc_model = RandomForestClassifier(
                                  max_depth=clf.best_params_['max_depth'],
                                  n_estimators=clf.best_params_['n_estimators'],
                                  n_jobs=-1)

# train
rfc_model.fit(Xtrain, Ytrain)


# predict
y_pred_train = rfc_model.predict(Xtrain)
y_pred = rfc_model.predict(Xtest)
y_prob = rfc_model.predict_proba(Xtest)[:, 1]


# ROC
fpr, tpr, _ = roc_curve(Ytest, y_prob)
roc_auc = auc(fpr, tpr)
print('auc为'+str(roc_auc))
# df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
# save
# df.to_csv('D:\PycharmProjects\CGM_HbA1c\二分类\样本平衡\ROC数据\RFC_ROC.csv')
print(f'fpr:{fpr}')
print(f'tpr:{tpr}')
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




# Confusion matrix
matrix = confusion_matrix(Ytest, y_pred)
dataframe = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
import seaborn as sns
sns.heatmap(dataframe, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion_Matrix'), plt.tight_layout
plt.show()

accuracy = accuracy_score(Ytest, y_pred)
F1score = f1_score(Ytest, y_pred)
print("RFC Accuracy: %.2f%%" % (accuracy * 100.0))
print('RFC' + f'F1-score is{F1score}')


