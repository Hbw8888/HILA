import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from collections import Counter

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


for kernel in [
               "linear",
               "poly",
               "rbf",
               "sigmoid"
               ]:
      clf = SVC(kernel = kernel
                 ,class_weight='balanced',
                probability = True
               ).fit(Xtrain, Ytrain)
      # result = clf.predict(Xtest)
      score = clf.score(Xtest, Ytest)
      Y_pred = clf.predict(Xtest)
      y_prob = clf.predict_proba(Xtest)[:, 1]
      # print(clf.predict_proba(Xtest))
      # print(y_prob)
      # print(score)
      F1score = f1_score(Ytest, Y_pred)
      print(kernel + f' F1-score is {F1score}')

      # Confusion matrix
      matrix = confusion_matrix(Ytest, Y_pred)
      dataframe = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
      import seaborn as sns
      sns.heatmap(dataframe, annot=True, fmt='d', cmap='YlGnBu')
      plt.title('Confusion_Matrix'), plt.tight_layout
      plt.show()

      # ROC
      fpr, tpr, thresholds = roc_curve(Ytest, y_prob)
      roc_auc = auc(fpr, tpr)
      df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
      # save
      df.to_csv('file path.csv')
      print(f'fpr:{fpr}')
      print(f'tpr:{tpr}')
      print(f'AUC:{roc_auc}')

      plt.figure()
      plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic')
      plt.legend(loc="lower right")
      plt.show()