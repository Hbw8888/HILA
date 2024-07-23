import os
from collections import Counter

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from

from Model.HILA import Multi_Manual_feature_Pre

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
file = r'D:\PycharmProjects\CGM_HbA1c\data\5years_feature_CGM_extract3.csv'
data = pd.read_csv(file, index_col=0)
# print(data.columns[0])
# data = data.drop(data.columns[0], axis=1)
# print(data)

data = data.dropna(subset=['HBA1C_x'], axis=0)
data = data.dropna(axis=0)
data.reset_index(drop=True, inplace=True)

# print(data)


X = data.iloc[:, 4:]
X = X.drop(['index', 'HBA1C_y', 'GA_y', 'GA/HbA1c_y'], axis=1)
X = X.drop(['大于13.9百p', '大于10百p', '3.9-10百p(含3.9和10)', 'A10Sp', 'A39B10Sp'], axis=1)
# X = X.values[:,:]
Y = data.iloc[:, 1]
print(X.shape)
print(Y)




Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=420)
def classify(x):
    if x > 7.0:
        return 1
    else:
        return 0


# def classify(x):
#     if x > 9.0:
#         return 2
#     elif x > 7.0 and x<= 9.0:
#         return 1
#     else:
#         return 0


Ytrain = Ytrain.apply(classify)
Ytest = Ytest.apply(classify)

print('样本平衡前标签分布为 %s' % Counter(Ytest))


# Y = Y.apply(classify)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #实例化
Xtrain = scaler.fit_transform(Xtrain) #使用fit_transform(data)一步达成结果
Xtest = scaler.fit_transform(Xtest)
# X = scaler.fit_transform(X)


# Xtrain = Xtrain.values
# Xtest = Xtest.values
Ytrain = Ytrain.ravel()
Ytest = Ytest.ravel()
# Y = Y.ravel()
# print("--------------------------")
# print(Ytrain)
# print("--------------------------")
# print(X.values())

#转为numpy
Xtrain = torch.from_numpy(Xtrain)
Xtest = torch.from_numpy(Xtest)
Ytrain = torch.from_numpy(Ytrain)
Ytest = torch.from_numpy(Ytest)

print(Xtrain.shape)
# Xtrain = torch.reshape(Xtrain, (len(Xtrain), 2, 288))
# Xtest = torch.reshape(Xtest, (len(Xtest), 2, 288))
# print(Xtrain.shape)0



Xtrain = Xtrain.float()
Xtest = Xtest.float()
# Ytrain = Ytrain.float()
# Ytest  = Ytest.float()
train_dataset = TensorDataset(Xtrain, Ytrain)
test_dataset = TensorDataset(Xtest, Ytest)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                             shuffle=False, num_workers=0)



# 数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(train_data_size)



# 创建网络模
chuanbo = Multi_Manual_feature_Pre()
# chuanbo = Multi_feature_Pre()
#************************添加GPU***********************
chuanbo.to(device)
#****************************************************
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()  # 损失函数


# optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(chuanbo.parameters(), lr=learning_rate)



total_train_step = 0

total_test_step = 0

epoch = 20
best_acc = 0.0
save_path = 'path'



# 添加tensorboard可视化
# writer = SummaryWriter("./logs_train")


for i in range(epoch):
    print('-----第{}轮训练-----'.format(i + 1))
    # 开始训练
    chuanbo.train()
    for data in train_dataloader:
        inputs, targets = data
        # print(inputs.shape)
        manual_feature = inputs[:, 0:20]
        CGM_data = inputs[:, 20:]
        # print(CGM_data.shape)
        # print(manual_feature.shape)
        output = chuanbo(CGM_data.to(device), manual_feature.to(device))
        loss = loss_fn(output, targets.to(device))
        # print(loss)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 进行参数调优
        total_train_step += 1  # 训练次数 加一
        if total_train_step % 100 == 0:  # 每100次输出一个
            print(
                "训练次数：{}，Loss={}".format(total_train_step, loss.item()))  # item()  加不加都行
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # save
    torch.save(chuanbo.state_dict(), save_path)

# writer.close()
print(best_acc)





