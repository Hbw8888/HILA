import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Dropout, Sigmoid, Conv1d, Conv2d


class Multi_Manual_feature_Pre(nn.Module):
    def __init__(self):
        super(Multi_Manual_feature_Pre, self).__init__()

        self.conv1 = Conv1d(in_channels=1, out_channels=1, kernel_size=12, stride=12, padding=0)
        self.conv4 = Conv1d(in_channels=1, out_channels=1, kernel_size=48, stride=48, padding=0)
        self.conv8 = Conv1d(in_channels=1, out_channels=1, kernel_size=96, stride=96, padding=0)
        self.conv12 = Conv1d(in_channels=1, out_channels=1, kernel_size=144, stride=144, padding=0)
        self.conv24 = Conv1d(in_channels=1, out_channels=1, kernel_size=288, stride=288, padding=0)



        self.multi_feature_fusion = Linear(36, 36)
        self.multi_feature_days_fusion = Linear(134, 67)
        self.multi_manual_feature_fusion = Linear(92, 92)

        self.FIL = Sequential(
            Linear(20, 33),
            ReLU(),
            Dropout(p=0.2),
            Linear(33, 33),
            ReLU(),
            Dropout(p=0.2),
            Linear(33, 20),
            nn.Softmax(dim=0)

        )


        self.Predict_module = Sequential(
            Linear(92, 138),
            ReLU(),
            Dropout(p=0.1),
            # BatchNorm1d(self.hidden_size),
            Linear(138, 92),
            ReLU(),
            Dropout(p=0.1),
            # BatchNorm1d(self.hidden_size),
            Linear(92, 2),
        )

    def feature_1h(self, x):
        day1 = x[:, :288]
        day1 = torch.unsqueeze(day1, dim=1)
        day2 = x[:, 288:]
        # print(day1.shape)
        day2 = torch.unsqueeze(day2, dim=1)
        feature1 = self.conv1(day1)
        # print(feature1.shape)
        feature1 = feature1.permute(0, 2, 1)
        feature1 = feature1.reshape(day1.shape[0], -1)
        feature2 = self.conv1(day2)
        feature2 = feature2.permute(0, 2, 1)
        feature2 = feature2.reshape(day2.shape[0], -1)
        # feature_1h = torch.cat([feature1, feature2], dim=1)
        # print(feature_1h.shape)
        return feature1, feature2



    def feature_4h(self, x):
        day1 = x[:, :288]
        day1 = torch.unsqueeze(day1, dim=1)
        day2 = x[:, 288:]
        # print(day1.shape)
        day2 = torch.unsqueeze(day2, dim=1)
        feature1 = self.conv4(day1)
        # print(feature1.shape)
        feature1 = feature1.permute(0, 2, 1)
        feature1 = feature1.reshape(day1.shape[0], -1)
        feature2 = self.conv4(day2)
        feature2 = feature2.permute(0, 2, 1)
        feature2 = feature2.reshape(day2.shape[0], -1)
        # feature_4h = torch.cat([feature1, feature2], dim=1)
        # print(feature_4h.shape)
        return feature1, feature2


    def feature_8h(self, x):
        day1 = x[:, :288]
        day1 = torch.unsqueeze(day1, dim=1)
        day2 = x[:, 288:]
        # print(day1.shape)
        day2 = torch.unsqueeze(day2, dim=1)
        feature1 = self.conv8(day1)
        # print(feature1.shape)
        feature1 = feature1.permute(0, 2, 1)
        feature1 = feature1.reshape(day1.shape[0], -1)
        feature2 = self.conv8(day2)
        feature2 = feature2.permute(0, 2, 1)
        feature2 = feature2.reshape(day2.shape[0], -1)
        # feature_12h = torch.cat([feature1, feature2], dim=1)
        # print(feature_12h.shape)
        return feature1, feature2



    def feature_12h(self, x):
        day1 = x[:, :288]
        day1 = torch.unsqueeze(day1, dim=1)
        day2 = x[:, 288:]
        # print(day1.shape)
        day2 = torch.unsqueeze(day2, dim=1)
        feature1 = self.conv12(day1)
        # print(feature1.shape)
        feature1 = feature1.permute(0, 2, 1)
        feature1 = feature1.reshape(day1.shape[0], -1)
        feature2 = self.conv12(day2)
        feature2 = feature2.permute(0, 2, 1)
        feature2 = feature2.reshape(day2.shape[0], -1)
        # feature_12h = torch.cat([feature1, feature2], dim=1)
        # print(feature_12h.shape)
        return feature1, feature2



    def feature_24h(self, x):
        day1 = x[:, :288]
        day1 = torch.unsqueeze(day1, dim=1)
        day2 = x[:, 288:]
        # print(day1.shape)
        day2 = torch.unsqueeze(day2, dim=1)
        feature1 = self.conv24(day1)
        # print(feature1.shape)
        feature1 = feature1.permute(0, 2, 1)
        feature1 = feature1.reshape(day1.shape[0], -1)
        feature2 = self.conv24(day2)
        feature2 = feature2.permute(0, 2, 1)
        feature2 = feature2.reshape(day2.shape[0], -1)
        # feature_12h = torch.cat([feature1, feature2], dim=1)
        # print(feature_12h.shape)
        return feature1, feature2



    # def _initialize_weights(self):
    #     for m in self.modules(): #self.modules()迭代器，遍历网络每个结构
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01) #正态分布初值
    #             nn.init.constant_(m.bias, 0) #初始化参数使其为常值，即每个参数值都相同。一般是给网络中bias进行初始化



    def forward(self, x, manual_feature):
        feature_1h1, feature_1h2 = self.feature_1h(x)
        feature_4h1, feature_4h2 = self.feature_4h(x)
        feature_8h1, feature_8h2 = self.feature_8h(x)
        feature_12h1, feature_12h2 = self.feature_12h(x)
        feature_24h1, feature_24h2 = self.feature_24h(x)
        feature_1 = torch.cat([feature_1h1, feature_4h1, feature_8h1, feature_12h1, feature_24h1], dim=1)
        feature_1 = self.multi_feature_fusion(feature_1)
        feature_2 = torch.cat([feature_1h2, feature_4h2, feature_8h2, feature_12h2, feature_24h2], dim=1)
        feature_2 = self.multi_feature_fusion(feature_2)

        # Feature importance learning
        weight = self.FIL(manual_feature)
        manual_feature = torch.mul(manual_feature, weight)  # torch.Size([64, 20])


        # feature fusion
        feature_1 = feature_1.unsqueeze(1)
        feature_2 = feature_2.unsqueeze(1)
        feature_1_2 = torch.cat([feature_1, feature_2], dim=1)
        feature_1_2 = feature_1_2.view(feature_1_2.shape[0], -1)
        feature = torch.cat([feature_1_2, manual_feature], dim=1)
        feature_fusion = self.multi_manual_feature_fusion(feature)

        return self.Predict_module(feature_fusion)







# 检验模型是否正确
if __name__=='__main__':
    chuanbo = Multi_Manual_feature_Pre()
    CGM = torch.randn(64, 576)
    manual_feature = torch.randn(64, 20)
    # output = input.permute(0, 2, 1)
    output = chuanbo(CGM, manual_feature)
    print(output.shape)
    # output = conv1(input)

