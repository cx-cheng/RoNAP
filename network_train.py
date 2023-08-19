import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import random
import glob
import re
import time
import csv

# 载入数据和标签

def load(directory, LaserNumber, StateNumber, LaserRange, StateRange):

    DataList = []

    for fname in glob.glob(directory):
        print(fname)

        TheFile = open(fname)

        for dataline in TheFile:

            dataline = dataline.replace('\n','')
            dataTuple = dataline.split(',')

            SingleDataList = []

            if dataTuple[-1] != 'WARNING':

                for i in range(0, LaserNumber):

                    SingleDataList.append(round(float(re.search(r'\d+', dataTuple[i]).group())/LaserRange,3))

                for i in range(LaserNumber+1, LaserNumber+StateNumber):

                    SingleDataList.append(round(float(re.search(r'\d+', dataTuple[i]).group())/StateRange[i-LaserNumber],3))

                if dataTuple[-1] == 'AHEAD':

                    SingleDataList.append(0)

                elif dataTuple[-1] == 'LEFT':

                    SingleDataList.append(1)

                elif dataTuple[-1] == 'RIGHT':

                    SingleDataList.append(2)

                DataList.append(SingleDataList)

        TheFile.close()

##    for i in range(0, len(DataList)):
##
##        print(DataList[i])

    return DataList


class NeuroNet(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(NeuroNet, self).__init__()

        # 全连接层

        self.fc = nn.Sequential(
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=18, out_features=64),
            nn.RReLU(),
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=128),
            nn.RReLU(),
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=64),
            nn.RReLU(),
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=32),
            nn.RReLU(),
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=32, out_features=8),
            nn.RReLU(),
##            nn.Dropout(p=0.1),
            nn.Linear(in_features=8, out_features=3),
            nn.Softmax(),
        )


    # 前向传播

    def forward(self, x):

        output = self.fc(x)

        return output


# 验证模型在验证集上的正确率

def validate(model, BatchSize, DataTensor, LaserNumber, StateNumber):

    result, num = 0.0, 0

    for i in range(0, int(len(DataTensor)/BatchSize)):

        LaserBatch = DataTensor[i*BatchSize:(i+1)*BatchSize, 0:LaserNumber]
        StateBatch = DataTensor[i*BatchSize:(i+1)*BatchSize, LaserNumber:LaserNumber+StateNumber-1]
        LabelBatch = DataTensor[i*BatchSize:(i+1)*BatchSize, -1]

        output = model.forward(torch.cat([LaserBatch.cuda(),StateBatch.cuda()], dim=1))
        output = output.cpu()
        pred = np.argmax(output.data.numpy(), axis=1)
        labels = LabelBatch.data.numpy()
        result += np.sum((pred == labels))
        num += BatchSize

    acc = 100*result / num

    return acc



def train(DataList, BatchSize, TrainProportion, LaserNumber, StateNumber, epochs, learning_rate, wt_decay):

    # 载入数据

    IndexList = list(range(0, len(DataList)))
    TrainIndexList = random.sample(IndexList, int(len(DataList)*TrainProportion))

    # 随机分配训练集和测试集

    TrainData = []
    TestData = []

    for i in range(0, len(DataList)):

        if i in TrainIndexList:

            TrainData.append(DataList[i])

        else:

            TestData.append(DataList[i])

    TrainDataTensor = torch.FloatTensor(TrainData)
    TestDataTensor = torch.FloatTensor(TestData)

    # 构建模型
    model = NeuroNet()

    # 部署GPU
    device = torch.device('cuda:0')
    model = model.to(device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss()
##    loss_function = nn.MSELoss()

    # 优化器
##    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 正确率数据

    SuccessRate = [[],[]]

    # 逐轮训练
    for epoch in range(epochs):

        # 记录损失值
        loss_rate = 0

        model.train()  # 模型训练

        for i in range(0, int(len(TrainDataTensor)/BatchSize)):

            LaserBatch = TrainDataTensor[i*BatchSize:(i+1)*BatchSize, 0:LaserNumber]
            StateBatch = TrainDataTensor[i*BatchSize:(i+1)*BatchSize, LaserNumber:LaserNumber+StateNumber-1]
            LabelBatch = TrainDataTensor[i*BatchSize:(i+1)*BatchSize, -1]

            # 前向传播

##            print(LaserBatch)
##            print(StateBatch)
##            print(torch.cat([LaserBatch.cuda(),StateBatch.cuda()], dim=1))

            output = model.forward(torch.cat([LaserBatch.cuda(),StateBatch.cuda()], dim=1))

            LabelBatch = LabelBatch.to(dtype=torch.long)

##            print(output)
##            print(LabelBatch)

##            LabelMSE = torch.zeros([BatchSize, len(output[0])])
##
##            for j in range(0, BatchSize):
##
##                LabelMSE[j,int(LabelBatch[j].item())] = 1

##            print(LabelMSE)

            # 误差计算
            loss_rate = loss_function(output, LabelBatch.cuda())

            # 梯度清零
            optimizer.zero_grad()

            # 误差的反向传播
            loss_rate.backward()

            # 更新参数
            optimizer.step()

##            for name,parameters in model.named_parameters():
##
##                print(name,':',parameters)

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), round(loss_rate.item(),2))

        if epoch % 1 == 0:

            model.eval()
            acc_train = validate(model, BatchSize,TrainDataTensor, LaserNumber, StateNumber)
            acc_val = validate(model, BatchSize,TestDataTensor, LaserNumber, StateNumber)

            print('After {} epochs , the acc_train is : '.format(epoch + 1), round(acc_train,2))
            print('After {} epochs , the acc_val is : '.format(epoch + 1), round(acc_val,2))

            SuccessRate[0].append(round(acc_train,2))
            SuccessRate[1].append(round(acc_val,2))

    TimeMark = time.strftime("%Y%m%d%H%M%S")
    SaveFile(SuccessRate, 'training_results'+TimeMark+'.csv')

    return model



def SaveFile(DataLog, StoreFile):

    with open(StoreFile, 'w') as f:

        csv_writer = csv.writer(f)

        # write object state

        for x1 in range(0, len(DataLog)):

            csv_writer.writerow(DataLog[x1])



def main():

    LaserNumber = 17
    StateNumber = 2
    BatchSize = 10
    TrainProportion = 0.8
    LaserRange = 500
    StateRange = [500, 360]
    directory = "DataFile/*.csv"

    # 数据集实例化(创建数据集)
    DataList = load(directory, LaserNumber, StateNumber, LaserRange, StateRange)

    # 构建模型
    model = NeuroNet()

    # 部署GPU
    device = torch.device('cuda:0')
    model = model.to(device)

    # 超参数可自行指定
    model = train(DataList, BatchSize, TrainProportion, LaserNumber, StateNumber, epochs=1000, learning_rate=0.0008, wt_decay=0)

    # 保存模型
    torch.save(model, 'model_net.pkl')


if __name__ == '__main__':

    main()
