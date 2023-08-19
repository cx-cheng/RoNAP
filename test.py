import torch
import torch.nn as nn





class NeuroNet(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(NeuroNet, self).__init__()

        # 全连接层

        self.fc = nn.Sequential(
            nn.Linear(in_features=7, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=3),
            nn.Sigmoid(),
        )

    # 前向传播

    def forward(self, x):

        output = self.fc(x)

        return output




def main():

    # 构建模型
    model = NeuroNet()

    # 部署GPU
    device = torch.device('cuda:0')
    model = model.to(device)

    Batch = torch.rand(10,7)

    print(Batch)

    output = model.forward(Batch.cuda())

    print(output)


if __name__ == '__main__':

    main()


