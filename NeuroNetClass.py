import torch
import torch.nn as nn



class NeuroNet(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(NeuroNet, self).__init__()

        # 全连接层
##        self.fc1 = nn.Sequential(
####            nn.Dropout(p=0.2),
##            nn.Linear(in_features=5, out_features=16),
##            nn.RReLU(inplace=True),
####            nn.Dropout(p=0.5),
##            nn.Linear(in_features=16, out_features=64),
##            nn.RReLU(inplace=True),
##        )
##
##        self.fc2 = nn.Sequential(
####            nn.Dropout(p=0.2),
##            nn.Linear(in_features=2, out_features=16),
##            nn.RReLU(inplace=True),
####            nn.Dropout(p=0.5),
##            nn.Linear(in_features=16, out_features=64),
##            nn.RReLU(inplace=True),
##        )
##
##        self.fc3 = nn.Sequential(
##            nn.Linear(in_features=128, out_features=32),
##            nn.RReLU(inplace=True),
##            nn.Linear(in_features=32, out_features=8),
##            nn.RReLU(inplace=True),
##            nn.Linear(in_features=8, out_features=3),
##            nn.Sigmoid(),
##        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=7, out_features=16),
            nn.RReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.RReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.RReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.RReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.RReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.RReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.RReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.RReLU(),
            nn.Linear(in_features=8, out_features=3),
            nn.Sigmoid(),
        )


    # 前向传播

    def forward(self, x):

##        x = self.fc1(x)
##        y = self.fc2(y)
##
##        # 合并子网络输出
##
##        z = torch.cat([x,y], dim=1)
##        z = self.fc3(z)

        z = self.fc(x)

        return z























##class Neural_Network(nn.Module):
##
##    def __init__(self, ):
##
##        super(Neural_Network, self).__init__()
##        # parameters
##        # TODO: parameters can be parameterized instead of declaring them here
##        self.inputSize = 2
##        self.outputSize = 1
##        self.hiddenSize = 3
##
##        # weights
##        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 2 X 3 tensor
##        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
##
##    def forward(self, X):
##
##        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
##        self.z2 = self.sigmoid(self.z) # activation function
##        self.z3 = torch.matmul(self.z2, self.W2)
##        o = self.sigmoid(self.z3) # final activation function
##        return o
##
##    def sigmoid(self, s):
##
##        return 1 / (1 + torch.exp(-s))
##
##    def sigmoidPrime(self, s):
##
##        # derivative of sigmoid
##        return s * (1 - s)
##
##    def backward(self, X, y, o):
##
##        self.o_error = y - o # error in output
##        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
##        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
##        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
##        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
##        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
##
##    def train(self, X, y):
##
##        # forward + backward pass for training
##        o = self.forward(X)
##        self.backward(X, y, o)
##
##    def saveWeights(self, model):
##
##        # we will use the PyTorch internal storage functions
##        torch.save(model, "NN")
##        # you can reload model with all the weights and so forth with:
##        # torch.load("NN")
##
##    def predict(self):
##
##        print ("Predicted data based on trained weights: ")
##        print ("Input (scaled): \n" + str(xPredicted))
##        print ("Output: \n" + str(self.forward(xPredicted)))