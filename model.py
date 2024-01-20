import torch 
import torch.nn as nn
import torch.optim as optim

class KI_CNN_model(nn.Module):

    def __init__(self):
        super(KI_CNN_model, self).__init__()
        self.conv1=nn.Conv2d(1, 8, 5)
        self.conv2=nn.Conv2d(8, 4, 5)
        self.conv3=nn.Conv2d(4, 2, 5)

        self.fc1=nn.Linear(2* 198* 148, 5)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.sigmoid(self.conv3(x))
        x=x.view(-1, 2* 198* 148)
        x=self.fc1(x)
        return x

class KI_MLP_model_grayscale(nn.Module):
    def __init__(self):
        super(KI_MLP_model_grayscale, self).__init__()
        self.fc1=nn.Linear(210*160, 512)
        self.fc2=nn.Linear(512, 128)
        self.fc3=nn.Linear(128, 5)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=x.view(-1, 210*160)
        x=self.sigmoid(self.fc1(x.float()))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        return x

class KI_MLP_model_ram(nn.Module):
    def __init__(self):
        super(KI_MLP_model_ram, self).__init__()
        self.fc1=nn.Linear(128, 512)
        self.fc2=nn.Linear(512, 5)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=self.sigmoid(self.fc1(x))
        x=self.relu(self.fc2(x))
        return x