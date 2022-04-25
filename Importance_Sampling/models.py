import torch
import torch.nn.functional as F
from gradcnn import crb as nn


class LeNet5(nn.Module):
    '''Create CNN of arbitrary size'''

    def __init__(self, input_size=(1, 28, 28), kernel_size=5):
        super().__init__()

        # building LeNet5 arch
        inputsize = 28

        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kernel_size, stride=1,
                               padding=int((32 - inputsize) / 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=kernel_size, stride=1, padding=0)
        # torch.nn.init.kaiming_uniform_(conv_layers[0].weight)
        # torch.nn.init.kaiming_uniform_(conv_layers[1].weight)

        # no idea why they do this, probably just to define the attribute...
        # self.conv_output = self.conv_block(torch.randn(1,*input_size)).size(1)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5,
                             120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        # torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        # torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(84, 10)  # convert matrix with 84 features to a matrix of 10 features (columns)
        # torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def conv_block(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        # x = F.tanh(self.conv3(x))
        return x.view(x.shape[0], -1)  # view = flatten

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


class ThreeDenseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        n = 200
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, n)
        self.fc2 = nn.Linear(n, int(n / 2))
        self.fc3 = nn.Linear(int(n / 2), int(n / 4))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # x = F.softmax(self.fc3(x), dim=1)
        return x
