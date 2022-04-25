import os
import time
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Importance_Sampling.models import ThreeDenseNet
from ImportanceSampling import train


class MyDataset(Dataset):  # needed for pytorch
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def training_validation_graphs(metric_train, metric_val, color, model_name='', skip=1):
    fontsize = 10
    plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)  # fontsize of the axes title
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)  # legend fontsize
    loss = metric_train[skip:]
    val_loss = metric_val[skip:]
    epochs = range(skip + 1, len(loss) + skip + 1)

    plt.plot(epochs, loss, color=color, linestyle='dotted', label=model_name + ' In-Sample MSE')
    plt.plot(epochs, val_loss, color=color, label=model_name + ' Out-of-Sample MSE')
    plt.title('Learning curves')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--nodp', action='store_true', default=False)
    parser.add_argument('--check_grad', action='store_true', default=False)
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--noise_multiplier', type=float, default=1.1)
    parser.add_argument('--l2_norm_clip', type=float, default=1.0)

    parser.add_argument('--save', type=str, default='./save/', help='location of results logs')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')

    # Setup
    params = parser.parse_args()
    save = params.save
    n_epochs = params.n_epochs
    batch_size = params.batch_size  # batchsize for IS gradient calculations
    lr = params.lr

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(42)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Dataset and data transformations
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # Scale data to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # retarded pytorch data transforms
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()
    train_set = MyDataset(X_train, y_train)
    test_set = MyDataset(X_test, y_test)

    # Model
    model = ThreeDenseNet(X_train.shape[1]).to(device)
    model_is = ThreeDenseNet(X_train.shape[1]).to(device)
    print(model)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # timer
    if use_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Train the model
    hist = train(model, train_set, test_set, lr, batch_size, save, n_epochs=n_epochs, useIS=False)
    hist_is = train(model_is, train_set, test_set, lr, batch_size, save, n_epochs=n_epochs, useIS=True)

    # compute overall training time
    if use_cuda:
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    print(('\tELAPSED: {:.3f}'.format(elapsed_time)))

    # Plot
    training_validation_graphs(hist[0], hist[2], color='b', model_name=str('uniform'))
    training_validation_graphs(hist_is[0], hist_is[2], color='r', model_name=str('IS'))
    plt.legend()
    plt.show()
    plt.savefig('learning_curves.png')
