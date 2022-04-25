import os
import glob
import copy
import time
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

from models import ThreeDenseNet
from ImportanceSampling import train

plt.style.use('ggplot')


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


def training_validation_graphs(metric_train, metric_val, color='', model_name='', skip=1):
    fontsize = 10
    plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)  # fontsize of the axes title
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)  # legend fontsize
    loss = metric_train[skip:]
    val_loss = metric_val[skip:]
    epochs = range(skip + 1, len(loss) + skip + 1)

    if color == '':
        # plt.plot(epochs, loss, linestyle='dotted', label=model_name+' In-Sample MSE')
        plt.plot(epochs, val_loss, label=model_name + ' Out-of-Sample MSE')
    else:
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
    parser.add_argument('--noise_multiplier', type=float, default=1.1)
    parser.add_argument('--l2_norm_clip', type=float, default=1.0)

    parser.add_argument('--save', type=str, default='./save/', help='location of results logs')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')

    # Setup
    params = parser.parse_args()
    save = params.save
    lr = params.lr

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device('cuda' if use_cuda else 'cpu')
    model_bkp = -1
    PATH = os.path.dirname(__file__)
    SORT_MEASURES = ["L2", "C1", "F2", "N1", "uniform", "IS"]

    for batch_size in [100, 200, 400]:  # 5%, 10% and 20% of total dataset
        n_epochs = int(batch_size / 100 * params.n_epochs)  # equalize for same amount of iterations
        for gen_measure in ["L2", "C1", "F2", "N1"]:  # Every dataset (4 in total)

            # performance histories
            histories = []

            for sort_measure in SORT_MEASURES:  # for every measure
                if sort_measure != "uniform" and sort_measure != "IS":
                    # load individual batches and stack them
                    target_folder = os.path.join(PATH, "data", gen_measure, "batches_" + sort_measure + "_sort\\")
                    os.chdir(target_folder)
                    train_data = []
                    for file in glob.glob("*.csv"):
                        print(file)
                        if len(train_data) == 0:
                            train_data = np.loadtxt(file, delimiter=',')
                        else:
                            train_data = np.concatenate((train_data, np.loadtxt(file, delimiter=',')), axis=0)

                    X_train = train_data[:, :2]
                    y_train = np.array(train_data[:, -1], dtype='int8')

                    # load test data
                    target_folder = os.path.join(PATH, "data", gen_measure, "test_set")
                    os.chdir(target_folder)
                    for file in glob.glob("*.csv"):
                        test_data = np.loadtxt(file, delimiter=',')
                    X_test = test_data[:, :2]
                    y_test = np.array(test_data[:, -1], dtype='int8')

                    # Scale data to have mean 0 and variance 1 
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # retarded pytorch data transforms
                    X_train = Variable(torch.from_numpy(X_train)).float()
                    y_train = Variable(torch.from_numpy(y_train)).long()
                    X_test = Variable(torch.from_numpy(X_test)).float()
                    y_test = Variable(torch.from_numpy(y_test)).long()
                    train_set = MyDataset(X_train, y_train)
                    test_set = MyDataset(X_test, y_test)

                # Fresh Model
                if model_bkp == -1:
                    model = ThreeDenseNet(X_train.shape[1]).to(device)
                    model_bkp = copy.deepcopy(model)
                else:
                    model = copy.deepcopy(model_bkp)

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
                if sort_measure == "IS":
                    hist = train(model, train_set, test_set, lr, batch_size, save, n_epochs=n_epochs, useIS=True)
                elif sort_measure == "uniform":
                    hist = train(model, train_set, test_set, lr, batch_size, save, n_epochs=n_epochs, useIS=False,
                                 presorted=False)
                else:  # CL
                    hist = train(model, train_set, test_set, lr, batch_size, save, n_epochs=n_epochs, useIS=False,
                                 presorted=True)

                histories.append(hist)
                training_validation_graphs(histories[-1][0], histories[-1][2], model_name=str(sort_measure))

                # compute overall training time
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_time = time.perf_counter() - start_time
                print(('\tELAPSED: {:.3f}'.format(elapsed_time)))

            # Plot
            # for i, hist in enumerate(histories):
            #    training_validation_graphs(hist[0], hist[2], model_name=str(SORT_MEASURES[i]))
            plt.yscale('log')
            plt.legend()
            # plt.show()
            plt.savefig("Gen=" + gen_measure + " batchsize=" + str(batch_size) + " learning_curves.svg")
            plt.clf()
