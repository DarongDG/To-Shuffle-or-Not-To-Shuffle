import os
import time
import argparse
import time
import torch
from torchvision import transforms, datasets
from Importance_Sampling.models import LeNet5
from ImportanceSampling import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # gradient calculation parameters
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--nodp', action='store_true', default=False)
    parser.add_argument('--check_grad', action='store_true', default=False)
    parser.add_argument('--noise_multiplier', type=float, default=1.1)
    parser.add_argument('--l2_norm_clip', type=float, default=1.0)

    # general parameters
    parser.add_argument('--data', type=str, default='/tmp/', help='location of MNIST dataset')
    parser.add_argument('--save', type=str, default='./save/', help='location of results logs')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')

    # Setup
    params = parser.parse_args()
    data = params.data
    save = params.save
    n_epochs = params.n_epochs
    use_cuda = torch.cuda.is_available()
    
    torch.manual_seed(42)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Dataset and data transformations
    train_set = datasets.MNIST(data, train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(data, train=False, transform=transforms.ToTensor(), download=True)

    n_classes = 10
    n_channels = 1
    input_size = 28

    # Model
    input_size = (n_channels, input_size, input_size)
    model = LeNet5(input_size=input_size, kernel_size=5).to(device)
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
    train(model, train_set, test_set, 32, save, n_epochs=n_epochs, useIS=False)

    # compute overall training time
    if use_cuda:
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    print(('\tELAPSED: {:.3f}'.format(elapsed_time)))