import os
import math
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset

from gradcnn import make_optimizer


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


def train_epoch_uniform(model, train_set, sgd, lr, batchsize, epoch, n_epochs, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    loss_fn = torch.nn.MSELoss()

    whitelist = [i for i in range(len(train_set))]  # samples not yet used for training
    indices = [[] for _ in range(math.floor(len(train_set) / batchsize))]
    end = time.time()

    for batch_idx in range(math.floor(len(train_set) / batchsize)):

        # sample examples uniformly
        idx = np.random.choice(np.arange(len(whitelist)), size=np.min([batchsize, len(whitelist)]), replace=False)

        # using for training -> remove from whitelist
        if len(whitelist) != 0:
            for i in sorted(idx, reverse=True):
                whitelist.pop(i)

        # mini-batch
        mini_batch = Subset(train_set, idx)
        # log selected datapoints
        indices[batch_idx].append(idx)
        train_loader = torch.utils.data.DataLoader(mini_batch, batch_size=batchsize)

        # train the model with the mini-batch
        model.eval()  # not train, but eval instead, due to custom class...
        for (input, target) in train_loader:

            sgd.param_groups[0]['lr'] = lr  # reset lr according to parameter

            if torch.cuda.is_available():
                input = input.cuda().to(device)
                target = target.cuda().to(device)

            # compute output
            output = model(input)

            target_bkp = target
            if target.shape != output.shape:
                target = one_hot_embedding(target, output.shape[1]).cuda().to(device)

            loss = loss_fn(output, target)

            # measure accuracy and record loss
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target_bkp.cpu()).float().sum() / batchsize, batchsize)
            losses.update(loss.item(), batchsize)

            # compute gradient and do SGD step
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % 50 == 49 or math.floor(len(train_set) / batchsize) < 20:
                res = '\t'.join([
                    'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                    'Iter: [%d/%d]' % (batch_idx + 1, math.floor(len(train_set) / batchsize)),
                    'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
                    'Loss %.4f' % losses.avg,
                    'Error %.4f' % error.avg,
                    'LR %.4f' % sgd.param_groups[0]['lr']
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, indices


def train_epoch_presorted(model, train_set, sgd, lr, batchsize, epoch, n_epochs, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    loss_fn = torch.nn.MSELoss()

    indices = [[] for _ in range(math.floor(len(train_set) / batchsize))]
    end = time.time()

    for batch_idx in range(math.floor(len(train_set) / batchsize)):

        # sample examples linearly
        idx = [_ for _ in range(batch_idx * batchsize, batch_idx * batchsize + batchsize)]

        # mini-batch
        mini_batch = Subset(train_set, idx)
        # log selected datapoints
        indices[batch_idx].append(idx)
        train_loader = torch.utils.data.DataLoader(mini_batch, batch_size=batchsize)

        # train the model with the mini-batch
        model.eval()  # not train, but eval instead, due to custom class...
        for (input, target) in train_loader:

            sgd.param_groups[0]['lr'] = lr  # reset lr according to parameter

            if torch.cuda.is_available():
                input = input.cuda().to(device)
                target = target.cuda().to(device)

            # compute output
            output = model(input)

            target_bkp = target
            if target.shape != output.shape:
                target = one_hot_embedding(target, output.shape[1]).cuda().to(device)

            loss = loss_fn(output, target)

            # measure accuracy and record loss
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target_bkp.cpu()).float().sum() / batchsize, batchsize)
            losses.update(loss.item(), batchsize)

            # compute gradient and do SGD step
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % 50 == 49 or math.floor(len(train_set) / batchsize) < 20:
                res = '\t'.join([
                    'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                    'Iter: [%d/%d]' % (batch_idx + 1, math.floor(len(train_set) / batchsize)),
                    'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
                    'Loss %.4f' % losses.avg,
                    'Error %.4f' % error.avg,
                    'LR %.4f' % sgd.param_groups[0]['lr']
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, indices


def train_epoch_is(model, train_set, sgd, lr, optimizer, is_batchsize, batchsize, epoch, n_epochs, device,
                   allow_duplicates=True, approximate_gnorm=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    ### IS Settings
    large_sample_count = len(train_set)  # //12
    allow_duplicates = True
    loss_fn = torch.nn.MSELoss()

    whitelist = [i for i in range(len(train_set))]  # samples not yet used for training
    indices = [[] for _ in range(math.floor(len(train_set) / batchsize))]
    end = time.time()

    for batch_idx in range(math.floor(len(train_set) / batchsize)):
        # samples not yet used for training
        leftover_samples = Subset(train_set, whitelist[:large_sample_count])
        is_loader = torch.utils.data.DataLoader(leftover_samples, batch_size=is_batchsize, shuffle=False)

        model.train()

        single_sample_gnorms = []
        count = 0

        ### determine single sample gradients
        # iterate all data points
        for (input, target) in is_loader:

            if count >= large_sample_count:
                break

            if torch.cuda.is_available():
                input = input.cuda().to(device)
                target = target.cuda().to(device)

            # reset gradient
            optimizer.zero_grad()

            # compute output
            output = model(input)

            if target.shape != output.shape:
                target = one_hot_embedding(target, output.shape[1]).cuda().to(device)

            loss = loss_fn(output, target)
            loss.backward()

            if approximate_gnorm:
                # approximate with last/loss-layer
                for sample in range(len(output)):  # for every sample
                    total_norm = 0
                    p = 0
                    # skip all layers
                    for p in model.parameters():
                        pass

                    param_norm = p.bgrad[sample].data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5 * is_batchsize  # multiply for correct scaling

                    # try to calc full norm if approximate norm vanished
                    if total_norm == 0.0:
                        for p in model.parameters():
                            param_norm = p.bgrad[sample].data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5 * is_batchsize  # multiply for correct scaling
                    single_sample_gnorms.append(total_norm)

            else:
                # calculates gradient norm across all layers
                for sample in range(len(output)):  # for every sample
                    total_norm = 0
                    for p in model.parameters():
                        param_norm = p.bgrad[sample].data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5 * is_batchsize  # multiply for correct scaling
                    single_sample_gnorms.append(total_norm)

            count += 1

        # print("Gnorms calculated:" + str(len(single_sample_gnorms)) + "\t Avg norm: " + str(np.average(single_sample_gnorms)) + "\t Max norm: " + str(np.max(single_sample_gnorms)) + "\t Std norm: " + str(np.std(single_sample_gnorms)))

        # sort gradient norms and the indices of samples
        single_sample_gnorms = np.array(single_sample_gnorms)
        idx = np.argsort(single_sample_gnorms, axis=0)[::-1]
        single_sample_gnorms = np.sort(single_sample_gnorms, axis=0)[::-1]

        # O-SGD: pick the samples with weighted probabilities proportional to gnorms
        idx = np.random.choice(idx, p=single_sample_gnorms / np.sum(single_sample_gnorms),
                               size=np.min([batchsize, len(whitelist)]), replace=False)

        # alternative: always use the highest gradient norms for unique mini-batches
        # idx = idx[:np.min([batch_size, len(whitelist)])]

        # using for training -> remove from whitelist
        if not allow_duplicates:
            if len(whitelist) != 0:
                for i in sorted(idx, reverse=True):
                    whitelist.pop(i)

        # Importance sampled mini-batch
        mini_batch = Subset(train_set, idx)
        # log selected datapoints
        indices[batch_idx].append(idx)
        train_loader = torch.utils.data.DataLoader(mini_batch, batch_size=batchsize)

        ### actually train the model with the mini-batch
        model.eval()  # not train because of custom class...
        for (input, target) in train_loader:

            target_bkp = target

            if torch.cuda.is_available():
                input = input.cuda().to(device)
                target = target.cuda().to(device)

            # compute output
            output = model(input)
            if target.shape != output.shape:
                target = one_hot_embedding(target, output.shape[1]).cuda().to(device)

            # gain-ratio (for adaptive learning rate)
            u_gnorm = np.average(single_sample_gnorms[np.random.choice(np.arange(len(single_sample_gnorms)),
                                                                       size=np.min(
                                                                           [3 * batchsize, len(single_sample_gnorms)]),
                                                                       replace=False)])
            o_gnorm = np.average(single_sample_gnorms[idx])
            gain_ratio = u_gnorm / o_gnorm
            if math.isinf(gain_ratio) or math.isnan(gain_ratio):
                gain_ratio = 1

            gain_ratio = np.min([gain_ratio, 3])  # limit gain ratio to avoid divergence with huge LR

            if o_gnorm == 0.0:
                print("Gradient vanished.")

            sgd.param_groups[0]['lr'] = lr * gain_ratio  # adapt LR according to O-SGD paper

            # compute loss
            loss = loss_fn(output, target)

            # measure accuracy and record loss
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target_bkp.cpu()).float().sum() / batchsize, batchsize)
            losses.update(loss.item(), batchsize)

            # compute gradient and do SGD step
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % 50 == 49 or math.floor(len(train_set) / batchsize) < 20:
                res = '\t'.join([
                    'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                    'Iter: [%d/%d]' % (batch_idx + 1, math.floor(len(train_set) / batchsize)),
                    'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
                    'Loss %.4f' % losses.avg,
                    'Error %.4f' % error.avg,
                    'LR %.4f' % sgd.param_groups[0]['lr'],
                    'min-grad %.4f' % np.min(single_sample_gnorms),
                    'max-grad %.4f' % np.max(single_sample_gnorms)
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, indices


def test_epoch(model, loader, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            target_bkp = target

            if torch.cuda.is_available():
                input = input.cuda().to(device)
                target = target.cuda().to(device)

            # compute output
            output = model(input)
            if target.shape != output.shape:
                target = one_hot_embedding(target, output.shape[1]).cuda().to(device)

            loss = F.mse_loss(output, target)  # F.cross_entropy(output, target)

            # measure accuracy and record loss
            batchsize = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target_bkp.cpu()).float().sum() / batchsize, batchsize)
            losses.update(loss.item(), batchsize)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            res = '\t'.join([
                'Test',
                'Iter: [%d/%d]' % ((batch_idx + 1), len(loader)),
                'Time %.3f (%.3f)' % (batch_time.sum, batch_time.avg),
                'Loss %.4f' % losses.avg,
                'Error %.4f' % error.avg,
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, test_set, lr, batchsize, save, n_epochs, useIS=False, presorted=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Optimizer
    optimizer_class = optim.SGD
    model.get_detail(True)
    optimizer_class = make_optimizer(
        cls=optimizer_class,
        noise_multiplier=1.1,
        l2_norm_clip=1.0,
    )

    use_lr_sched = False

    if lr == 0:
        optimizer_params = {'lr': 3.4 / math.sqrt(100)}
        sgd = optim.SGD(model.parameters(), lr=3.4 / math.sqrt(100), weight_decay=2.5e-4)  # O-SGD L2-penalty
        use_lr_sched = True
    else:
        optimizer_params = {'lr': lr}
        sgd = optim.SGD(model.parameters(), lr=lr, weight_decay=2.5e-4)

    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    log_path = 'results.csv'
    if useIS:
        log_path = 'results_is.csv'
    if presorted:
        log_path = 'results_cl.csv'

        # Start log
    with open(os.path.join(save, log_path), 'w') as f:
        f.write('epoch,train_loss,train_error,test_loss,test_error, learning_rate\n')

    with open(os.path.join(save, 'indices_' + log_path), 'w') as f:
        f.write('epoch, batchidx, datapoint_index\n')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False)

    train_loss_hist = []
    train_error_hist = []
    test_loss_hist = []
    test_error_hist = []
    train_loss_hist = []

    # Train model
    for epoch in range(n_epochs):

        if use_lr_sched:
            lr = 3.4 / math.sqrt(100 + epoch)

        if useIS:
            _, train_loss, train_error, indices = train_epoch_is(
                model,
                train_set,
                sgd,  # used for training model
                lr,  # LR schedule from O-SGD paper
                optimizer,  # used for IS
                1000,  # set IS batchsize here, as large as possible for speedup
                batchsize,  # set training batchsize here # O-SGD paper used 32 for MNIST
                epoch,
                n_epochs,
                device
            )
        else:
            if presorted:
                _, train_loss, train_error, indices = train_epoch_presorted(
                    model,
                    train_set,
                    sgd,  # used for training model
                    lr,  # LR schedule from O-SGD paper
                    batchsize,  # set training batchsize here # O-SGD paper used 32 for MNIST
                    epoch,
                    n_epochs,
                    device
                )

            else:
                _, train_loss, train_error, indices = train_epoch_uniform(
                    model,
                    train_set,
                    sgd,  # used for training model
                    lr,  # LR schedule from O-SGD paper
                    batchsize,  # set training batchsize here # O-SGD paper used 32 for MNIST
                    epoch,
                    n_epochs,
                    device
                )

        _, test_loss, test_error = test_epoch(
            model,
            test_loader,
            device
        )

        train_loss_hist.append(train_loss)
        train_error_hist.append(train_error)
        test_loss_hist.append(test_loss)
        test_error_hist.append(test_error)

        # Log indices used
        with open(os.path.join(save, 'indices_' + log_path), 'a') as f:
            for i, batch in enumerate(indices):
                for dp in batch[0]:
                    f.write('%03d,%03d,%03d\n' % (
                        (epoch + 1),
                        (i + 1),
                        dp
                    ))

        # Log results
        with open(os.path.join(save, log_path), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                test_loss,
                test_error
            ))

    return (train_loss_hist, train_error_hist, test_loss_hist, test_error_hist)
