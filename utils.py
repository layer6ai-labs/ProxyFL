import sys
import logging

import numpy as np
import torch
import torchvision

CIFAR10_TRAIN_MEAN = np.array((0.4914, 0.4822, 0.4465))[None, :, None, None]
CIFAR10_TRAIN_STD = np.array((0.2470, 0.2435, 0.2616))[None, :, None, None]


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler(filename, mode='w')  # default is 'a' to append
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def get_data(args):

    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':

        data_file = f"{args.data_path}/{args.dataset}.npz"
        dataset = np.load(data_file)
        train_X, train_y = dataset['x_train'], dataset['y_train'].astype(np.int64)
        test_X, test_y = dataset['x_test'], dataset['y_test'].astype(np.int64)

        if args.dataset == 'fashion-mnist':
            train_X = np.reshape(train_X, (-1, 1, 28, 28))
            test_X = np.reshape(test_X, (-1, 1, 28, 28))
        else:
            train_X = np.expand_dims(train_X, 1)
            test_X = np.expand_dims(test_X, 1)

    elif args.dataset == 'cifar10':

        # Only load data, transformation done later

        trainset = torchvision.datasets.CIFAR10(root=f"{args.data_path}/{args.dataset}/",
                                                train=True)
                                                # download = True,
        train_X = trainset.data.transpose([0, 3, 1, 2])
        train_y = np.array(trainset.targets)

        testset = torchvision.datasets.CIFAR10(root=f"{args.data_path}/{args.dataset}/",
                                               train=False)
        test_X = testset.data.transpose([0, 3, 1, 2])
        test_y = np.array(testset.targets)

    else:

        raise ValueError("Unknown dataset")

    return train_X, train_y, test_X, test_y


def data_loader(dataset, inputs, targets, batch_size, is_train=True):

    def cifar10_norm(x):
        x -= CIFAR10_TRAIN_MEAN
        x /= CIFAR10_TRAIN_STD
        return x

    def no_norm(x):
        return x

    if dataset == 'cifar10':
        norm_func = cifar10_norm
    else:
        norm_func = no_norm

    assert inputs.shape[0] == targets.shape[0]
    n_examples = inputs.shape[0]

    sample_rate = batch_size / n_examples
    num_blocks = int(n_examples / batch_size)
    if is_train:
        for i in range(num_blocks):
            mask = np.random.rand(n_examples) < sample_rate
            if np.sum(mask) != 0:
                yield (norm_func(inputs[mask].astype(np.float32) / 255.),
                       targets[mask])
    else:
        for i in range(num_blocks):
            yield (norm_func(inputs[i * batch_size: (i+1) * batch_size].astype(np.float32) / 255.),
                   targets[i * batch_size: (i+1) * batch_size])
        if num_blocks * batch_size != n_examples:
            yield (norm_func(inputs[num_blocks * batch_size:].astype(np.float32) / 255.),
                   targets[num_blocks * batch_size:])


def partition_data(train_X, train_y, args):

    idx = np.arange(0, len(train_X))
    np.random.shuffle(idx)

    # Preparation
    if args.partition_type == 'class':
        avail_idx = np.arange(len(train_X))
        train_labels = train_y
    elif args.partition_type == 'iid':
        idx = np.arange(0, len(train_X))
        np.random.shuffle(idx)

    # Get data
    client_data_list = []

    for i in range(args.n_clients):

        if args.partition_type == 'class':

            client_major_class = np.random.randint(args.n_class)

            avail_X = train_X[avail_idx]
            avail_labels = train_labels[avail_idx]

            major_mask = avail_labels == client_major_class
            major_idx = np.where(major_mask)[0]
            np.random.shuffle(major_idx)
            assert int(args.n_client_data * args.major_percent) <= len(major_idx)
            major_idx = major_idx[:int(args.n_client_data * args.major_percent)]

            minor_idx = np.where(~major_mask)[0]
            np.random.shuffle(minor_idx)
            assert args.n_client_data - int(args.n_client_data * args.major_percent) <= len(minor_idx)
            minor_idx = minor_idx[:args.n_client_data - int(args.n_client_data * args.major_percent)]

            client_data_idx = np.concatenate((major_idx, minor_idx))
            np.random.shuffle(client_data_idx)
            client_data = avail_X[client_data_idx], train_y[avail_idx][client_data_idx]
            client_data_list.append(client_data)

            remaining_idx = set(range(len(avail_idx))) - set(client_data_idx)
            avail_idx = avail_idx[list(remaining_idx)]

        elif args.partition_type == 'class-rep':

            client_major_class = np.random.randint(args.n_class)
            major_mask = train_y == client_major_class
            major_idx = np.where(major_mask)[0]
            np.random.shuffle(major_idx)
            major_idx = major_idx[:int(args.n_client_data * args.major_percent)]

            minor_idx = np.where(~major_mask)[0]
            np.random.shuffle(minor_idx)
            minor_idx = minor_idx[:args.n_client_data - int(args.n_client_data * args.major_percent)]

            client_data_idx = np.concatenate((major_idx, minor_idx))
            np.random.shuffle(client_data_idx)
            client_data = train_X[client_data_idx], train_y[client_data_idx]
            client_data_list.append(client_data)

        elif args.partition_type == 'iid':

            client_data_idx = idx[i * args.n_client_data:(i + 1) * args.n_client_data]
            client_data = train_X[client_data_idx], train_y[client_data_idx]
            client_data_list.append(client_data)

    return client_data_list


def evaluate_model(model, data, args):

    model.eval()
    x, y = data

    loader = data_loader(args.dataset, x, y, batch_size=1000, is_train=False)
    acc = 0.
    for xt, yt in loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(args.device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(args.device)
        preds_labels = torch.squeeze(torch.max(model(xt), 1)[1])
        acc += torch.sum(preds_labels == yt).item()

    return acc / x.shape[0]


def extract_numpy_weights(model):

    tensor_weights = model.state_dict()
    numpy_weights = {}

    for k in tensor_weights.keys():
        numpy_weights[k] = tensor_weights[k].detach().cpu().numpy()

    return numpy_weights


def convert_np_weights_to_tensor(weights):

    for k in weights.keys():
        weights[k] = torch.from_numpy(weights[k])

    return weights
