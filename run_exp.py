import os
import argparse

import numpy as np
import torch
from mpi4py import MPI

from client import Client
from trainer import Trainer
from privacy_checker import check_privacy
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Name of the dataset: [mnist, fashion-mnist, cifar10].",
                    type=str, choices=['mnist', 'fashion-mnist', 'cifar10'], default="mnist")
parser.add_argument("--algorithm",
                    help="Algorithm: ['Regular', 'Joint', 'FedAvg', 'AvgPush', 'FML', 'ProxyFL'].",
                    type=str, choices=['Regular', 'Joint', 'FedAvg', 'AvgPush', 'FML', 'ProxyFL'],
                    default="ProxyFL")
parser.add_argument("--partition_type", help="Name of the dataset: [iid, class, class-rep].",
                    type=str, choices=['iid', 'class', 'class-rep'], default="class")
parser.add_argument("--private_model_type", help="Private model architecture.",
                    type=str, choices=['LeNet5', 'MLP', 'CNN1', 'CNN2', 'Mixed'], default="MLP")
parser.add_argument("--proxy_model_type", help="Proxy model architecture.",
                    type=str, choices=['LeNet5', 'MLP', 'CNN1', 'CNN2'], default="MLP")
parser.add_argument("--result_path", help="Where to save results.",
                    type=str, default="./results")
parser.add_argument("--data_path", help="Where to find the data.",
                    type=str, default="./datasets")
parser.add_argument("--n_clients", help="Number of clients.",
                    type=int, default=2)
parser.add_argument("--use_private_SGD", help="[int as bool] Use private SGD or not.",
                    type=int, default=1)
parser.add_argument("--n_client_data", help="Number of data points for each client.",
                    type=int, default=1000)
parser.add_argument("--optimizer", help="Optimizer.",
                    type=str, default='adam')
parser.add_argument("--lr", help="Learning rate.",
                    type=float, default=0.001)
parser.add_argument("--momentum", help="Momentum for SGD.",
                    type=float, default=0.9)
parser.add_argument("--dml_weight", help="DML weight.",
                    type=float, default=0.5)
parser.add_argument("--major_percent", help="Percentage of majority class for client data partition.",
                    type=float, default=0.8)
parser.add_argument("--noise_multiplier", help="Gaussian noise deviation for DP SGD.",
                    type=float, default=1.0)
parser.add_argument("--l2_norm_clip", help="L2 norm maximum for clipping in DP SGD.",
                    type=float, default=1.0)
parser.add_argument("--n_epochs", help="Number of DML epochs.",
                    type=int, default=1)
parser.add_argument("--n_rounds", help="Number of FL rounds.",
                    type=int, default=300)
parser.add_argument("--batch_size", help="Batch size during training.",
                    type=int, default=250)
parser.add_argument("--device", help="Which cuda device to use.",
                    type=int, default=0)
parser.add_argument("--seed", help="Random seed.",
                    type=int, default=0)
parser.add_argument("--verbose", help="Verbose level.",
                    type=int, default=0)
args = parser.parse_args()

assert 0. <= args.dml_weight <= 1.

comm = MPI.COMM_WORLD
if torch.cuda.device_count() >= comm.size:
    args.device = torch.device(f"cuda:{comm.rank}" if torch.cuda.is_available() else "cpu")
else:
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

args.in_channel = 3 if args.dataset.startswith('cifar') else 1

result_path = os.path.join(args.result_path,
                           args.dataset,
                           f'n_clients_{args.n_clients}',
                           f'data_partition_{args.partition_type}',
                           f'n_client_data_{args.n_client_data}',
                           f'batch_size_{args.batch_size}',
                           f'optimizer_{args.optimizer}',
                           f'lr_{args.lr}',
                           f'use_private_SGD_{args.use_private_SGD}',
                           f'noise_multiplier_{args.noise_multiplier}',
                           f'l2_norm_clip_{args.l2_norm_clip}',
                           f'private_model_type_{args.private_model_type}',
                           f'proxy_model_type_{args.proxy_model_type}',
                           f'dml_weight_{args.dml_weight}',
                           f'major_percent_{args.major_percent}',
                           f'n_epochs_{args.n_epochs}',
                           f'n_rounds_{args.n_rounds}',
                           args.algorithm,
                           )

if not os.path.exists(result_path) and comm.rank == 0:
    os.makedirs(result_path)

# Seed

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data preparation

train_X, train_y, test_X, test_y = utils.get_data(args)

test_data = (test_X, test_y)

args.n_class = len(np.unique(train_y))
client_data_list = utils.partition_data(train_X, train_y, args)

# Clients
logger = utils.get_logger(os.path.join(result_path,
                                       f"seed_{args.seed}_client_{comm.rank}.log"))

if args.algorithm == 'Joint':
    combined_data_x = np.concatenate([data[0] for data in client_data_list], axis=0)
    combined_data_y = np.concatenate([data[1] for data in client_data_list], axis=0)
    client_data = (combined_data_x, combined_data_y)
else:
    client_data = client_data_list[comm.rank]

if comm.rank == 0:
    epsilon, alpha = check_privacy(args)
    delta = 1.0 / args.n_client_data
    logger.info(f"Expected privacy use is ε={epsilon:.2f} and δ={delta:.4f} at α={alpha:.2f}")

if args.private_model_type == 'Mixed':
    model_types = {0: 'MLP',
                   1: 'MLP',
                   2: 'LeNet5',
                   3: 'LeNet5',
                   4: 'CNN1',
                   5: 'CNN1',
                   6: 'CNN2',
                   7: 'CNN2'}
    args.private_model_type = model_types[comm.rank]
    if args.algorithm == 'Regular':
        args.proxy_model_type = model_types[comm.rank]

client = Client(client_data, args)

trainer = Trainer(args)

results = trainer.train(client, test_data, comm, logger, args)

if comm.rank == 0:
    np.savez(os.path.join(result_path,
                          f"seed_{args.seed}.npz"),
             **results)
