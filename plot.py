import os
import argparse

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='plot script')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--n_runs', type=int, default=5)  # number of trials
parser.add_argument("--partition_type", help="Type of data partition.",
                    type=str, default="class")
parser.add_argument("--n_clients", help="Number of clients.",
                    type=int, default=8)
parser.add_argument("--n_client_data", help="Number of data points for each client.",
                    type=int, default=1000)
parser.add_argument("--use_private_SGD", help="[int as bool] Use private SGD or not.",
                    type=int, default=1)
parser.add_argument("--optimizer", help="Optimizer.",
                    type=str, default='adam')
parser.add_argument("--lr", help="Learning rate.",
                    type=float, default=0.001)
parser.add_argument("--noise_multiplier", help="Gaussian noise deviation for DP SGD.",
                    type=float, default=1.0)
parser.add_argument("--l2_norm_clip", help="L2 norm maximum for clipping in DP SGD.",
                    type=float, default=1.0)
parser.add_argument("--private_model_type", help="Private model architecture.",
                    type=str, default="MLP")
parser.add_argument("--proxy_model_type", help="Proxy model architecture.",
                    type=str, default="MLP")
parser.add_argument("--dml_weight", help="DML strength.",
                    type=float, default=0.5)
parser.add_argument("--major_percent", help="Percentage of majority class for client data partition.",
                    type=float, default=0.8)
parser.add_argument("--n_epochs", help="Number of DML epochs.",
                    type=int, default=1)
parser.add_argument("--n_rounds", help="Number of FL rounds.",
                    type=int, default=300)
parser.add_argument("--batch_size", help="Batch size during training.",
                    type=int, default=250)
parser.add_argument('--result_path', type=str, default='./results')
args = parser.parse_args()

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
                           )

algorithms = ['Regular', 'Joint', 'FedAvg', 'AvgPush', 'FML', 'ProxyFL']
assert algorithms[0] == 'Regular' and algorithms[1] == 'Joint'
labels = []

all_accuracies = []

# load the results
for algorithm in algorithms:
    algo_acc = []
    private_acc = []
    for seed in range(args.n_runs):

        load_file = os.path.join(result_path,
                                 algorithm,
                                 f'seed_{seed}.npz',
                                 )

        load_results = np.load(load_file)

        algo_acc.append(load_results['proxy_accuracies'])
        if algorithm == 'FML' or algorithm == 'ProxyFL':
            private_acc.append(load_results['private_accuracies'])

    algo_acc = np.stack(algo_acc)

    all_accuracies.append(algo_acc)

    if algorithm == 'FML' or algorithm == 'ProxyFL':
        private_acc = np.stack(private_acc)
        all_accuracies.append(private_acc)
        labels.append(algorithm + '-proxy')
        labels.append(algorithm + '-private')
    else:
        labels.append(algorithm)

all_accuracies = np.stack(all_accuracies)  # methods x runs x clients x rounds
mean_accuracies = np.mean(all_accuracies, axis=(1, 2))
std_accuracies = np.std(all_accuracies, axis=(1, 2))

mean_accuracies[0] = np.amax(mean_accuracies[0])
std_accuracies[0] = 0
mean_accuracies[1] = np.max(mean_accuracies[1])
std_accuracies[1] = 0

n_algo, n_runs, n_clients, n_rounds = all_accuracies.shape

x = np.arange(n_rounds)
cmap = plt.get_cmap('jet')  # hsv, jet
colors = cmap(np.linspace(0, 1.0, n_algo))
markers = [None, None,
           'o', '^', 's', 'd', 'P', 'v', '*', 'X', 'h', 'D']  # may be not enough
font = {'family': 'monospace',
        'size': 12}
rc('font', **font)
markersize = 9

# Plot accuracy
markevery = args.n_rounds // 10
plt.figure(figsize=(6, 4))  # ratio
for i in range(len(labels)):

    matched_idx = (i - 1) if labels[i].endswith('-private') else i

    linestyle = ':' if labels[i].endswith('-private') else '-'
    linestyle = '--' if i <= 1 else linestyle  # for regular and joint

    plt.plot(x, mean_accuracies[i], label=labels[i], color=colors[matched_idx],
             linestyle=linestyle,
             marker=markers[matched_idx], markersize=markersize, markevery=markevery)
    plt.fill_between(x,
                     mean_accuracies[i] - std_accuracies[i],
                     mean_accuracies[i] + std_accuracies[i],
                     facecolor=colors[matched_idx], alpha=0.2)

plt.grid(True)
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
# plt.xticks(x, np.arange(n_rounds))
plt.legend()
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")  # default legend
fig_file = os.path.join(result_path,
                        'acc.pdf')
plt.savefig(fig_file, bbox_inches='tight')
plt.close()
