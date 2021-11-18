#!/usr/bin/env bash

n_epochs=1
use_private_SGD=1
noise_multiplier=1.0
l2_norm_clip=1.0

lr=0.001
optimizer='adam'

major_percent=0.8
dml_weight=0.5
verbose=1

# CIFAR10
# dataset='cifar10'
# partition_type='class'
# n_rounds=200
# private_model_type='CNN2'
# proxy_model_type='CNN1'
# n_client_data=3000
# n_clients=8
# batch_size=250

# MNIST and Fashion-MNIST
dataset='mnist'  # 'fashion-mnist'
partition_type='class'
n_rounds=300
private_model_type='MLP'  # LeNet5
proxy_model_type='MLP'
n_clients=8
n_client_data=1000
batch_size=250

for major_percent in 0.8  # 0.1 0.3 0.5
# for dml_weight in 0.1 0.3 0.5 0.7 0.9
# for n_clients in 3 4 5 6 7 8
do
    for algorithm in 'Regular' 'Joint' 'AvgPush' 'FedAvg' 'FML' 'ProxyFL'
    # 'Regular' 'Joint' 'AvgPush' 'FedAvg' 'FML' 'ProxyFL'
    do
        for sd in {0..4}
        do
            mpiexec -n ${n_clients} python run_exp.py \
                    --dataset=${dataset} \
                    --algorithm=${algorithm} \
                    --seed=${sd} \
                    --partition_type=${partition_type} \
                    --n_clients=${n_clients} \
                    --n_client_data=${n_client_data} \
                    --private_model_type=${private_model_type} \
                    --proxy_model_type=${proxy_model_type} \
                    --use_private_SGD=${use_private_SGD} \
                    --noise_multiplier=${noise_multiplier} \
                    --l2_norm_clip=${l2_norm_clip} \
                    --optimizer=${optimizer} \
                    --lr=${lr} \
                    --dml_weight=${dml_weight} \
                    --major_percent=${major_percent} \
                    --n_epochs=${n_epochs} \
                    --n_rounds=${n_rounds} \
                    --batch_size=${batch_size} \
                    --verbose=${verbose}
        done
        # wait
    done
    python plot.py --dataset=${dataset} \
                    --n_runs=5 \
                    --partition_type=${partition_type} \
                    --n_clients=${n_clients} \
                    --n_client_data=${n_client_data} \
                    --use_private_SGD=${use_private_SGD} \
                    --optimizer=${optimizer} \
                    --lr=${lr} \
                    --noise_multiplier=${noise_multiplier} \
                    --l2_norm_clip=${l2_norm_clip} \
                    --private_model_type=${private_model_type} \
                    --proxy_model_type=${proxy_model_type} \
                    --dml_weight=${dml_weight} \
                    --major_percent=${major_percent} \
                    --batch_size=${batch_size} \
                    --n_epochs=${n_epochs} \
                    --n_rounds=${n_rounds}
done
