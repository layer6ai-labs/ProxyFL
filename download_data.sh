#!/usr/bin/env bash

mkdir datasets
cd datasets
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

mkdir cifar10
cd cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
