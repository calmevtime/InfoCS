#!/usr/bin/env bash
mkdir -p ../results/cifar10/cr10/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 10 | tee results/cifar10/cr10/reconnet_sparse/log.log &&
mkdir -p ../results/cifar10/cr20/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 20 | tee results/cifar10/cr20/reconnet_sparse/log.log &&
mkdir -p ../results/cifar10/cr40/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 40 | tee results/cifar10/cr40/reconnet_sparse/log.log &&
mkdir -p ../results/cifar10/cr80/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 80 | tee results/cifar10/cr80/reconnet_sparse/log.log
