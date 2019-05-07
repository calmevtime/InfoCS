#!/usr/bin/env bash
mkdir -p ../results/cifar10/cr5/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 5 | tee results/cifar10/cr5/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr10/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 10 | tee results/cifar10/cr10/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr20/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 20 | tee results/cifar10/cr20/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr30/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 30 | tee results/cifar10/cr30/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr40/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 40 | tee results/cifar10/cr40/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr50/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 50 | tee results/cifar10/cr50/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr60/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 60 | tee results/cifar10/cr60/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr70/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 70 | tee results/cifar10/cr70/reconnet_sparse/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr80/reconnet_sparse/ && cd .. && python main_reconnet_sparse.py --cr 80 | tee results/cifar10/cr80/reconnet_sparse/log.log
