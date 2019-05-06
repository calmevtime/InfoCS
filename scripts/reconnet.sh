#!/usr/bin/env bash
mkdir -p ../results/cifar10/cr10/reconnet_basic/ && cd .. && python main_reconnet.py --cr 10 | tee results/cifar10/cr10/reconnet_basic/log.log &&
mkdir -p ../results/cifar10/cr20/reconnet_basic/ && cd .. && python main_reconnet.py --cr 20 | tee results/cifar10/cr20/reconnet_basic/log.log &&
mkdir -p ../results/cifar10/cr40/reconnet_basic/ && cd .. && python main_reconnet.py --cr 40 | tee results/cifar10/cr40/reconnet_basic/log.log &&
mkdir -p ../results/cifar10/cr80/reconnet_basic/ && cd .. && python main_reconnet.py --cr 80 | tee results/cifar10/cr80/reconnet_basic/log.log
