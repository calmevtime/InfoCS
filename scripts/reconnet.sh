#!/usr/bin/env bash
mkdir -p ../results/cifar10/cr5/reconnet_basic/ && cd .. && python main_reconnet.py --cr 5 | tee results/cifar10/cr5/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr10/reconnet_basic/ && cd .. && python main_reconnet.py --cr 10 | tee results/cifar10/cr10/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr20/reconnet_basic/ && cd .. && python main_reconnet.py --cr 20 | tee results/cifar10/cr20/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr30/reconnet_basic/ && cd .. && python main_reconnet.py --cr 30 | tee results/cifar10/cr30/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr40/reconnet_basic/ && cd .. && python main_reconnet.py --cr 40 | tee results/cifar10/cr40/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr50/reconnet_basic/ && cd .. && python main_reconnet.py --cr 50 | tee results/cifar10/cr50/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr60/reconnet_basic/ && cd .. && python main_reconnet.py --cr 60 | tee results/cifar10/cr60/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr70/reconnet_basic/ && cd .. && python main_reconnet.py --cr 70 | tee results/cifar10/cr70/reconnet_basic/log.log && cd scripts &&
mkdir -p ../results/cifar10/cr80/reconnet_basic/ && cd .. && python main_reconnet.py --cr 80 | tee results/cifar10/cr80/reconnet_basic/log.log
