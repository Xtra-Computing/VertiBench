#!/usr/bin/env bash


# mnist
# mkdir -p out/mnist
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 0.1 -s 0 -g 0 > out/mnist/mnist_imp_w0.1_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 1.0 -s 0 -g 0 > out/mnist/mnist_imp_w1.0_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 10 -s 0 -g 1 > out/mnist/mnist_imp_w10.0_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 100 -s 0 -g 1 > out/mnist/mnist_imp_w100.0_seed0.txt &
# wait
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 0 > out/mnist/mnist_corr_w0.0_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.3 -s 0 -g 0 > out/mnist/mnist_corr_w0.3_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.6 -s 0 -g 1 > out/mnist/mnist_corr_w0.6_seed0.txt &
# python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 1.0 -s 0 -g 1 > out/mnist/mnist_corr_w1.0_seed0.txt &

# wait

# cifar10
mkdir -p out/cifar10
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 0.1 -s 0 -g 0 > out/cifar10/cifar10_imp_w0.1_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 1.0 -s 0 -g 0 > out/cifar10/cifar10_imp_w1.0_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 10 -s 0 -g 1 > out/cifar10/cifar10_imp_w10.0_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 100 -s 0 -g 1 > out/cifar10/cifar10_imp_w100.0_seed0.txt &
wait
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 0 > out/cifar10/cifar10_corr_w0.0_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.3 -s 0 -g 0 > out/cifar10/cifar10_corr_w0.3_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.6 -s 0 -g 1 > out/cifar10/cifar10_corr_w0.6_seed0.txt &
python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 1.0 -s 0 -g 1 > out/cifar10/cifar10_corr_w1.0_seed0.txt &
wait
# # wait
