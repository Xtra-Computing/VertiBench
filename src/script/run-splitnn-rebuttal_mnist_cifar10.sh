#!/usr/bin/env bash
# cifar10
dataset=mnist
n_class=10
root_dir=/data/zhaomin/VertiBench/data/syn/$dataset
for i in 1 2 3 4
do
    mkdir -p out/mnist
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 0.1 -s $i -g 0 > out/mnist/mnist_imp_w0.1_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 1.0 -s $i -g 1 > out/mnist/mnist_imp_w1.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 10 -s $i -g 2 > out/mnist/mnist_imp_w10.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp imp -w 100 -s $i -g 3 > out/mnist/mnist_imp_w100.0_seed"$i".txt &
    wait
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.0 -s $i -g 0 > out/mnist/mnist_corr_w0.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.3 -s $i -g 1 > out/mnist/mnist_corr_w0.3_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 0.6 -s $i -g 2 > out/mnist/mnist_corr_w0.6_seed"$i".txt &
    python src/algorithm/SplitNN.py -d mnist -c 10 -m acc -p 4 -sp corr -b 1.0 -s $i -g 3 > out/mnist/mnist_corr_w1.0_seed"$i".txt &
    wait
done



# cifar10
dataset=cifar10
n_class=10
root_dir=/data/zhaomin/VertiBench/data/syn/$dataset
for i in 1 2 3 4
do
    mkdir -p out/cifar10
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 0.1 -s $i -g 0 > out/cifar10/cifar10_imp_w0.1_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 1.0 -s $i -g 1 > out/cifar10/cifar10_imp_w1.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 10 -s $i -g 2 > out/cifar10/cifar10_imp_w10.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp imp -w 100 -s $i -g 3 > out/cifar10/cifar10_imp_w100.0_seed"$i".txt &
    wait
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.0 -s $i -g 0 > out/cifar10/cifar10_corr_w0.0_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.3 -s $i -g 1 > out/cifar10/cifar10_corr_w0.3_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 0.6 -s $i -g 2 > out/cifar10/cifar10_corr_w0.6_seed"$i".txt &
    python src/algorithm/SplitNN.py -d cifar10 -c 10 -m acc -p 4 -sp corr -b 1.0 -s $i -g 3 > out/cifar10/cifar10_corr_w1.0_seed"$i".txt &
    wait
done

