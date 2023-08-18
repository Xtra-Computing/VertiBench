#!/usr/bin/bash

# mnist
# dataset=mnist
# n_class=10
# root_dir=/data/zhaomin/VertiBench/data/syn/$dataset

# for i in 1 2 3 4
# do
#     python src/preprocess/pkl_to_csv.py $root_dir/*seed$i*.pkl
#     mkdir -p out/fedtree/$dataset

#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s $i > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s $i > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s $i > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s $i > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$i".txt &

#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$i".txt &
#     python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s $i > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$i".txt &
#     wait
# done
# rm $root_dir/*seed1*.csv


# cifar10
dataset=cifar10
n_class=10
root_dir=/data/junyi/syn/$dataset
for i in 1 2 3 4
do
    python src/preprocess/pkl_to_csv.py $root_dir/*seed$i*.pkl
    mkdir -p out/fedtree/$dataset

    echo "cifar10 imp"
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s $i > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s $i > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s $i > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s $i > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$i".txt &

    echo "cifar10 corr"
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s $i > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$i".txt &
    python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s $i > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$i".txt &
    wait
done
# rm $root_dir/*seed"$i"*.csv

