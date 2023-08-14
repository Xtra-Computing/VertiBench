#!/usr/bin/bash

# mnist
# dataset=mnist
# n_class=10
# root_dir=/data/zhaomin/VertiBench/data/syn/$dataset
# python src/preprocess/pkl_to_csv.py $root_dir/*seed0*.pkl
# mkdir -p out/fedtree/$dataset
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed0.txt &
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed0.txt &
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed0.txt &
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed0.txt &

# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed0.txt &
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed0.txt &
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed0.txt &
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed0.txt &

# rm $root_dir/*seed0*.csv


# cifar10
dataset=cifar10
n_class=10
root_dir=/data/junyi/syn/$dataset
# python src/preprocess/pkl_to_csv.py $root_dir/*seed0*.pkl
# mkdir -p out/fedtree/$dataset
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed0.txt
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed0.txt
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed0.txt
# python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s 0 > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed0.txt

python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed0.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed0.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed0.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s 0 > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed0.txt

# # rm $root_dir/*seed0*.csv

