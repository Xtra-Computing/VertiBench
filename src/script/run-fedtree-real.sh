#!/usr/bin/bash

# covtype
dataset=satellite
n_class=4
n_parties=16
mkdir -p out/fedtree/$dataset
root_dir=data/real/$dataset/cache

#python src/preprocess/pkl_to_csv.py "$root_dir"/*party0*pkl &
#shopt -s extglob
#printf '%s\0' "$root_dir"/!(*party0*pkl) | xargs -r -0 -P16 -n1 python src/preprocess/pkl_to_csv.py -dy
#shopt -u extglob
#wait
python src/algorithm/FedTree.py -d $dataset -c $n_class -p "$n_parties" -rd -r $root_dir -s 0 > out/fedtree/$dataset/"$dataset"_seed0.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p "$n_parties" -rd -r $root_dir -s 1 > out/fedtree/$dataset/"$dataset"_seed1.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p "$n_parties" -rd -r $root_dir -s 2 > out/fedtree/$dataset/"$dataset"_seed2.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p "$n_parties" -rd -r $root_dir -s 3 > out/fedtree/$dataset/"$dataset"_seed3.txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p "$n_parties" -rd -r $root_dir -s 4 > out/fedtree/$dataset/"$dataset"_seed4.txt

