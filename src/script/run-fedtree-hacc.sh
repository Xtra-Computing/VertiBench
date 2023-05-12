#!/usr/bin/bash

seed=$1

# covtype
dataset=covtype
n_class=7
root_dir=data/syn/$dataset
python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl
mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt

python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt

rm $root_dir/*seed"$seed"*.csv


# msd
dataset=msd
n_class=1
root_dir=data/syn/$dataset
python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl
mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt

python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt

rm $root_dir/*seed"$seed"*.csv


# gisette
dataset=gisette
n_class=2
root_dir=data/syn/$dataset
python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl -sy
mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt

python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt

rm $root_dir/*seed"$seed"*.csv
#
## epsilon
#dataset=epsilon
#n_class=2
#root_dir=data/syn/$dataset
#python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl -sy
#mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt
#
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt
#
#rm $root_dir/*seed"$seed"*.csv
#
#
## realsim
#dataset=realsim
#n_class=2
#root_dir=data/syn/$dataset
#python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl -sy
#mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt
#
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt
#
#rm $root_dir/*seed"$seed"*.csv
#
#
## letter
#dataset=letter
#n_class=26
#root_dir=data/syn/$dataset
#python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl
#mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt
#
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt
#
#rm $root_dir/*seed"$seed"*.csv
#
#
## radar
#dataset=radar
#n_class=7
#root_dir=data/syn/$dataset
#python src/preprocess/pkl_to_csv.py $root_dir/*seed"$seed"*.pkl
#mkdir -p out/fedtree/$dataset
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 0.1 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w0.1_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w1.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 10 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w10.0_seed"$seed".txt
#python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp imp -w 100 -s "$seed" > out/fedtree/$dataset/"$dataset"_imp_w100.0_seed"$seed".txt
#
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.0_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.3 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.3_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 0.6 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w0.6_seed"$seed".txt
##python src/algorithm/FedTree.py -d $dataset -c $n_class -p 4 -sp corr -b 1.0 -s "$seed" > out/fedtree/$dataset/"$dataset"_corr_w1.0_seed"$seed".txt
#
#rm $root_dir/*seed"$seed"*.csv
#
