#!/usr/bin/bash


# vehicle
mkdir -p out/fedtree/vehicle
python src/preprocess/pkl_to_csv.py data/real/vehicle/processed/*.pkl
python src/algorithm/FedTree.py -d vehicle -c 3 -p 2 --is-real --root data/real/vehicle/processed/ -s 0 > out/fedtree/vehicle/vehicle_seed0.txt &
python src/algorithm/FedTree.py -d vehicle -c 3 -p 2 --is-real --root data/real/vehicle/processed/ -s 1 > out/fedtree/vehicle/vehicle_seed1.txt &
python src/algorithm/FedTree.py -d vehicle -c 3 -p 2 --is-real --root data/real/vehicle/processed/ -s 2 > out/fedtree/vehicle/vehicle_seed2.txt &
python src/algorithm/FedTree.py -d vehicle -c 3 -p 2 --is-real --root data/real/vehicle/processed/ -s 3 > out/fedtree/vehicle/vehicle_seed3.txt &
python src/algorithm/FedTree.py -d vehicle -c 3 -p 2 --is-real --root data/real/vehicle/processed/ -s 4 > out/fedtree/vehicle/vehicle_seed4.txt &

wait


# wide
mkdir -p out/fedtree/wide
python src/preprocess/pkl_to_csv.py data/real/wide/processed/*.pkl --scale-y
python src/algorithm/FedTree.py -d wide -c 2 -p 5 --is-real --root data/real/wide/processed/ -s 0 > out/fedtree/wide/wide_seed0.txt &
python src/algorithm/FedTree.py -d wide -c 2 -p 5 --is-real --root data/real/wide/processed/ -s 1 > out/fedtree/wide/wide_seed1.txt &
python src/algorithm/FedTree.py -d wide -c 2 -p 5 --is-real --root data/real/wide/processed/ -s 2 > out/fedtree/wide/wide_seed2.txt &
python src/algorithm/FedTree.py -d wide -c 2 -p 5 --is-real --root data/real/wide/processed/ -s 3 > out/fedtree/wide/wide_seed3.txt &
python src/algorithm/FedTree.py -d wide -c 2 -p 5 --is-real --root data/real/wide/processed/ -s 4 > out/fedtree/wide/wide_seed4.txt &
wait
