#!/usr/bin/env bash

# vehicle
mkdir -p out/vehicle
python src/algorithm/SplitNN.py -d vehicle -c 3 -m acc -p 2 -s 0 -g 0 > out/vehicle/vehicle_seed0.txt &
python src/algorithm/SplitNN.py -d vehicle -c 3 -m acc -p 2 -s 1 -g 0 > out/vehicle/vehicle_seed1.txt &
python src/algorithm/SplitNN.py -d vehicle -c 3 -m acc -p 2 -s 2 -g 1 > out/vehicle/vehicle_seed2.txt &
python src/algorithm/SplitNN.py -d vehicle -c 3 -m acc -p 2 -s 3 -g 2 > out/vehicle/vehicle_seed3.txt &
python src/algorithm/SplitNN.py -d vehicle -c 3 -m acc -p 2 -s 4 -g 3 > out/vehicle/vehicle_seed4.txt &
wait


# wide
mkdir -p out/wide
python src/algorithm/SplitNN.py -d wide -c 2 -m acc -p 5 -s 0 -g 3 -lr 1e-4 > out/wide/wide_seed0.txt &
python src/algorithm/SplitNN.py -d wide -c 2 -m acc -p 5 -s 1 -g 4 -lr 1e-4 > out/wide/wide_seed1.txt &
python src/algorithm/SplitNN.py -d wide -c 2 -m acc -p 5 -s 2 -g 5 -lr 1e-4 > out/wide/wide_seed2.txt &
python src/algorithm/SplitNN.py -d wide -c 2 -m acc -p 5 -s 3 -g 6 -lr 1e-4 > out/wide/wide_seed3.txt &
python src/algorithm/SplitNN.py -d wide -c 2 -m acc -p 5 -s 4 -g 7 -lr 1e-4 > out/wide/wide_seed4.txt &
wait