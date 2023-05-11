#!/usr/bin/env bash


# satellite
mkdir -p out/splitnn/satellite
python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 16 -lr 1e-5 -bs 32 -g 0 > out/splitnn/satellite/satellite_seed0.txt &
python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 16 -lr 1e-5 -bs 32 -g 1 > out/splitnn/satellite/satellite_seed1.txt &
python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 16 -lr 1e-5 -bs 32 -g 2 > out/splitnn/satellite/satellite_seed2.txt &
python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 16 -lr 1e-5 -bs 32 -g 3 > out/splitnn/satellite/satellite_seed3.txt &
python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 16 -lr 1e-5 -bs 32 -g 3 > out/splitnn/satellite/satellite_seed4.txt &