#!/usr/bin/env bash

data=$1

mkdir -p out/"$data"
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 0 > out/"$data"/"$data"_corr_w0.0_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp corr -b 0.3 -s 0 -g 0 > out/"$data"/"$data"_corr_w0.3_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp corr -b 0.6 -s 0 -g 1 > out/"$data"/"$data"_corr_w0.6_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp corr -b 1.0 -s 0 -g 1 > out/"$data"/"$data"_corr_w1.0_seed0.txt &
