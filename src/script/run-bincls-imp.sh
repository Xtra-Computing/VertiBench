#!/usr/bin/env bash

data=$1

mkdir -p out/"$data"
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp imp -w 0.1 -s 0 -g 2 > out/"$data"/"$data"_imp_w0.1_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp imp -w 0.3 -s 0 -g 2 > out/"$data"/"$data"_imp_w0.3_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp imp -w 0.6 -s 0 -g 3 > out/"$data"/"$data"_imp_w0.6_seed0.txt &
python src/algorithm/SplitNN.py -d "$data" -c 2 -m acc -p 4 -sp imp -w 1.0 -s 0 -g 3 > out/"$data"/"$data"_imp_w1.0_seed0.txt &
