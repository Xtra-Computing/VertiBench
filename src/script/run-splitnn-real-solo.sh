#!/usr/bin/env bash


# satellite solo
mkdir -p out/splitnn/satellite
for p in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 1 -lr 1e-5 -bs 32 -pp $p -g 0 > out/splitnn/satellite/satellite_solo"$p"_seed0.txt &
  python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 1 -lr 1e-5 -bs 32 -pp $p -g 1 > out/splitnn/satellite/satellite_solo"$p"_seed1.txt &
  python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 1 -lr 1e-5 -bs 32 -pp $p -g 2 > out/splitnn/satellite/satellite_solo"$p"_seed2.txt &
  python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 1 -lr 1e-5 -bs 32 -pp $p -g 3 > out/splitnn/satellite/satellite_solo"$p"_seed3.txt &
  python src/algorithm/SplitNN.py -d satellite -c 4 -m acc -p 1 -lr 1e-5 -bs 32 -pp $p -g 0 > out/splitnn/satellite/satellite_solo"$p"_seed4.txt &
  wait
done