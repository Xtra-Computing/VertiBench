#!/bin/bash

mkdir -p out/syn-vehicle/
for seed in 0 1 2 3 4; do
  fmt=libsvm
  for dataset in covtype msd gisette realsim epsilon letter; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp imp -w 9.39 -t 0.2 -s $seed -g 1 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp imp -w 1.39 -t 0.2 -s $seed -g 1 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp corr -b 0.0 -t 0.2 -s $seed -g 0 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp corr -b 0.0 -t 0.2 -s $seed -v -g 0
    wait
  done

  fmt=csv
  for dataset in radar; do
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp imp -w 9.39 -t 0.2 -s $seed -g 1 -lc -1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp imp -w 1.39 -t 0.2 -s $seed -g 1 -lc -1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp corr -b 0.0 -t 0.2 -s $seed -v -g 0 -lc -1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp corr -b 0.0 -t 0.2 -s $seed -v -g 1 -lc -1 &
    wait
  done
  wait
done