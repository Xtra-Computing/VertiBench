#!/bin/bash


for seed in 1 2 3 4; do
  fmt=libsvm
  for dataset in covtype msd gisette realsim letter epsilon; do
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.3 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.6 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s $seed &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s $seed -g 0 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s $seed -g 1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s $seed -g 2 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s $seed -g 3 &
    wait
  done

  fmt=csv
  for dataset in higgs; do
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.3 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.6 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s $seed &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s $seed -g 0 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s $seed -g 1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s $seed -g 2 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s $seed -g 3 &
    wait
  done
done