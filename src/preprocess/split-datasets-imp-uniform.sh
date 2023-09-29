#!/bin/bash

gs=( "$@" )
for seed in 101 102 103 104 105; do
  fmt=libsvm
  dataset=covtype
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed -v -g 0 --decimal 2 --uniform &

  dataset=msd
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed -v -g 2 --decimal 2 --uniform &


  dataset=letter
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed -v -g 5 --decimal 2 --uniform &


  dataset=gisette
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed -v -g 6 --decimal 2 --uniform &

  fmt=csv
  dataset=radar
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed -v -g 7 --decimal 2 --uniform &

  wait
done