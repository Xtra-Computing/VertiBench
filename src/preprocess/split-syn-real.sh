#!/bin/bash

ngpu=8
cnt=0
#for seed in 0 1 2 3 4; do
#  fmt=libsvm
#  for dataset in covtype msd letter; do
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp imp -w 9.39 -t 0.2 -s $seed -g 1 &
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp imp -w 1.39 -t 0.2 -s $seed -g 1 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#  done
#done
#
#for seed in 0 1 2 3 4; do
#  fmt=csv
#  for dataset in radar; do
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp imp -w 9.39 -t 0.2 -s $seed -g 1 &
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp imp -w 1.39 -t 0.2 -s $seed -g 1 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#  done
#done

for seed in 0 1 2 3 4; do
  fmt=libsvm
  for dataset in epsilon; do
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp imp -w 9.39 -t 0.2 -s $seed -g 1 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp imp -w 1.39 -t 0.2 -s $seed -g 1 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 2 -sp corr -b 0.0 -t 0.2 -s $seed --fast -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 5 -sp corr -b 0.0 -t 0.2 -s $seed --fast -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
  done
done