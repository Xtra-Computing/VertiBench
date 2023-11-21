#!/bin/bash

cnt=0
nf=100 # number of features
# [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
for ni in 500 1000 5000 10000 50000 100000 500000 1000000; do
  python src/preprocess/vertical_split.py data/syn/sklearn/syn_"$ni"_"$nf".csv -p 2 -sp corr -b 0.5 -t 0 -v -cf pearson -g $cnt -et --fast \
    1> out/time/syn_"$ni"_"$nf".txt 2>> nohup.out &
  cnt=$((cnt+1))
done

wait

cnt=0
ni=1000  # number of instances
# [10, 100, 500, 1000, 5000, 10000, 50000, 100000]
for nf in 10 100 500 1000 5000 10000 50000 100000; do
  python src/preprocess/vertical_split.py data/syn/sklearn/syn_"$ni"_"$nf".csv -p 2 -sp corr -b 0.5 -t 0 -v -cf pearson -g $cnt -et --fast \
    1> out/time/syn_"$ni"_"$nf".txt 2>> nohup.out &
  cnt=$((cnt+1))
done

wait

