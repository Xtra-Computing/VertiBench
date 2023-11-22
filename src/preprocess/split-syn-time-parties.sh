#!/bin/bash



cnt=0
ni=1000  # number of instances
nf=1000  # number of features
# [10, 100, 500, 1000, 5000, 10000, 50000, 100000]
for np in 2 4 10 25 100 400 800 1000; do
  python src/preprocess/vertical_split.py data/syn/sklearn/syn_"$ni"_"$nf".csv -p 2 -sp corr -b 0.5 -t 0 -v -cf pearson -g $cnt -et --fast \
    1> out/time/syn_"$ni"_"$nf"_"$np".txt 2>> nohup.out &
  cnt=$((cnt+1))
done

wait

