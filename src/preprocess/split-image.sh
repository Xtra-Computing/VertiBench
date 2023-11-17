#!/bin/bash

ngpu=8
cnt=0
for seed in 0; do
  fmt=csv
  for dataset in mnist; do

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
    -p 4 -sp corr -b 0.0 -t 0 -s $seed --split-image --fast -cf spearmanr_pandas -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 0.3 -t 0 -s $seed --split-image --fast -cf spearmanr_pandas -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 0.6 -t 0 -s $seed --split-image --fast -cf spearmanr_pandas -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 1.0 -t 0 -s $seed --split-image --fast -cf spearmanr_pandas -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
  done

  for dataset in cifar10; do

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
    -p 4 -sp corr -b 0.0 -t 0 -s $seed --split-image --fast -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 0.3 -t 0 -s $seed --split-image --fast -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 0.6 -t 0 -s $seed --split-image --fast -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"_train."$fmt" data/syn/"$dataset"/"$dataset"_test."$fmt" \
     -p 4 -sp corr -b 1.0 -t 0 -s $seed --split-image --fast -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
  done
done

