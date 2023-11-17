#!/bin/bash

ngpu=8
cnt=0
for seed in 0 1 2 3 4; do
  fmt=libsvm
  for dataset in covtype msd letter; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.1 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.3 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.6 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.3 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.6 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 1.0 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
  done
done


for seed in 0 1 2 3 4; do
  fmt=csv
  for dataset in radar; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.1 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.3 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.6 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.3 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.6 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 1.0 -t 0.2 -s $seed -g $cnt &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
  done
done


for seed in 0 1 2 3 4; do
  fmt=libsvm
  for dataset in gisette realsim epsilon; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.1 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.3 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.6 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt --fast &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.3 -t 0.2 -s $seed -g $cnt --fast &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.6 -t 0.2 -s $seed -g $cnt --fast &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 1.0 -t 0.2 -s $seed -g $cnt --fast &
    cnt=$(((cnt + 1) % $ngpu))
    if [ $cnt -eq 0 ]; then
      wait
    fi
  done
done

