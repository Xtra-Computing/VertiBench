#!/bin/bash

ngpu=8
cnt=0
#for seed in {5..50}; do
#  fmt=libsvm
#  for dataset in letter; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.1 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 10.0 -t 0.2 -s $seed -g 0 &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 100.0 -t 0.2 -s $seed -g 1 &
#
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
##    cnt=$(((cnt + 1) % $ngpu))
##    if [ $cnt -eq 0 ]; then
##      wait
##    fi
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.3 -t 0.2 -s $seed -g $cnt &
##    cnt=$(((cnt + 1) % $ngpu))
##    if [ $cnt -eq 0 ]; then
##      wait
##    fi
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.6 -t 0.2 -s $seed -g $cnt &
##    cnt=$(((cnt + 1) % $ngpu))
##    if [ $cnt -eq 0 ]; then
##      wait
##    fi
##    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 1.0 -t 0.2 -s $seed -g $cnt &
##    cnt=$(((cnt + 1) % $ngpu))
##    if [ $cnt -eq 0 ]; then
##      wait
##    fi
#  done
#done
#wait

for seed in {5..50}; do
  fmt=csv
  for dataset in radar; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 0.1 -t 0.2 -s $seed &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 1.0 -t 0.2 -s $seed &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 10.0 -t 0.2 -s $seed -g 2 &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w 100.0 -t 0.2 -s $seed -g 3 &

#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.3 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 0.6 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b 1.0 -t 0.2 -s $seed -g $cnt &
#    cnt=$(((cnt + 1) % $ngpu))
#    if [ $cnt -eq 0 ]; then
#      wait
#    fi
  done
done

