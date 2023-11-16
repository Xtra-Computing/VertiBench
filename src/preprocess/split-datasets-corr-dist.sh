#!/bin/bash

gs=( "$@" )
for seed in 0; do
#  fmt=libsvm
#  for dataset in covtype; do
#    bs=(0.25 0.55 1.00 0.93 0.97 0.99 0.02 0.58 0.58 0.67)
#    for i in "${!bs[@]}"; do
#      b=${bs[$i]}
#      python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b $b -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
#      # wait every 5 jobs
#      if (( $i % 5 == 4 )); then
#        wait
#      fi
#    done
#  done
#
#  for dataset in msd; do
#    bs=(0.36 0.84 0.59 0.02 0.97 0.07 0.03 0.03 0.65 0.59)
#    for i in "${!bs[@]}"; do
#      b=${bs[$i]}
#      python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b $b -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
#      # wait every 5 jobs
#      if (( $i % 5 == 4 )); then
#        wait
#      fi
#    done
#  done
#
#  for dataset in letter; do
#    bs=(0.44 0.76 0.77 0.16 0.40 0.21 0.13 0.83 0.77 0.06)
#    for i in "${!bs[@]}"; do
#      b=${bs[$i]}
#      python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b $b -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
#      # wait every 5 jobs
#      if (( $i % 5 == 4 )); then
#        wait
#      fi
#    done
#  done
  fmt=libsvm
  for dataset in gisette; do
    bs=(0.80 0.40 0.62 0.38 0.33 0.27 0.85 0.63 0.20 0.34)
    for i in "${!bs[@]}"; do
      b=${bs[$i]}
      python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b $b -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 --fast &
      # wait every 5 jobs
      if (( $i % 5 == 4 )); then
        wait
      fi
    done
  done

  fmt=csv
  for dataset in radar; do
    bs=(0.77 0.48 0.90 0.92 0.59 0.61 0.59 0.28 0.50 0.95)
    for i in "${!bs[@]}"; do
      b=${bs[$i]}
      python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp corr -b $b -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
      # wait every 5 jobs
      if (( $i % 5 == 4 )); then
        wait
      fi
    done
  done
done