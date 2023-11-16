#!/bin/bash

gs=( "$@" )

for seed in 0; do
  fmt=libsvm
  dataset=covtype
  ws=(46.17 0.22 30.72 0.21 33.17 0.81 0.16 7.26 20.27 3.63)
  for i in "${!ws[@]}"; do
    w=${ws[$i]}
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w $w -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
    # wait every 5 jobs
    if (( $i % 5 == 4 )); then
      wait
    fi
  done


  dataset=msd
  ws=(0.34 3.23 1.17 10.71 10.71 4.95 0.17 41.28 0.22 14.93)
  for i in "${!ws[@]}"; do
    w=${ws[$i]}
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w $w -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
    # wait every 5 jobs
    if (( $i % 5 == 4 )); then
      wait
    fi
  done

  dataset=letter
  ws=(18.04 0.69 2.12 0.65 37.82 74.55 3.72 28.29 3.60 91.28)
  for i in "${!ws[@]}"; do
    w=${ws[$i]}
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w $w -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
    # wait every 5 jobs
    if (( $i % 5 == 4 )); then
      wait
    fi
  done

  dataset=gisette
  ws=(0.21 19.17 10.57 0.97 0.78 0.20 9.56 22.70 10.42 2.41)
  for i in "${!ws[@]}"; do
    w=${ws[$i]}
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w $w -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
    # wait every 5 jobs
    if (( $i % 5 == 4 )); then
      wait
    fi
  done

  fmt=csv
  dataset=radar
  ws=(1.45 0.17 88.73 12.94 0.34 0.17 0.18 0.18 26.50 22.87)
  for i in "${!ws[@]}"; do
    w=${ws[$i]}
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" -p 4 -sp imp -w $w -t 0.2 -s $seed -v -g ${gs[$((i % 5))]} --decimal 2 &
    # wait every 5 jobs
    if (( $i % 5 == 4 )); then
      wait
    fi
  done
done