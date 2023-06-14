#!/bin/bash

mkdir -p out/time/
for seed in 0; do
  fmt=libsvm
  for dataset in epsilon; do
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_0.1_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_1.0_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 10  -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_10_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 100 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_100_seed"$seed".txt &

    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s $seed -et -v -g 0 > out/time/"$dataset"_corr_0.0_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s $seed -et -v -g 3 > out/time/"$dataset"_corr_0.3_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s $seed -et -v -g 6 > out/time/"$dataset"_corr_0.6_seed"$seed".txt &
    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s $seed -et -v -g 7 > out/time/"$dataset"_corr_1.0_seed"$seed".txt &
    wait
  done

#  fmt=csv
#  for dataset in radar; do
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_0.1_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_1.0_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 10  -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_10_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 100 -t 0.2 -s $seed -et -v > out/time/"$dataset"_imp_100_seed"$seed".txt &
#
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s $seed -et -v -g 0 > out/time/"$dataset"_corr_0.0_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s $seed -et -v -g 3 > out/time/"$dataset"_corr_0.3_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s $seed -et -v -g 6 > out/time/"$dataset"_corr_0.6_seed"$seed".txt &
#    python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s $seed -et -v -g 7 > out/time/"$dataset"_corr_1.0_seed"$seed".txt &
#    wait
#  done

  wait
done