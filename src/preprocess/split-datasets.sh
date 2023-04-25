fmt=libsvm
for dataset in covtype msd gisette realsim letter epsilon; do
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.3 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.6 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s 0 &

  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s 1 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s 2 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s 3 &
  wait
done

fmt=csv
for dataset in higgs radar; do
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.1 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.3 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 0.6 -t 0.2 -s 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp imp -w 1.0 -t 0.2 -s 0 &

  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.0 -t 0.2 -s 0 -g 0 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.3 -t 0.2 -s 0 -g 1 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 0.6 -t 0.2 -s 0 -g 2 &
  python src/preprocess/vertical_split.py data/syn/"$dataset"/"$dataset"."$fmt" 4 -sp corr -b 1.0 -t 0.2 -s 0 -g 3 &
  wait
done
