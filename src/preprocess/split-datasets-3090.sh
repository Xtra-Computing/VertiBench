#!/bin/bash

python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.0 -t 0.2 -s 3 -g 0 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.3 -t 0.2 -s 3 -g 1 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.6 -t 0.2 -s 3 -g 2 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 1.0 -t 0.2 -s 3 -g 3 --fast &
wait
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.0 -t 0.2 -s 4 -g 4 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.3 -t 0.2 -s 4 -g 5 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 0.6 -t 0.2 -s 4 -g 6 --fast &
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm -p 4 -sp corr -b 1.0 -t 0.2 -s 4 -g 7 --fast &

