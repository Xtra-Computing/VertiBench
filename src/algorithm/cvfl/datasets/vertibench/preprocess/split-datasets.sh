## covtype
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &
#
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 &
wait

## msd
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &
#
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 &
wait


## higgs
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.1 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 1.0 -t 0.2 -s 0 &
#
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.0 -t 0.2 -s 0
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.3 -t 0.2 -s 0
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.6 -t 0.2 -s 0
#python src/preprocess/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 1.0 -t 0.2 -s 0
wait

## gisette
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &
#
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 -g 0 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 -g 1 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 -g 2 &
#python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 -g 3 &
wait

# realsim
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &

python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 -g 0 &
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 -g 1 &
python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 -g 2 &
#python src/preprocess/vertical_split.py data/syn/realsim/realsim.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 -g 3 &