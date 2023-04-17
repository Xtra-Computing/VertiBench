# covtype
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &

python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/covtype/covtype.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 &

# msd
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.1 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.3 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 0.6 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp imp -w 1.0 -t 0.2 -s 0 &

python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.0 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.3 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 0.6 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/msd/msd.libsvm 4 -sp corr -b 1.0 -t 0.2 -s 0 &


# higgs
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.1 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.3 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 0.6 -t 0.2 -s 0 &
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp imp -w 1.0 -t 0.2 -s 0 &

python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.0 -t 0.2 -s 0
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.3 -t 0.2 -s 0
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 0.6 -t 0.2 -s 0
python src/script/vertical_split.py data/syn/higgs/higgs.csv 4 -sp corr -b 1.0 -t 0.2 -s 0

