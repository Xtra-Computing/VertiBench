data=$1
fmt=$2
seed=$3

python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 2 -sp imp -w 1 -t 0.2 -s $seed -g 0
python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 8 -sp imp -w 1 -t 0.2 -s $seed -g 0
python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 32 -sp imp -w 1 -t 0.2 -s $seed -g 1
python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 128 -sp imp -w 1 -t 0.2 -s $seed -g 2
python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 512 -sp imp -w 1 -t 0.2 -s $seed -g 3
python src/preprocess/vertical_split.py data/syn/"$data"/"$data"."$fmt" 2048 -sp imp -w 1 -t 0.2 -s $seed -g 3


