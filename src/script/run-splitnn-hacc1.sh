#!/usr/bin/env bash

# radar
mkdir -p out/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.1 -s 0 -g 0 > out/radar/radar_imp_w0.1_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.3 -s 0 -g 0 > out/radar/radar_imp_w0.3_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.6 -s 0 -g 1 > out/radar/radar_imp_w0.6_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 1.0 -s 0 -g 1 > out/radar/radar_imp_w1.0_seed0.txt &

python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 2 > out/radar/radar_corr_w0.0_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.3 -s 0 -g 2 > out/radar/radar_corr_w0.3_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.6 -s 0 -g 3 > out/radar/radar_corr_w0.6_seed0.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 1.0 -s 0 -g 3 > out/radar/radar_corr_w1.0_seed0.txt &



# radar
mkdir -p out/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.1 -s 1 -g 0 > out/radar/radar_imp_w0.1_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.3 -s 1 -g 0 > out/radar/radar_imp_w0.3_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.6 -s 1 -g 1 > out/radar/radar_imp_w0.6_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 1.0 -s 1 -g 1 > out/radar/radar_imp_w1.0_seed1.txt &

python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.0 -s 1 -g 2 > out/radar/radar_corr_w0.0_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.3 -s 1 -g 2 > out/radar/radar_corr_w0.3_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.6 -s 1 -g 3 > out/radar/radar_corr_w0.6_seed1.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 1.0 -s 1 -g 3 > out/radar/radar_corr_w1.0_seed1.txt &


# radar
mkdir -p out/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.1 -s 2 -g 0 > out/radar/radar_imp_w0.1_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.3 -s 2 -g 0 > out/radar/radar_imp_w0.3_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.6 -s 2 -g 1 > out/radar/radar_imp_w0.6_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 1.0 -s 2 -g 1 > out/radar/radar_imp_w1.0_seed2.txt &

python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.0 -s 2 -g 2 > out/radar/radar_corr_w0.0_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.3 -s 2 -g 2 > out/radar/radar_corr_w0.3_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.6 -s 2 -g 3 > out/radar/radar_corr_w0.6_seed2.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 1.0 -s 2 -g 3 > out/radar/radar_corr_w1.0_seed2.txt &


# radar
mkdir -p out/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.1 -s 3 -g 0 > out/radar/radar_imp_w0.1_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.3 -s 3 -g 0 > out/radar/radar_imp_w0.3_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.6 -s 3 -g 1 > out/radar/radar_imp_w0.6_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 1.0 -s 3 -g 1 > out/radar/radar_imp_w1.0_seed3.txt &

python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.0 -s 3 -g 2 > out/radar/radar_corr_w0.0_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.3 -s 3 -g 2 > out/radar/radar_corr_w0.3_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.6 -s 3 -g 3 > out/radar/radar_corr_w0.6_seed3.txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 1.0 -s 3 -g 3 > out/radar/radar_corr_w1.0_seed3.txt &


