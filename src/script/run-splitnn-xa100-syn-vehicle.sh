#!/usr/bin/env bash

round=$1

# covtype
mkdir -p out/syn_real/covtype
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 0 > out/syn_real/covtype/covtype_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 0 > out/syn_real/covtype/covtype_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/covtype/covtype_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/covtype/covtype_p5_corr_w0.0_seed"$round".txt &


# msd
mkdir -p out/syn_real/msd
python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 2 -sp imp -w 9.4 -s 0 -g 1 > out/syn_real/msd/msd_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 5 -sp imp -w 1.4 -s 0 -g 1 > out/syn_real/msd/msd_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 2 -sp corr -b 0.0 -s 0 -g 1 > out/syn_real/msd/msd_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/msd/msd_p5_corr_w0.0_seed"$round".txt &


# gisette
mkdir -p out/syn_real/gisette
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 2 > out/syn_real/gisette/gisette_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 2 > out/syn_real/gisette/gisette_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 2 > out/syn_real/gisette/gisette_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/gisette/gisette_p5_corr_w0.0_seed"$round".txt &



# realsim
mkdir -p out/syn_real/realsim
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 3 > out/syn_real/realsim/realsim_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 3 > out/syn_real/realsim/realsim_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 3 > out/syn_real/realsim/realsim_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/realsim/realsim_p5_corr_w0.0_seed"$round".txt &

wait

# letter
mkdir -p out/syn_real/letter
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 0 > out/syn_real/letter/letter_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 0 > out/syn_real/letter/letter_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/letter/letter_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/letter/letter_p5_corr_w0.0_seed"$round".txt &


# epsilon
mkdir -p out/syn_real/epsilon
python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 1 > out/syn_real/epsilon/epsilon_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 1 > out/syn_real/epsilon/epsilon_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 1 > out/syn_real/epsilon/epsilon_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/epsilon/epsilon_p5_corr_w0.0_seed"$round".txt &

# radar
mkdir -p out/syn_real/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 2 -sp imp -w 9.4 -s 0 -g 2 > out/syn_real/radar/radar_p2_imp_w9.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 5 -sp imp -w 1.4 -s 0 -g 2 > out/syn_real/radar/radar_p5_imp_w1.4_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 2 > out/syn_real/radar/radar_p2_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 5 -sp corr -b 0.0 -s 0 -g 0 > out/syn_real/radar/radar_p5_corr_w0.0_seed"$round".txt &
wait
