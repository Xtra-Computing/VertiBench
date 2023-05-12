#!/usr/bin/env bash

round=1

# covtype
mkdir -p out/covtype
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/covtype/covtype_imp_w0.1_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/covtype/covtype_imp_w1.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp imp -w 10 -s "$round" -g 1 > out/covtype/covtype_imp_w10.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp imp -w 100 -s "$round" -g 1 > out/covtype/covtype_imp_w100.0_seed"$round".txt &

python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/covtype/covtype_corr_w0.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/covtype/covtype_corr_w0.3_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/covtype/covtype_corr_w0.6_seed"$round".txt &
python src/algorithm/SplitNN.py -d covtype -c 7 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/covtype/covtype_corr_w1.0_seed"$round".txt &

#wait

## msd
#mkdir -p out/msd
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/msd/msd_imp_w0.1_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/msd/msd_imp_w1.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp imp -w 10 -s "$round" -g 1 > out/msd/msd_imp_w10.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp imp -w 100 -s "$round" -g 1 > out/msd/msd_imp_w100.0_seed"$round".txt &

#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/msd/msd_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/msd/msd_corr_w0.3_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/msd/msd_corr_w0.6_seed"$round".txt &
#python src/algorithm/SplitNN.py -d msd -c 1 -m rmse -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/msd/msd_corr_w1.0_seed"$round".txt &

#wait

## gisette
#mkdir -p out/gisette
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/gisette/gisette_imp_w0.1_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/gisette/gisette_imp_w1.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp imp -w 10 -s "$round" -g 1 > out/gisette/gisette_imp_w10.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp imp -w 100 -s "$round" -g 1 > out/gisette/gisette_imp_w100.0_seed"$round".txt &

#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/gisette/gisette_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/gisette/gisette_corr_w0.3_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/gisette/gisette_corr_w0.6_seed"$round".txt &
#python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/gisette/gisette_corr_w1.0_seed"$round".txt &

#wait

## realsim
#mkdir -p out/realsim
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/realsim/realsim_imp_w0.1_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/realsim/realsim_imp_w1.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp imp -w 10 -s "$round" -g 1 > out/realsim/realsim_imp_w10.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp imp -w 100 -s "$round" -g 1 > out/realsim/realsim_imp_w100.0_seed"$round".txt &

#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/realsim/realsim_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/realsim/realsim_corr_w0.3_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/realsim/realsim_corr_w0.6_seed"$round".txt &
#python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/realsim/realsim_corr_w1.0_seed"$round".txt &

#wait

# letter
mkdir -p out/letter
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp imp -w 0.1 -s "$round" -lr 3e-3 -g 0 > out/letter/letter_imp_w0.1_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp imp -w 1.0 -s "$round" -lr 3e-3 -g 0 > out/letter/letter_imp_w1.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp imp -w 10 -s "$round" -lr 3e-3 -g 1 > out/letter/letter_imp_w10.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp imp -w 100 -s "$round" -lr 3e-3 -g 1 > out/letter/letter_imp_w100.0_seed"$round".txt &

python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp corr -b 0.0 -s "$round" -lr 3e-3 -g 2 > out/letter/letter_corr_w0.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp corr -b 0.3 -s "$round" -lr 3e-3 -g 2 > out/letter/letter_corr_w0.3_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp corr -b 0.6 -s "$round" -lr 3e-3 -g 3 > out/letter/letter_corr_w0.6_seed"$round".txt &
python src/algorithm/SplitNN.py -d letter -c 26 -m acc -p 4 -sp corr -b 1.0 -s "$round" -lr 3e-3 -g 3 > out/letter/letter_corr_w1.0_seed"$round".txt &
#wait


# radar
mkdir -p out/radar
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/radar/radar_imp_w0.1_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/radar/radar_imp_w1.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 10 -s "$round" -g 1 > out/radar/radar_imp_w10.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp imp -w 100 -s "$round" -g 1 > out/radar/radar_imp_w100.0_seed"$round".txt &

python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/radar/radar_corr_w0.0_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/radar/radar_corr_w0.3_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/radar/radar_corr_w0.6_seed"$round".txt &
python src/algorithm/SplitNN.py -d radar -c 7 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/radar/radar_corr_w1.0_seed"$round".txt &
#wait


## epsilon
#mkdir -p out/epsilon
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/epsilon/epsilon_imp_w0.1_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 0 > out/epsilon/epsilon_imp_w1.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp imp -w 10 -s "$round" -g 1 > out/epsilon/epsilon_imp_w10.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp imp -w 100 -s "$round" -g 1 > out/epsilon/epsilon_imp_w100.0_seed"$round".txt &

#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 2 > out/epsilon/epsilon_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 2 > out/epsilon/epsilon_corr_w0.3_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 3 > out/epsilon/epsilon_corr_w0.6_seed"$round".txt &
#python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/epsilon/epsilon_corr_w1.0_seed"$round".txt &
#wait


## higgs
#mkdir -p out/higgs
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp imp -w 0.1 -s "$round" -g 0 > out/higgs/higgs_imp_w0.1_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp imp -w 1.0 -s "$round" -g 1 > out/higgs/higgs_imp_w1.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp imp -w 10 -s "$round" -g 2 > out/higgs/higgs_imp_w10.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp imp -w 100 -s "$round" -g 3 > out/higgs/higgs_imp_w100.0_seed"$round".txt &
#wait
#
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp corr -b 0.0 -s "$round" -g 0 > out/higgs/higgs_corr_w0.0_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp corr -b 0.3 -s "$round" -g 1 > out/higgs/higgs_corr_w0.3_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp corr -b 0.6 -s "$round" -g 2 > out/higgs/higgs_corr_w0.6_seed"$round".txt &
#python src/algorithm/SplitNN.py -d higgs -c 2 -m acc -p 4 -sp corr -b 1.0 -s "$round" -g 3 > out/higgs/higgs_corr_w1.0_seed"$round".txt &
#wait
