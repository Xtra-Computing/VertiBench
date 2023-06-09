mkdir -p out/gisette_scale_test
mkdir -p out/realsim_scale_test
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 1.0 -s 0 -g 0 > out/gisette_scale_test/gisette_imp_w1.0_party2_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 1.0 -s 1 -g 0 > out/gisette_scale_test/gisette_imp_w1.0_party2_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 1.0 -s 2 -g 0 > out/gisette_scale_test/gisette_imp_w1.0_party2_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 1.0 -s 3 -g 0 > out/gisette_scale_test/gisette_imp_w1.0_party2_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2 -sp imp -w 1.0 -s 4 -g 0 > out/gisette_scale_test/gisette_imp_w1.0_party2_seed4.txt &

python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 8 -sp imp -w 1.0 -s 0 -g 1 > out/gisette_scale_test/gisette_imp_w1.0_party8_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 8 -sp imp -w 1.0 -s 1 -g 1 > out/gisette_scale_test/gisette_imp_w1.0_party8_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 8 -sp imp -w 1.0 -s 2 -g 1 > out/gisette_scale_test/gisette_imp_w1.0_party8_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 8 -sp imp -w 1.0 -s 3 -g 1 > out/gisette_scale_test/gisette_imp_w1.0_party8_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 8 -sp imp -w 1.0 -s 4 -g 1 > out/gisette_scale_test/gisette_imp_w1.0_party8_seed4.txt &
🟢
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 32 -sp imp -w 1.0 -s 0 -g 2 > out/gisette_scale_test/gisette_imp_w1.0_party32_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 32 -sp imp -w 1.0 -s 1 -g 2 > out/gisette_scale_test/gisette_imp_w1.0_party32_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 32 -sp imp -w 1.0 -s 2 -g 2 > out/gisette_scale_test/gisette_imp_w1.0_party32_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 32 -sp imp -w 1.0 -s 3 -g 2 > out/gisette_scale_test/gisette_imp_w1.0_party32_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 32 -sp imp -w 1.0 -s 4 -g 2 > out/gisette_scale_test/gisette_imp_w1.0_party32_seed4.txt &
🟢
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 128 -sp imp -w 1.0 -s 0 -g 3 > out/gisette_scale_test/gisette_imp_w1.0_party128_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 128 -sp imp -w 1.0 -s 1 -g 3 > out/gisette_scale_test/gisette_imp_w1.0_party128_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 128 -sp imp -w 1.0 -s 2 -g 3 > out/gisette_scale_test/gisette_imp_w1.0_party128_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 128 -sp imp -w 1.0 -s 3 -g 3 > out/gisette_scale_test/gisette_imp_w1.0_party128_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 128 -sp imp -w 1.0 -s 4 -g 3 > out/gisette_scale_test/gisette_imp_w1.0_party128_seed4.txt &

python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 512 -sp imp -w 1.0 -s 0 -g 4 > out/gisette_scale_test/gisette_imp_w1.0_party512_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 512 -sp imp -w 1.0 -s 1 -g 4 > out/gisette_scale_test/gisette_imp_w1.0_party512_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 512 -sp imp -w 1.0 -s 2 -g 4 > out/gisette_scale_test/gisette_imp_w1.0_party512_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 512 -sp imp -w 1.0 -s 3 -g 4 > out/gisette_scale_test/gisette_imp_w1.0_party512_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 512 -sp imp -w 1.0 -s 4 -g 4 > out/gisette_scale_test/gisette_imp_w1.0_party512_seed4.txt &

python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 0 -g 5 > out/gisette_scale_test/gisette_imp_w1.0_party2048_seed0.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 1 -g 5 > out/gisette_scale_test/gisette_imp_w1.0_party2048_seed1.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 2 -g 5 > out/gisette_scale_test/gisette_imp_w1.0_party2048_seed2.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 3 -g 5 > out/gisette_scale_test/gisette_imp_w1.0_party2048_seed3.txt &
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 4 -g 5 > out/gisette_scale_test/gisette_imp_w1.0_party2048_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 1.0 -s 0 -g 6 > out/realsim_scale_test/realsim_imp_w1.0_party2_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 1.0 -s 1 -g 6 > out/realsim_scale_test/realsim_imp_w1.0_party2_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 1.0 -s 2 -g 6 > out/realsim_scale_test/realsim_imp_w1.0_party2_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 1.0 -s 3 -g 6 > out/realsim_scale_test/realsim_imp_w1.0_party2_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2 -sp imp -w 1.0 -s 4 -g 6 > out/realsim_scale_test/realsim_imp_w1.0_party2_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 8 -sp imp -w 1.0 -s 0 -g 7 > out/realsim_scale_test/realsim_imp_w1.0_party8_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 8 -sp imp -w 1.0 -s 1 -g 7 > out/realsim_scale_test/realsim_imp_w1.0_party8_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 8 -sp imp -w 1.0 -s 2 -g 7 > out/realsim_scale_test/realsim_imp_w1.0_party8_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 8 -sp imp -w 1.0 -s 3 -g 7 > out/realsim_scale_test/realsim_imp_w1.0_party8_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 8 -sp imp -w 1.0 -s 4 -g 7 > out/realsim_scale_test/realsim_imp_w1.0_party8_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 32 -sp imp -w 1.0 -s 0 -g 0 > out/realsim_scale_test/realsim_imp_w1.0_party32_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 32 -sp imp -w 1.0 -s 1 -g 0 > out/realsim_scale_test/realsim_imp_w1.0_party32_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 32 -sp imp -w 1.0 -s 2 -g 0 > out/realsim_scale_test/realsim_imp_w1.0_party32_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 32 -sp imp -w 1.0 -s 3 -g 0 > out/realsim_scale_test/realsim_imp_w1.0_party32_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 32 -sp imp -w 1.0 -s 4 -g 0 > out/realsim_scale_test/realsim_imp_w1.0_party32_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 128 -sp imp -w 1.0 -s 0 -g 1 > out/realsim_scale_test/realsim_imp_w1.0_party128_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 128 -sp imp -w 1.0 -s 1 -g 1 > out/realsim_scale_test/realsim_imp_w1.0_party128_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 128 -sp imp -w 1.0 -s 2 -g 1 > out/realsim_scale_test/realsim_imp_w1.0_party128_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 128 -sp imp -w 1.0 -s 3 -g 1 > out/realsim_scale_test/realsim_imp_w1.0_party128_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 128 -sp imp -w 1.0 -s 4 -g 1 > out/realsim_scale_test/realsim_imp_w1.0_party128_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 512 -sp imp -w 1.0 -s 0 -g 2 > out/realsim_scale_test/realsim_imp_w1.0_party512_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 512 -sp imp -w 1.0 -s 1 -g 2 > out/realsim_scale_test/realsim_imp_w1.0_party512_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 512 -sp imp -w 1.0 -s 2 -g 2 > out/realsim_scale_test/realsim_imp_w1.0_party512_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 512 -sp imp -w 1.0 -s 3 -g 2 > out/realsim_scale_test/realsim_imp_w1.0_party512_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 512 -sp imp -w 1.0 -s 4 -g 2 > out/realsim_scale_test/realsim_imp_w1.0_party512_seed4.txt &

python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 0 -g 3 > out/realsim_scale_test/realsim_imp_w1.0_party2048_seed0.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 1 -g 3 > out/realsim_scale_test/realsim_imp_w1.0_party2048_seed1.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 2 -g 3 > out/realsim_scale_test/realsim_imp_w1.0_party2048_seed2.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 3 -g 3 > out/realsim_scale_test/realsim_imp_w1.0_party2048_seed3.txt &
python src/algorithm/SplitNN.py -d realsim -c 2 -m acc -p 2048 -sp imp -w 1.0 -s 4 -g 3 > out/realsim_scale_test/realsim_imp_w1.0_party2048_seed4.txt &