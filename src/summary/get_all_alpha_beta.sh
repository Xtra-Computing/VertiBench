mkdir -p log/alpha_beta

#nohup taskset -c 0-55 python src/summary/get_alpha_beta.py -s 0 -g 0 -lp log/alpha_beta/alpha_beta_seed0.log --decimal 2 > out/alpha_beta_seed0.out &
nohup taskset -c 0-55 python src/summary/get_alpha_beta.py -s 1 -g 6 -lp log/alpha_beta/alpha_beta_seed1.log --decimal 2 > out/alpha_beta_seed1.out &
nohup taskset -c 0-55 python src/summary/get_alpha_beta.py -s 2 -g 7 -lp log/alpha_beta/alpha_beta_seed2.log --decimal 2 > out/alpha_beta_seed2.out &
#nohup taskset -c 0-55 python src/summary/get_alpha_beta.py -s 3 -g 6 -lp log/alpha_beta/alpha_beta_seed3.log --decimal 2 > out/alpha_beta_seed3.out &
#nohup taskset -c 0-55 python src/summary/get_alpha_beta.py -s 4 -g 7 -lp log/alpha_beta/alpha_beta_seed4.log --decimal 2 > out/alpha_beta_seed4.out &