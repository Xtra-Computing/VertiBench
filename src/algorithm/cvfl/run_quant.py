import numpy as np
import os
import random

for i in range(5):
    seed = random.randint(0, 2**31)
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 0') 
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 4 --vecdim 2 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 4 --vecdim 1 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 4 --vecdim 1 --comp topk')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 8 --vecdim 1 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 8 --vecdim 1 --comp topk')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 16 --vecdim 2 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 16 --vecdim 1 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 16 --vecdim 1 --comp topk')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 1 --epochs 500 --lr 0.01 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 4 --seed {seed} --b 16 --local_epochs 25 --epochs 20 --lr 0.01 --quant_level 8 --vecdim 2 --comp quantize')
    os.system(f'python quant.py 10class/classes/ --num_clients 12 --seed {seed} --b 16 --local_epochs 10 --epochs 50 --lr 0.01 --quant_level 8 --vecdim 2 --comp quantize')

