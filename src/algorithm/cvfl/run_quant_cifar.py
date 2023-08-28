import numpy as np
import os
import random

for i in range(1):
    seed = 42 #random.randint(0, 2**31)
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 0') 
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 2 --comp quantize')
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 4 --vecdim 1 --comp topk')
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize')
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 1 --comp topk')
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 2 --comp quantize')
    #os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 1 --comp quantize')
    os.system(f'python quant_cifar.py 10class/classes/ --num_clients 4 --seed {seed} --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 16 --vecdim 1 --comp topk')

# 532  pip install latbin
#   533  pip install numpy
#   534  pip install latbin
#   535  pip install pandas
#   536  pip install latbin
#   538  pip install scikit-learn
#   539  pip install latbin
#   540  pip install matplotlib
#   541  pip install latbin
# 546  pip install deprecated
# 548  pip install torchmetrics
# 550  pip install shap