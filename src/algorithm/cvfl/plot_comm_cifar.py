"""
Plot adaptive experimental results
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import os
import glob
import math
import scipy.interpolate as interp

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

batch_size = 100 
embedding_size = 16 
server_model_size = 16*4*10 
batches_per_epoch = 500 
min_bits_per_epoch = batches_per_epoch*(2*embedding_size*batch_size*4**2 + server_model_size*4)*2

def all_seeds(prefix, result_out, num_clients, bits, local_epochs=10):
    files = glob.glob(f'results/quant_fix/{prefix}_seed*.pkl')
    pickles = []    
    print(f'{prefix}_seed')
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        pkl = np.array([x.cpu().numpy() for x in pkl])
        pickles.append(pkl)
    pickles = np.array(pickles)
       
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    bits_per_epoch = batches_per_epoch*bits*(2*embedding_size*batch_size*num_clients**2 + server_model_size*num_clients)
    # Make losses by bits sents
    final_epoch = 0
    bits_so_far = 0
    while bits_so_far < 200*min_bits_per_epoch:
        bits_so_far += bits_per_epoch
        final_epoch += 1
    avg_tmp = avg[0:final_epoch+1]
    avg_interp = interp.interp1d(np.arange(avg_tmp.size),avg_tmp)
    avg_bits = avg_interp(np.linspace(0,avg_tmp.size-1, 201))
    std_tmp = std[0:final_epoch+1]
    std_interp = interp.interp1d(np.arange(std_tmp.size),std_tmp)
    std_bits = std_interp(np.linspace(0,std_tmp.size-1, 201))

    return (avg, std, np.array(avg_bits), np.array(std_bits))

types = ['loss', 'accs_train', 'accs_test']
for t in types:
    comps = ['quantize','quantize','topk']
    dims = [1,2,1]
    f = None
    
    for comp,dim in zip(comps,dims):
        # Parse results
        losses0 = all_seeds(f'{t}_cifar_NC4_LE10_quant0_dim1_comp',None,4,32)
        losses1 = all_seeds(f'{t}_cifar_NC4_LE10_quant4_dim{dim}_comp{comp}',f,4,2)
        losses2 = all_seeds(f'{t}_cifar_NC4_LE10_quant8_dim{dim}_comp{comp}',f,4,3)
        losses3 = all_seeds(f'{t}_cifar_NC4_LE10_quant16_dim{dim}_comp{comp}',f,4,4)
    
        fig, ax = plt.subplots()
        # Plot loss by bits sent 
        min_inds = min(losses0[2].shape[0], losses1[2].shape[0], 
                            losses2[2].shape[0],losses3[2].shape[0])
        x_axis_bits = np.linspace(0, 200*min_bits_per_epoch/(8*2**20), min_inds)

        plt.plot(x_axis_bits, losses0[2][:min_inds], label='b=32')
        plt.plot(x_axis_bits, losses1[2][:min_inds], label='b=2')
        plt.plot(x_axis_bits, losses2[2][:min_inds], label='b=3')
        plt.plot(x_axis_bits, losses3[2][:min_inds], label='b=4')
      
        plt.fill_between(x_axis_bits, losses0[2][:min_inds] - losses0[3][:min_inds], losses0[2][:min_inds] + losses0[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses1[2][:min_inds] - losses1[3][:min_inds], losses1[2][:min_inds] + losses1[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses2[2][:min_inds] - losses2[3][:min_inds], losses2[2][:min_inds] + losses2[3][:min_inds], alpha=0.3)
        plt.fill_between(x_axis_bits, losses3[2][:min_inds] - losses3[3][:min_inds], losses3[2][:min_inds] + losses3[3][:min_inds], alpha=0.3)
    
        plt.xlabel('Communication Cost (MB)')
        if t == 'losses':
            plt.ylim(0.2, 0.6)
            plt.ylabel('Loss')
        else:
            #if t == 'accs_train':
            #    plt.ylim(50, 100)
            #if t == 'accs_test':
            #    plt.ylim(40, 85)
            plt.ylabel('Accuracy')
    
        plt.legend(prop={'size': 15})
        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'{t}_quantcomm_{comp}{dim}_cifar.png')
        #plt.show()
