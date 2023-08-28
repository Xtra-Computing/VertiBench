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

batch_size = 64
embedding_size = 4096
server_model_size = 163840
batches_per_epoch = 16
min_bits_per_epoch = batches_per_epoch*(2*embedding_size*batch_size*4**2 + server_model_size*4)*2

results = open('results.txt','w')
results.write(f'Accuracy reached after {100*min_bits_per_epoch} bits sent\n')
results.write('Max Std Best Std\n')

resultsMB = open('resultsMB.txt','w')
resultsMB.write('Avg Std\n')

def all_seeds(prefix, result_out, num_clients, bits, local_epochs=10):
    files = glob.glob(f'results/quant_fix/{prefix}_seed*.pkl')
    pickles = []    
    print(f'results/quant_fix/{prefix}_seed')
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        pkl = np.array([x.cpu().numpy() for x in pkl])
        pickles.append(pkl)
    pickles = np.array(pickles)
    if local_epochs == 25:
        pickles = np.array([p[:20] for p in pickles])
       
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    bits_per_epoch = batches_per_epoch*bits*(2*embedding_size*batch_size*num_clients**2 + server_model_size*num_clients)
    # Make losses by bits sents
    final_epoch = 0
    bits_so_far = 0
    while bits_so_far < 100*min_bits_per_epoch:
        bits_so_far += bits_per_epoch
        final_epoch += 1
    avg_tmp = avg[0:final_epoch+1]
    avg_interp = interp.interp1d(np.arange(avg_tmp.size),avg_tmp)
    avg_bits = avg_interp(np.linspace(0,avg_tmp.size-1, 101))
    std_tmp = std[0:final_epoch+1]
    std_interp = interp.interp1d(np.arange(std_tmp.size),std_tmp)
    std_bits = std_interp(np.linspace(0,std_tmp.size-1, 101))

    if result_out is not None:
        final_epoch = 0
        bits_so_far = 0
        while bits_so_far < 10*8*2**20: #10 MB
            bits_so_far += bits_per_epoch
            final_epoch += 1
        ind = np.argmax(avg[0:final_epoch+1])
        max_ind = np.argmax(avg)
        if local_epochs > 10:
            max_ind = np.argmax(avg[:int(100*10/local_epochs)])
        max_error = std[max_ind] 
        best_error = std[ind] 
        result_out.write(f'&{avg[max_ind]:.2f}\% $\pm$ {max_error:.2f}\% & {avg[ind]:.2f}\% $\pm$ {best_error:.2f}\%\n')

        bits_to_reach = []
        for i in range(len(pickles)):
            final_epoch = 0
            bits_so_far = 0
            while pickles[i][final_epoch] < 75:
                bits_so_far += bits_per_epoch
                final_epoch += 1
            bits_to_reach.append(bits_so_far/(8*2**20))
        bits_to_reach = np.array(bits_to_reach)
        bits_avg = np.average(bits_to_reach)
        bits_std = np.std(bits_to_reach)

        resultsMB.write(f'& {bits_avg:.2f} $\pm$ {bits_std:.2f}\n')

    return (avg, std, np.array(avg_bits), np.array(std_bits))

types = ['loss', 'accs_train', 'accs_test']
for t in types:
    Qs = [4,8,16]
    f = None
    if t == 'accs_test':
        f = results
        losses0 = all_seeds(f'{t}_mvcnn_NC4_LE10_quant0_dim1_comp',f,4,32)
    
    for q in Qs:
        bits = math.log(q,2)
        # Parse results
        losses0 = all_seeds(f'{t}_mvcnn_NC4_LE10_quant0_dim1_comp',None,4,32)
        losses1 = all_seeds(f'{t}_mvcnn_NC4_LE10_quant{q}_dim1_compquantize',f,4,bits)
        losses2 = all_seeds(f'{t}_mvcnn_NC4_LE10_quant{q}_dim2_compquantize',f,4,bits)
        losses3 = all_seeds(f'{t}_mvcnn_NC4_LE10_quant{q}_dim1_comptopk',f,4,bits)
    
        fig, ax = plt.subplots()
        # Plot loss
        plt.plot(losses0[0], label='No Compression')
        plt.plot(losses1[0], label='Scalar Quantize')
        plt.plot(losses2[0], label='Vector Quantize')
        plt.plot(losses3[0], label='Top-k')
      
        plt.fill_between(np.linspace(0,100,101), losses0[0] - losses0[1], losses0[0] + losses0[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses1[0] - losses1[1], losses1[0] + losses1[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses2[0] - losses2[1], losses2[0] + losses2[1], alpha=0.3)
        plt.fill_between(np.linspace(0,100,101), losses3[0] - losses3[1], losses3[0] + losses3[1], alpha=0.3)
    
        plt.xlabel('Epochs')
        if t == 'loss':
            plt.ylim(0, 1)
            plt.ylabel('Loss')
        else:
            if t == 'accs_train':
                plt.ylim(0, 100)
            if t == 'accs_test':
                plt.ylim(0, 90)
            plt.ylabel('Accuracy')
    
        plt.legend()
        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'{t}_quant_mvcnn{q}.png')
        #plt.show()
    
        fig, ax = plt.subplots()
        # Plot loss by bits sent 
        min_inds = min(losses0[2].shape[0], losses1[2].shape[0], 
                            losses2[2].shape[0])#,losses3[2].shape[0])
        x_axis_bits = np.linspace(0, 100*min_bits_per_epoch/(8*2**20), min_inds)

        plt.plot(x_axis_bits, losses0[2][:min_inds], label='No Compression')
        plt.plot(x_axis_bits, losses1[2][:min_inds], label='Scalar Quantize')
        plt.plot(x_axis_bits, losses2[2][:min_inds], label='Vector Quantize')
        plt.plot(x_axis_bits, losses3[2][:min_inds], label='Top-k')
      
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
            if t == 'accs_test':
                plt.ylim(0, 90)
            plt.ylabel('Accuracy')
    
        plt.legend()
        #ratio = 0.5
        #xleft, xright = ax.get_xlim()
        #ybottom, ytop = ax.get_ylim()
        #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        plt.tight_layout()
        plt.savefig(f'{t}_quantcomm_{q}_mvcnn.png')
        #plt.show()
    
    if t == 'accs_test':
        #losses_LE1 = all_seeds(f'{t}_mvcnn_NC4_LE1_quant8_dim2_compquantize',f,4,3,1)
        #losses_LE25 = all_seeds(f'{t}_mvcnn_NC4_LE25_quant8_dim2_compquantize',f,4,3,25)
        losses_NC12 = all_seeds(f'{t}_mvcnn_NC12_LE10_quant8_dim2_compquantize',f,12,3)
