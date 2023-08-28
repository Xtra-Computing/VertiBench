This is VertiBench's modified version of the C-VFL MVCNN

File `run_scale_test_rebuttal*.py` encapsulates the project's execution commands














Repo is a private clone of MVCNN-PyTorch extended for
experiments with Compressed Local BC-SGD.

For specific details, read the original README below.
Our scripts additionally require that the ModelNet10
dataset is preprocessed and placed in a folder
named '10class/classes/'.

One can install our environment with Anaconda:
    conda env create -f flearn.yml 

Our results are saved as pickle files in the 'results/quant_fix' folder.
To plot the results in 'quant_fix':
    python plot_all.py
    python plot_comm.py
    python plot_12.py
    python plot_cifar.py
    python plot_comm_cifar.py
This will generate all .png plots, as well as the files
'results.txt' and 'resultsMB.txt', which contain the
results seen in Table 2 of the paper.

If you wish to rerun the experiments,
the script we used to run all experiments sequentially
can be run as follows:
    python run_quant.py
    python run_quant_cifar.py
This will choose 5 random seeds, run all experiments sequentially,
and places the results in the current working directory.
In our experiments, the seeds that were chosen were:
[707412115,1928644128,16910772,1263880818,1445547577]
We used a different script (not included) to run all experiments
in parallel on our internal cluster. It is not included here
to avoid breaching the blind review process.

Original README:

# MVCNN-PyTorch
## Multi-View CNN built on ResNet/AlexNet to classify 3D objects
A PyTorch implementation of MVCNN using ResNet, inspired by the paper by [Hang Su](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf).
MVCNN uses multiple 2D images of 3D objects to classify them. You can use the provided dataset or create your own.

Also check out my [RotationNet](https://github.com/RBirkeland/RotationNet) implementation whitch outperforms MVCNN (Under construction).

![MVCNN](https://preview.ibb.co/eKcJHy/687474703a2f2f7669732d7777772e63732e756d6173732e6564752f6d76636e6e2f696d616765732f6d76636e6e2e706e67.png)

### Dependencies
* torch
* torchvision
* numpy
* tensorflow (for logging)

### Dataset
ModelNet40 12-view PNG dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view).

You can also create your own 2D dataset from 3D objects (.obj, .stl, and .off), using [BlenderPhong](https://github.com/WeiTang114/BlenderPhong)

### Setup
```bash
mkdir checkpoint
mkdir logs
```

### Train
To start training, simply point to the path of the downloaded dataset. All the other settings are optional.

```
python controller.py <path to dataset>  [--depth N] [--model MODEL] [--epochs N] [-b N]
                                        [--lr LR] [--momentum M] [--lr-decay-freq W]
                                        [--lr-decay W] [--print-freq N] [-r PATH] [--pretrained]
```

To resume from a checkpoint, use the -r tag together with the path to the checkpoint file.

### Tensorboard
To view training logs
```
tensorboard --logdir='logs' --port=6006
```
