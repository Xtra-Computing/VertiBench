# VertiBench

# Prerequisites

## Prepare the environment
```bash
# Create environment for VertiBench
conda create -n vertibench python=3.10
conda activate vertibench

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other libraries
pip3 install scikit-learn requests pytz tqdm pandas deprecated torchmetrics shap matplotlib tifffile opencv-python scipy
```

## Prepare the dataset
Please ensure that your folder structure of `data/` looks like this:

```bash
~/VertiBench (master*) » tree -L 3
.
|-- comm-cost.ipynb
|-- correlation.ipynb
|-- data/       # <=== the dataset folder. Make sure that the folder structure is the same.
|   |-- real/
|   |   |-- nus-wide/
|   |   |-- paper/
|   |   |-- satellite/
|   |   `-- vehicle/
|   `-- syn/
|       |-- covtype/
|       |-- epsilon/
|       |-- gisette/
|       |-- letter/
|       |-- msd/
|       |-- radar/
|       `-- realsim/
|-- feature-importance.ipynb
|-- fig/
|-- log/
|-- playground.ipynb
|-- playground.py
|-- plot-image.ipynb
|-- README.md
|-- real-data.ipynb
|-- requirements.txt
|-- run.sh
|-- scale_test.ipynb
|-- src/
|   |-- algorithm/
|   |-- dataset/
|   |-- main.py
|   |-- preprocess/
|   |-- script/
|   |-- summary/
|   `-- utils/
`-- unsupervised-learning.ipynb
```

You are all set! Please proceed to the "Examples" section for more details on how to run the test.

# Examples

- Running `SplitNN` algorithm on `epsilon` dataset, the dataset have `2` classes, using metric `acc`, `4` parties collaboration, using `correlation` splitter with beta `0.0`, running on GPU id `2`
```bash
git clone 
cd VertiBench
python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 2
```

# Citation
```txt
```
