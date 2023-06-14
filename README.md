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
pip3 install scikit-learn requests pytz tqdm pandas deprecated torchmetrics shap matplotlib tifffile opencv-python scipy pymoo xgboost difflib
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

Please make sure that your current working directory is set to "VertiBench" with the following commands:

```bash
git clone <repository_url> 
cd VertiBench
```

Please replace <repository_url> with the actual URL of this Git repository.

## Vertical Split the dataset
This Python script is designed to split a dataset into vertical partitions. The script leverages different splitting methods and can operate on various hardware including GPUs for efficient data partitioning. 

**Usage:**
```bash
python src/preprocess/vertical_split.py <dataset_path> <num_parties> [-sp <splitter>] [-w <weights>] [-b <beta>] [-s <seed>] [-t <test_ratio>] [-g <gpu_id>] [-j <jobs>] [-v]
```

Where:
- `dataset_path`: The path to the dataset file to be split.
- `num_parties`: The number of parties for data distribution.
- `splitter`: Optional, the method used to split the dataset, either 'imp' (ImportanceSplitter) or 'corr' (CorrelationSplitter). Default is `imp`.
- `weights`: Optional, the weights for the ImportanceSplitter. Default is 1.
- `beta`: Optional, the beta value for the CorrelationSplitter. Default is 1.
- `seed`: Optional, the random seed used for data splitting.
- `test_ratio`: Optional, the ratio of data to be allocated for testing. If not specified, no test split is performed.
- `gpu_id`: Optional, the ID of the GPU used for the CorrelationSplitter and CorrelationEvaluator. If not specified, the operation runs on the CPU.
- `jobs`: Optional, the number of jobs for the CorrelationSplitter. Default is 1.
- `v`: Optional, prints verbose information during execution.

The script will first load the dataset, then apply the splitting method as per the provided arguments. The subsets of the data are then shuffled and split into train-test datasets based on the provided test ratio. The script will finally store the datasets into separate files. The storage paths are generated using the 'PartyPath' utility which incorporates the input parameters to ensure unique file paths for different settings.

**Example:**

```bash
# vertical split "gisette" dataset to 4 parties, using importance splitter with weight 1.0, 20% data will be allocated for testing, random seed for splitting is 3, running on gpu 0
python src/preprocess/vertical_split.py data/syn/gisette/gisette.libsvm 4 -sp imp -w 1 -t 0.2 -s 3 -g 0
```


## Evaluate SplitNN on splited dataset

- Running `SplitNN` algorithm on `gisette` dataset, the dataset have `2` classes, using evaluation metric `acc`, there are `4` parties collaboration, using `importance` splitter with weight `1.0`, using seed `3`, running on GPU id `0`

```bash
python src/algorithm/SplitNN.py -d gisette -c 2 -m acc -p 4 -sp imp -w 1.0 -s 3 -g 0
```

# Citation
```txt
```
