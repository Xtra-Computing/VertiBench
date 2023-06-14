# VertiBench

# Prerequisites
```bash
conda create -n vertibench python=3.10
conda activate vertibench
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install scikit-learn requests pytz tqdm pandas deprecated torchmetrics shap matplotlib tifffile opencv-python
```

```bash
git clone 
cd VertiBench
python src/algorithm/SplitNN.py -d epsilon -c 2 -m acc -p 4 -sp corr -b 0.0 -s 0 -g 2

```
