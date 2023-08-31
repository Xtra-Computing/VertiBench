from datasets import Radar
import numpy as np
from tqdm import tqdm
# ['CovType','MSD','Higgs','Gisette','Realsim','Epsilon','Letter','Radar']

log = []
for dataseed in tqdm(["0", "1", "2", "3", "4"]):
    for corr in ["0.0", "0.3", "0.6", "1.0"]:
        c = Radar("", "train", "corr", corr, dataseed)
        log.append(("corr", corr, c.partitions, np.sum(c.partitions)))
    
    for impt in ["0.1", "1.0", "10.0", "100.0"]:
        c = Radar("", "test", "imp", impt, dataseed)
        log.append(("imp", impt, c.partitions, np.sum(c.partitions)))

for l in log:
    print(l)