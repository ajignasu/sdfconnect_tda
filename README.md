# sdfconnect_tda
SDFConnect: Neural Implicit Surface Reconstruction of Sparse Point Clouds with Topological Constraints



# Preliminaries
use the environment.yml file to create a conda environment with all the necessary dependencies. 
```bash
conda env create -f environment.yml
```
Note: This repository was tested on Windows 10 with CUDA 11.6.1 and Python 3.8.18.

# Running surface reconstruction
To run surface reconstruction, use the following command:
```bash
conda activate npull

python run.py --gpu 0 --conf confs/tda_npull.conf --mode train_tda --dataname lord_quas --dir lord_quas

```

