<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# ProxyFL
Code accompanying the paper "ProxyFL: Decentralized Federated Learning through Proxy Model Sharing"

## Prerequisite
- Python 3.9
```bash
conda create -n ProxyFL python=3.9
conda activate ProxyFL
```
- PyTorch 1.9.0
```bash
conda install pytorch=1.9.0 torchvision=0.10.0 numpy=1.21.2 -c pytorch
```
- mpi4py 3.1.2
```bash
conda install -c conda-forge mpi4py=3.1.2
```
- opacus 0.14.0
```bash
pip install 'opacus==0.14.0'
```
- matplotlib 3.4.3
```bash
conda install -c conda-forge matplotlib=3.4.3
```

## Run experiment
Download data via
```bash
bash download_data.sh
```
Then run the script
```bash
bash run_exp.sh
```
