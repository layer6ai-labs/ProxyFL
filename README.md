<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# ProxyFL
Code accompanying the paper "Decentralized Federated Learning through Proxy Model Sharing" published in [Nature Communications](https://www.nature.com/articles/s41467-023-38569-4).

Authors: [Shivam Kalra*](https://scholar.google.ca/citations?user=iEwZn18AAAAJ&hl=en), [Junfeng Wen*](https://junfengwen.github.io/), [Jesse C. Cresswell*](https://scholar.google.ca/citations?user=7CwOlvoAAAAJ&hl=en), [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs), [Hamid R. Tizhoosh&dagger;](https://scholar.google.ca/citations?user=Mzinpo0AAAAJ&hl=en)  
* &ast; Denotes equal contribution
* &dagger; University of Waterloo / Vector Institute

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

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{kalra2021,
      title={ProxyFL: Decentralized Federated Learning through Proxy Model Sharing},
      author={Shivam Kalra, Junfeng Wen, Jesse C. Cresswell, Maksims Volkovs, Hamid R. Tizhoosh},
      year={2021}
    }
