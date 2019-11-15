# Towards Extremely Efﬁcient Neural Architecture Search via Dynamic Distribution Pruning

There is the code of the paper ``Towards Extremely Efﬁcient Neural Architecture Search via Dynamic Distribution Pruning`` for network searching and generation.


## Searching

```bash
git clone https://github.com/CVPR2020-1074/Towards-Extremely-Efficient-Neural-Architecture-Search-via-Dynamic-Distribution-Pruning.git
cd Towards-Extremely-Efficient-Neural-Architecture-Search-via-Dynamic-Distribution-Pruning
```
Change the config file in ./utils/config.py, then run

```
python train_search.py
```



## Network generation

Set the search dir in ./network_generator.py, then run

```bash
python network_generator.py
```

The script will generate the neural genotypes with different height and model size.

## Train

The network training can be done by directly using the code in previous NAS work DARTS https://github.com/khanrc/pt.darts
