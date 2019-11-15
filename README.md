# Towards Extremely Efﬁcient Neural Architecture Search via Dynamic Distribution Pruning

There is the code of the paper ``Towards Extremely Efﬁcient Neural Architecture Search via Dynamic Distribution Pruning`` for searching and network generation.


## Searching

```bash
git clone https://github.com/CVPR2020-1074/Towards-Extremely-Efficient-Neural-Architecture-Search-via-Dynamic-Distribution-Pruning.git
cd Towards-Extremely-Efficient-Neural-Architecture-Search-via-Dynamic-Distribution-Pruning
```
check the config file in ./utils/config.py

```
python train_search.py
```



## Network generation
set the search dir in line 118 of ./network_generator.py

```bash
python network_generator.py
```

The script will generate the neural genotypes with different height and model size.

## Train

For the network training, after get the model genotype, the network training can directly use the code in previous NAS work DARTS https://github.com/khanrc/pt.darts