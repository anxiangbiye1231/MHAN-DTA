# MHAN-DTA
---
MHAN-DTA: A Multiscale Hybrid Attention  Network for Drug-Target Affinity Prediction

## Datasets
All data used in this paper are publicly available and can be accessed here:  
- PDBbind v2020: http://www.pdbbind.org.cn/download.php  
- 2013 and 2016 core sets: http://www.pdbbind.org.cn/casf.php  
You can find processed data from `./data`, and change the path to run.

## Note 

This project contains several GNN-based models for protein-ligand binding affinity prediction, which are mainly taken from  
- PotentialNet: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/potentialne  
- GNN_DTI: https://github.com/jaechanglim/GNN_DTI  
- IGN: https://github.com/zjujdj/InteractionGraphNet/tree/master  
- SchNet: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html  
- GraphDTA: https://github.com/thinng/GraphDTA   
- MGraphGTA: https://github.com/guaguabujianle/MGraphDTA   
- MPM-DTA: https://github.com/Jthy-af/HaPPy   
- MMDTA: https://github.com/dldxzx/MMDTA
- Co-VAE：https://github.com/LiminLi-xjtu/CoVAE
- MT-DTA: https://github.com/Lamouryz/MT-DTA/tree/main/MT-DTA
- MDCT-DTA: https://github.com/zhengxin-plog/MD-CT-DTA  

## Requirements
biopython==1.79  
networkx==3.2.1  
numpy==1.23.5    
pandas==2.2.1    
pymol==3.0.0  
python==3.10.0   
rdkit==2023.9.5    
scikit-learn==1.4.1    
scipy==1.12.0    
torch==2.0.1     
torch-geometric==2.5.2   
tqdm==4.66.2  
## Resource requirements
We tested the training and inference time, along with the resource requirement of MHAN-DTA on a server computer embedded with a graphical card NVIDIA GeForce RTX4090D with a memory of 24G. MHAN-DTA takes 86 minutes to train. Once the training is finished, the model takes only 4.6 seconds for the internal test set, with an average of only 2.43 ms for each test sample. For the resource requirement, the training of MHAN-DTA costs 17520M of graphics memory.

## Usage
We provide a demo to show how to train and test MHAN-DTA.   
### 1. Training 
1. Firstly,  unzip the preprocessed data from `./data/train.tar.gz` and `./data/valid.tar.gz`.       
2. Secondly, run train.py using `python train.py`.  
### 2. Testing  
1. Unzip the preprocessed data from `./data/internal_test.tar.gz`, `./data/test_2013.tar.gz`, `./data/test_2016.tar.gz` and `./data/test_hiq.tar.gz`.   
2. Run test.py using `python test.py`.    
3. You may need to modify some file paths in the source code before running it.    
Or download the trained model from https://pan.baidu.com/s/15m-wyDQBaysJQ3hWphDOLg. Password：td7c   
### 3. Process raw data  
1. Run preprocessing.py using `python preprocessing.py`.       
2. Run process_data.py using `python process_data.py`.    
### 4. Baseline
The using details of training, testing, and evaluation metrics for the benchmark can be found in the `./baseline` directory, and the trained model can be downloaded from the link above 2.Testing.   



