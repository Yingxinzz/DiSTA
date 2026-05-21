import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from staig.staig import STAIG
import yaml
from yaml import SafeLoader
import torch

from staig.adata_processing import LoadSingle10xAdata
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import numpy as np


file_fold = "../RAW_SLICE/"

# args = argparse.Namespace(
#     dataset='Dataset/v10x',
#     slide="Human_Breast_Cancer",
#     config='/root/BE/STAIG/train_img_config_new.yaml',
#     label=True,
# )

args = argparse.Namespace(
    dataset='test10x',
    #slide="Human_Breast_Cancer",
    slide="Mouse_Brain_Anterior",
    #slide="Mouse_Brain_Posterior",
    #slide="WS_PLA_S9101764",
    #slide="WS_PLA_S9101765",
    #slide="WS_PLA_S9101767",
    config='/data/ZhangYx/STAIG/train_img_config_new.yaml',
    label=True,
)    
config = yaml.load(open(args.config), Loader=SafeLoader)[args.slide]
slide_path = os.path.join(file_fold, args.dataset, args.slide)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random.seed(12345)
torch.use_deterministic_algorithms(True)
data = LoadSingle10xAdata(path=slide_path,n_neighbors=config['num_neigh'],n_top_genes=config['num_gene'],image_emb=True, label = args.label).run()
staig = STAIG(args=args, config=config, single=False)
staig.adata = data
staig.train()
staig.eva()
staig.cluster(args.label)
output_filename = f"results/dista_Mouse_Brain_Anterior.h5ad"
staig.adata.write(output_filename)
