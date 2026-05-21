import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from staig.adata_processing import LoadBatch10xAdata

file_fold = "../RAW_SLICE/"
# args = argparse.Namespace(
#     dataset='DLPFC',
#     slide='integration_vertical07080910',
#     config='/data/ZhangYx/STAIG/train_img_config_new.yaml',
#     label=True,
#     filelist=['151507','151508','151509','151510']
# )
# args = argparse.Namespace(
#     dataset='DLPFC',
#     slide='integration_vertical076973',
#     config='/data/ZhangYx/STAIG/train_img_config_new.yaml',
#     label=True,
#     filelist=['151507','151669','151673']
# )

args = argparse.Namespace(
    dataset='DLPFC',
    #slide='test7576',
    slide='integration_vertical73747576',
    config='/data/ZhangYx/STAIG/train_img_config_new.yaml',
    label=True,
    filelist=['151673','151674','151675','151676']
 )
config = yaml.load(open(args.config), Loader=SafeLoader)[str(args.slide)]
slide_path = os.path.join(file_fold, args.dataset)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

loader = LoadBatch10xAdata(
        dataset_path=slide_path,
        file_list=args.filelist,
        n_neighbors=config['num_neigh'],
        n_top_genes=config['num_gene'],
        image_emb=True,
        label=args.label,
        use_gated_fusion=True,
        auto_pretrain=True,
        fusion_output_dim=384,
        pretrain_epochs=500,
        batch_mix_weight=20.0,
        batch_mix_method='simple',
        gene_harmony_theta=2,
        harmonize_image=True,
        image_harmony_theta=10,
        cache_model=False,
        random_seed=42,
    )
# loader = LoadBatch10xAdata(
#         dataset_path=slide_path,
#         file_list=args.filelist,
#         n_neighbors=config['num_neigh'],
#         n_top_genes=config['num_gene'],
#         image_emb=True,
#         label=args.label,
#         strategy='inter'
#     )

data = loader.run()
# training STAIG
staig = STAIG(args=args, config=config, single=False , refine=False)
staig.adata = data
staig.train()
staig.eva()
staig.cluster(args.label)

output_filename = f"results/staigraw_adata3.h5ad"
staig.adata.write(output_filename)

