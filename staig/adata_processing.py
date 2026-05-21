import scanpy as sc
import scanpy.external as sce 
import ot
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from scipy.spatial.distance import cdist,euclidean,cosine
from scipy.special import softmax
from anndata import AnnData
from scipy.linalg import block_diag
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from tqdm import tqdm
import scGeneClust as gc
import PyWGCNA
import NaiveDE
import SpatialDE
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import harmonypy as hm

def generate_pseudo_labels(img_emb, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    pseudo_labels = kmeans.labels_
    return pseudo_labels



class LoadSingle10xAdata:
    def __init__(self, path: str, n_top_genes: int = 3000, n_neighbors: int = 3, 
                 image_emb: bool = False, label: bool = True, filter_na: bool = True,
                 select='default', T: float = None):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.kernel = 'euclidean'
        self.select = 'default'
        self.T = T

    def load_data(self):
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()


    def preprocess(self):
        if self.select == 'default':
            sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'mvp':
            sc.pp.highly_variable_genes(self.adata, flavor="seurat")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'geneclust':
            self.adata.X = self.adata.X.toarray()
            info, selected_genes_ps = gc.scGeneClust(self.adata, n_var_clusters=200, version='fast', return_info=True)
            top_variable_genes = selected_genes_ps.tolist()
            self.adata.var['highly_variable'] = self.adata.var_names.isin(top_variable_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'wgcna':
            pyWGCNA_data = PyWGCNA.WGCNA(name='data', 
                              species='human', 
                              anndata=self.adata, 
                              outputPath='',
                              save=True)
            pyWGCNA_data.preprocess()
            pyWGCNA_data.findModules()
            module_colors = np.unique(pyWGCNA_data.datExpr.var['moduleColors']).tolist()
            all_hub_genes = []
            for module_color in module_colors:
                df_hub_genes = pyWGCNA_data.top_n_hub_genes(moduleName=module_color, n=100)
                gene_names = df_hub_genes.index.tolist()  
                all_hub_genes.extend(gene_names)  
            self.adata.var['highly_variable'] = self.adata.var_names.isin(all_hub_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)

        if self.select == 'spatialde':
            x_coords = self.adata.obsm['spatial'][:, 0]
            y_coords = self.adata.obsm['spatial'][:, 1]
            counts = pd.DataFrame(
                self.adata.X.toarray(),
                index=self.adata.obs_names,
                columns=self.adata.var_names
            )
            counts = counts.T[counts.sum(0) >= 3].T
            self.adata.obs['total_counts'] = np.ravel(self.adata.X.sum(axis=1))
            sample_info = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                'total_counts': self.adata.obs['total_counts']
            }, index=self.adata.obs_names)
            norm_expr = NaiveDE.stabilize(counts.T).T
            resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
            sample_resid_expr = resid_expr.sample(n=1000, axis=1, random_state=1)
            X = sample_info[['x', 'y']].values 
            results = SpatialDE.run(X, resid_expr)
            top_genes_list = results.sort_values('qval')['g'].head(1000).tolist()
            self.adata.var['highly_variable'] = self.adata.var_names.isin(top_genes_list)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
    def construct_interaction(self):
        position = self.adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)

        self.adata.obsm['graph_neigh'] = adj

    def generate_gene_expr(self):
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['feat'] = feat

    def load_label(self):
        #slide_name = os.path.basename(os.path.normpath(self.path))
        #truth_filename = f"{slide_name}_truth.txt"
        
        truth_filename = f"truth.txt"
        label_path = os.path.join(self.path, truth_filename)
        if not os.path.exists(label_path):
           print(f"[Warning] Ground truth file not found: {label_path}. Skipping label loading.")
           self.adata.obs['ground_truth'] = None
           return
               
        df_meta = pd.read_csv(label_path, sep='\t', header=None)
        df_meta_layer = df_meta[1]

        self.adata.obs['ground_truth'] = df_meta_layer.values
        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]


    def load_image_emb(self):        
        sample_name = os.path.basename(os.path.normpath(self.path))
        feat_path = os.path.join("features/spatial_patches_dinov2_spatial_full_dino", f"{sample_name}.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Image feature file not found: {feat_path}")
        
        feat_dict = np.load(feat_path, allow_pickle=True).item()
        features_all = feat_dict["features"]
        if features_all.ndim == 3:
           features_all = features_all.squeeze(axis=1)
        
        barcode_to_feat = dict(zip(feat_dict["barcodes"], features_all))
        features = []
        unmatched = 0
        for spot_id in self.adata.obs_names:
            if spot_id in barcode_to_feat:
                features.append(barcode_to_feat[spot_id])
            else:
                unmatched += 1
                features.append(np.zeros_like(features_all[0]))
        
        features = np.stack(features, axis=0)
        if unmatched > 0:
            print(f"There are {unmatched} unmatched spots; zeros filled.")

        scaler_img = StandardScaler()
        features = features.reshape(features.shape[0], -1)
        data_scaled = scaler_img.fit_transform(features)

        pca_img = PCA(n_components=16, random_state=42)
        img_emb = pca_img.fit_transform(data_scaled)
        self.adata.obsm['img_emb'] = img_emb
        print(f"[INFO] Image PCA done: {img_emb.shape}")

        pca_gene = PCA(n_components=64, random_state=42)
        gene_emb = pca_gene.fit_transform(self.adata.obsm['feat'])
        self.adata.obsm['feat_pca'] = gene_emb 
        print(f"[INFO] Gene PCA done: {gene_emb.shape}")
        

    # def load_image_emb(self):
    #     data = np.load(os.path.join(self.path, 'embeddings.npy'))
    #     data = data.reshape(data.shape[0], -1)
    #     print(f"[INFO] Image features aligned: {data.shape}, adata: {self.adata.shape}")
    #     scaler = StandardScaler()
    #     embedding = scaler.fit_transform(data)
    #     scaler = StandardScaler()
    #     pca = PCA(n_components=16, random_state=42)
    #     embedding = pca.fit_transform(embedding)
    #     self.adata.obsm['img_emb'] = embedding
    #     pca_g = PCA(n_components=64, random_state=42)
    #     self.adata.obsm['feat_pca'] = pca_g.fit_transform(self.adata.obsm['feat'])

    #     self.adata.obsm['con_feat'] = np.concatenate([self.adata.obsm['feat_pca'], self.adata.obsm['img_emb']], axis=1)
    #     con_feat = self.adata.obsm['con_feat']
    #     scaler = StandardScaler()
    #     con_feat_standardized = scaler.fit_transform(con_feat)
    #     self.adata.obsm['con_feat'] = con_feat_standardized
    
    def calculate_edge_weights(self):

        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        
        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities


        if self.kernel=='rbf':

            gamma = 0.01 
            similarity_matrix = rbf_kernel(node_emb, gamma=gamma)
            

            edge_weights = np.where(graph_neigh == 1, 1 - similarity_matrix, 0)
            

            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)  
                edge_probabilities[i][non_zero_indices] = softmax_weights


            self.adata.obsm['edge_probabilities'] = edge_probabilities

        if self.kernel=='cosine':

            euclidean_distances = cdist(node_emb, node_emb, metric='cosine')


            edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)


            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)  
                edge_probabilities[i][non_zero_indices] = softmax_weights

            self.adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        if self.label:
            self.load_label()
        self.preprocess()
        self.construct_interaction()
        self.generate_gene_expr()

        if self.image_emb:
            self.load_image_emb()
            self.calculate_edge_weights()
        else:
            self.calculate_edge_weights_gene()

        print('adata load done')

        return self.adata

class BatchMixingGatedFusion(nn.Module):
    def __init__(self, gene_dim, img_dim, hidden_dim=256, output_dim=256):
        super().__init__()

        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, gene_feat, img_feat):
        g = self.gene_proj(gene_feat)
        i = self.img_proj(img_feat)

        alpha = self.gate(torch.cat([g, i], dim=1))
        fused = alpha * g + (1 - alpha) * i

        return self.out_proj(fused), alpha

class BatchMixingTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def simplified_batch_loss(self, features, batch_labels):
        n_samples = min(1024, len(features))
        if n_samples < len(features):
            idx = torch.randperm(len(features))[:n_samples]
            features_sample = features[idx]
            batch_sample = batch_labels[idx]
        else:
            features_sample = features
            batch_sample = batch_labels
        
        dist_matrix = torch.cdist(features_sample, features_sample, p=2)
        
        batch_matrix = batch_sample.unsqueeze(1) != batch_sample.unsqueeze(0)
        
        cross_batch_distances = dist_matrix[batch_matrix]
        
        if len(cross_batch_distances) > 0:
            loss = cross_batch_distances.mean()
        else:
            loss = torch.tensor(0.0, device=features.device)
        
        return loss
    
    def batch_separation_penalty(self, features, batch_labels):
        unique_batches = torch.unique(batch_labels)
        
        if len(unique_batches) < 2:
            return torch.tensor(0.0, device=features.device)
        
        batch_means = []
        for batch_id in unique_batches:
            mask = (batch_labels == batch_id)
            batch_feat = features[mask]
            batch_means.append(batch_feat.mean(dim=0))
        
        batch_means = torch.stack(batch_means)  # (n_batches, feature_dim)
        
        global_mean = batch_means.mean(dim=0)
        mean_distances = ((batch_means - global_mean) ** 2).sum(dim=1)
        loss = mean_distances.mean()
        
        return loss
    
    def mmd_batch_loss(self, features, batch_labels):
        unique_batches = torch.unique(batch_labels)
        
        if len(unique_batches) < 2:
            return torch.tensor(0.0, device=features.device)
        
        batch_features = []
        for batch_id in unique_batches:
            mask = (batch_labels == batch_id)
            batch_features.append(features[mask])
        
        def rbf_kernel(X, Y, sigma=1.0):
            if len(X) > 256:
                X = X[torch.randperm(len(X))[:256]]
            if len(Y) > 256:
                Y = Y[torch.randperm(len(Y))[:256]]
            
            XX = torch.sum(X**2, dim=1, keepdim=True)
            YY = torch.sum(Y**2, dim=1, keepdim=True)
            XY = torch.matmul(X, Y.T)
            
            dist = XX + YY.T - 2 * XY
            return torch.exp(-dist / (2 * sigma**2))
        
        mmd_total = 0.0
        count = 0
        
        for i in range(len(batch_features)):
            for j in range(i+1, len(batch_features)):
                X = batch_features[i]
                Y = batch_features[j]
                
                # MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2E[k(X,Y)]
                K_XX = rbf_kernel(X, X).mean()
                K_YY = rbf_kernel(Y, Y).mean()
                K_XY = rbf_kernel(X, Y).mean()
                
                mmd = K_XX + K_YY - 2 * K_XY
                mmd_total += mmd
                count += 1
        
        return mmd_total / count if count > 0 else torch.tensor(0.0, device=features.device)
    
    def train(self, gene_data, img_data, batch_labels, 
              epochs=100, batch_size=512,
              batch_mix_weight=0.0,
              batch_mix_method='simple',
              verbose=True):
        gene_tensor = torch.FloatTensor(gene_data).to(self.device)
        img_tensor = torch.FloatTensor(img_data).to(self.device)
        batch_tensor = torch.LongTensor(batch_labels).to(self.device)
        
        if verbose:
            if batch_mix_weight > 0:
                print(f"\n Batch Mixing Training:")
                print(f"  Method: {batch_mix_method}")
                print(f"  Weight: {batch_mix_weight}")
            else:
                print(f"\n Pure Reconstruction Training (No Batch Mixing)")
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_batch = 0
            
            indices = torch.randperm(len(gene_tensor))
            n_batches_iter = 0
            
            for i in range(0, len(gene_tensor), batch_size):
                batch_idx = indices[i:i+batch_size]
                
                gene_batch = gene_tensor[batch_idx]
                img_batch = img_tensor[batch_idx]
                batch_label_batch = batch_tensor[batch_idx]
                
                self.optimizer.zero_grad()
                
                fused, alpha = self.model(gene_batch, img_batch)
                
                recon_loss = (
                    F.mse_loss(fused[:, :gene_batch.shape[1]], gene_batch) + 
                    F.mse_loss(fused[:, :img_batch.shape[1]], img_batch)
                ) / 2
                
                # 🔥 损失2：批次混合损失（仅当weight>0时计算）
                if batch_mix_weight > 0:
                    if batch_mix_method == 'simple':
                        batch_loss = self.simplified_batch_loss(fused, batch_label_batch)
                    elif batch_mix_method == 'separation':
                        batch_loss = self.batch_separation_penalty(fused, batch_label_batch)
                    elif batch_mix_method == 'mmd':
                        batch_loss = self.mmd_batch_loss(fused, batch_label_batch)
                    else:
                        raise ValueError(f"Unknown batch_mix_method: {batch_mix_method}")
                    
                    loss = recon_loss + batch_mix_weight * batch_loss
                    epoch_batch += batch_loss.item()
                else:
                    loss = recon_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                n_batches_iter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / n_batches_iter
                avg_recon = epoch_recon / n_batches_iter
                
                if batch_mix_weight > 0:
                    avg_batch = epoch_batch / n_batches_iter
                    print(f"  Epoch {epoch+1}/{epochs}")
                    print(f"    Total Loss: {avg_loss:.4f}")
                    print(f"    Recon Loss: {avg_recon:.4f}")
                    print(f"    Batch Mix Loss: {avg_batch:.4f}")
                else:
                    print(f"  Epoch {epoch+1}/{epochs} - Recon Loss: {avg_recon:.4f}")
        
        if verbose:
            print("  ✓ Training completed")


def apply_gated_fusion(gene_feat, img_feat, model_path=None,
                       gene_dim=256, img_dim=256, output_dim=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BatchMixingGatedFusion(
        gene_dim=gene_dim,
        img_dim=img_dim,
        output_dim=output_dim
    ).to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    with torch.no_grad():
        gene_tensor = torch.FloatTensor(gene_feat).to(device)
        img_tensor = torch.FloatTensor(img_feat).to(device)
        
        fused_feat, alpha = model(gene_tensor, img_tensor)
        
        fused_np = fused_feat.cpu().numpy()
        alpha_np = alpha.cpu().numpy().squeeze()
    
    return fused_np, alpha_np


class LoadBatch10xAdata:
    def __init__(self, dataset_path: str, file_list: list, 
                 strategy: str = 'intra', 
                 
                 n_top_genes: int = 3000, 
                 n_neighbors: int = 5,
                 image_emb: bool = True, 
                 label: bool = True, 
                 filter_na: bool = True, 
                 do_log: bool = True,
                 
                 use_gated_fusion: bool = True,
                 fusion_output_dim: int = 384,
                 auto_pretrain: bool = True,
                 pretrain_epochs: int = 500,
                 cache_model: bool = False,
                 random_seed: int = 42):
        
        self.dataset_path = dataset_path
        self.file_list = file_list
        self.strategy = strategy
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.do_log = do_log
        self.use_gated_fusion = use_gated_fusion
        self.fusion_output_dim = fusion_output_dim
        self.auto_pretrain = auto_pretrain
        self.pretrain_epochs = pretrain_epochs
        self.cache_model = cache_model
        self.random_seed = random_seed
        self.batch_mix_method = 'simple'
        self.force_nn_graph = False

        if self.strategy == 'intra':
            print(f"\n[Mode] Intra-Donor Strategy ")
            self.batch_mix_weight = 0.01
            self.harmonize_image = True
            self.image_harmony_theta = 0.5
            self.harmonize_gene = False

        elif self.strategy == 'inter':
            print(f"\n[Mode] Inter-Donor Strategy (Strong Integration)")
            self.batch_mix_weight = 20.0
            self.harmonize_image = True
            self.image_harmony_theta = 10.0
            self.harmonize_gene = True
            self.gene_harmony_theta = 2.0
            self.force_nn_graph = True
            
        else:
            raise ValueError("strategy must be 'intra' or 'inter'")

        self.adata_list = []
        self.merged_adata = None
        self.model_cache_path = self._get_model_cache_path()
        self._fusion_model_trained = False

    def _get_model_cache_path(self):
        dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
        file_hash = hash(tuple(sorted(self.file_list))) % 10000
        return os.path.join('fusion_model_cache', 
                            f"fusion_{dataset_name}_{file_hash}_{self.strategy}_d{self.fusion_output_dim}.pth")

    def _should_pretrain(self):
        if not self.use_gated_fusion or not self.auto_pretrain or not self.image_emb: return False
        if self.cache_model and os.path.exists(self.model_cache_path):
            print(f"✓ Found cached model: {self.model_cache_path}")
            return False
        return True
    
    def construct_interaction(self, input_adata):
        position = input_adata.obsm['spatial']
        n_spot = input_adata.shape[0]
        
        if not getattr(self, "force_nn_graph", False):
            distance_matrix = ot.dist(position, position, metric='euclidean')
            interaction = np.zeros([n_spot, n_spot])
            for i in range(n_spot):
                vec = distance_matrix[i, :]
                distance = vec.argsort()
                for t in range(1, self.n_neighbors + 1):
                    y = distance[t]
                    interaction[i, y] = 1
        else:
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(position)
            _, indices = nbrs.kneighbors(position)
            interaction = np.zeros([n_spot, n_spot])
            for i in range(n_spot):
                interaction[i, indices[i, 1:]] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)
        input_adata.obsm['local_graph'] = adj
        return input_adata

    def _harmonize_features(self, key, theta, name):
        print(f"\n   -> Harmonizing {name} Features (Theta={theta})...")
        all_emb = np.vstack([a.obsm[key] for a in self.adata_list])
        batch_labels = []
        for i, a in enumerate(self.adata_list): batch_labels.extend([str(i)] * len(a))
        meta = pd.DataFrame({'batch': batch_labels})
        
        try:
            import harmonypy as hm
            harmony_out = hm.run_harmony(all_emb, meta, 'batch', max_iter_harmony=20, theta=theta, sigma=0.1, verbose=False)
            emb_corrected = harmony_out.Z_corr
            start = 0
            for adata in self.adata_list:
                end = start + len(adata)
                adata.obsm[f'{key}_raw'] = adata.obsm[key].copy()
                adata.obsm[key] = emb_corrected[start:end]
                start = end
            print(f" {name} features harmonized.")
        except Exception as e:
            print(f" [Warning] Harmony failed: {e}")

    def load_data(self):
        print(f"\n{'='*40}\nLoading Datasets (Strategy: {self.strategy})\n{'='*40}")
        for i in self.file_list:
            print(f'Processing slice: {i}')
            load_path = os.path.join(self.dataset_path, i)
            
            adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
            adata.var_names_make_unique()
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            if self.label:
                truth_path = os.path.join(load_path, "truth.txt")
                if not os.path.exists(truth_path): truth_path = os.path.join(load_path, f"{i}_truth.txt")
                if os.path.exists(truth_path):
                    df_meta = pd.read_csv(truth_path, sep='\t', header=None)
                    adata.obs['ground_truth'] = df_meta[1].values
                    if self.filter_na: adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()

            if self.image_emb:
                feat_path = os.path.join("features/spatial_patches_dinov2_spatial_full_dino", f"{i}.npy")
                if not os.path.exists(feat_path): raise FileNotFoundError(f"Missing: {feat_path}")
                feat_dict = np.load(feat_path, allow_pickle=True).item()
                barcodes, features_raw = feat_dict["barcodes"], feat_dict["features"]
                if features_raw.ndim == 3: features_raw = features_raw.squeeze(axis=1)
                
                barcode_to_feat = dict(zip(barcodes, features_raw))
                features_aligned = np.array([barcode_to_feat.get(bc, np.zeros_like(features_raw[0])) for bc in adata.obs_names])
                
                if 'spatial' in adata.obsm:
                    nbrs = NearestNeighbors(n_neighbors=20).fit(adata.obsm['spatial'])
                    _, indices = nbrs.kneighbors(adata.obsm['spatial'])
                    features_aligned = np.mean(features_aligned[indices], axis=1)

                scaler_img = StandardScaler()
                adata.obsm['img_emb'] = PCA(n_components=256, random_state=42).fit_transform(scaler_img.fit_transform(features_aligned))
            
            gene_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            target_pca_dim = self.fusion_output_dim if self.strategy == 'inter' else 256
            #target_pca_dim = min(target_pca_dim, gene_mat.shape[0], gene_mat.shape[1])
            adata.obsm['feat_pca'] = PCA(n_components=target_pca_dim, random_state=42).fit_transform(gene_mat)

            adata = self.construct_interaction(input_adata=adata)
            self.adata_list.append(adata)

        print('\n[SUCCESS] All slices loaded.')
        
        if self.harmonize_gene and len(self.adata_list) > 1:
            self._harmonize_features('feat_pca', self.gene_harmony_theta, "Gene")

        if self.harmonize_image and self.image_emb and len(self.adata_list) > 1:
            self._harmonize_features('img_emb', self.image_harmony_theta, "Image")

        if self._should_pretrain(): self._pretrain_fusion_model()
        if self.use_gated_fusion: self._apply_fusion_to_all_slices()
        
        return self.adata_list
    
    def _pretrain_fusion_model(self):
        if self._fusion_model_trained: return

        
        print(f"\n{'='*60}\nAuto-Pretraining (Mix={self.batch_mix_weight})\n{'='*60}")
        
        all_gene = np.vstack([a.obsm['feat_pca'] for a in self.adata_list])
        all_img = np.vstack([a.obsm['img_emb'] for a in self.adata_list])
        batch_labels = []
        for i, a in enumerate(self.adata_list): batch_labels.extend([i] * len(a))
        batch_labels = np.array(batch_labels)
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        model = BatchMixingGatedFusion(all_gene.shape[1], all_img.shape[1], 256, self.fusion_output_dim)
        trainer = BatchMixingTrainer(model, device='cuda')
        trainer.train(all_gene, all_img, batch_labels, epochs=self.pretrain_epochs, batch_size=512, 
                      batch_mix_weight=self.batch_mix_weight, verbose=True) # 🔥 使用配置的权重
        
        if self.cache_model: torch.save(model.state_dict(), self.model_cache_path)
        self._fusion_model = model
        self._fusion_model_trained = True

    def _apply_fusion_to_all_slices(self):
        print(f"\nApplying Trained Fusion Model...")
        for idx, adata in enumerate(self.adata_list):
            fused_feat, gate_weights = apply_gated_fusion(
                adata.obsm['feat_pca'],
                adata.obsm['img_emb'],
                model_path=self.model_cache_path if self.cache_model else None,
                gene_dim=adata.obsm['feat_pca'].shape[1],
                img_dim=adata.obsm['img_emb'].shape[1],
                output_dim=self.fusion_output_dim
            )
            adata.obsm['con_feat'] = fused_feat
            adata.obs['gate_weight'] = gate_weights


    def concatenate_slices(self):
        hv_genes = set(self.adata_list[0].var_names[self.adata_list[0].var['highly_variable']])
        for adata in self.adata_list[1:]:
            hv_genes = hv_genes.intersection(set(adata.var_names[adata.var['highly_variable']]))
        self.merged_adata = AnnData.concatenate(*self.adata_list, join='outer')
        self.merged_adata.obsm['feat'] = self.merged_adata.X[:, self.merged_adata.var_names.isin(hv_genes)].toarray()
        print('✓ Slices concatenated')
        return self.merged_adata
    
    
    def construct_whole_graph(self):
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        adjacency = block_diag(*matrix_list)
        self.merged_adata.obsm['graph_neigh'] = adjacency

        mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
        mask = block_diag(*mask_list)
        self.merged_adata.obsm['mask_neigh'] = mask

    def calculate_edge_weights(self):
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        if self.use_gated_fusion and 'con_feat' in self.merged_adata.obsm:
            node_emb = self.merged_adata.obsm['con_feat']
            print("✓ Using con_feat for edge weights (fused gene+image features)")
        elif 'img_emb' in self.merged_adata.obsm:
            node_emb = self.merged_adata.obsm['img_emb']
            print("✓ Using img_emb for edge weights (image features only)")
        else:
            node_emb = self.merged_adata.obsm['feat']
            print("✓ Using feat for edge weights (gene features only)")

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = np.log(edge_weights[i][non_zero_indices])
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def build_hybrid_graph(self):
        print(f"\n{'='*40}")
        print("Building GOLDEN Standard Graph (Phys=5.0, MNN=1.0, k=200)...")
        
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        spatial_adj = block_diag(*matrix_list)
        
        if 'feat_pca' in self.merged_adata.obsm:
            X_search = self.merged_adata.obsm['feat_pca']
        else:
            X_search = self.merged_adata.obsm['feat']
            
        from sklearn.neighbors import NearestNeighbors
        
        k_candidates = 200
        
        nbrs = NearestNeighbors(n_neighbors=k_candidates + 1, metric='euclidean').fit(X_search)
        distances, indices = nbrs.kneighbors(X_search)
        
        n_spot = X_search.shape[0]
        row_indices = []
        col_indices = []
        neighbor_sets = [set(indices[i, 1:]) for i in range(n_spot)]
        
        for i in tqdm(range(n_spot), desc="MNN Check"):
            candidates = indices[i, 1:]
            for j in candidates:
                if i in neighbor_sets[j]:
                    row_indices.append(i)
                    col_indices.append(j)
        print(f"   Found {len(row_indices)} MNN bridges.")

        from scipy.sparse import coo_matrix
        
        phys_rows, phys_cols = np.where(spatial_adj == 1)
        phys_data = np.full(len(phys_rows), 5.0)
        
        mnn_data = np.full(len(row_indices), 1.0)
        
        final_rows = np.concatenate([phys_rows, np.array(row_indices)])
        final_cols = np.concatenate([phys_cols, np.array(col_indices)])
        final_data = np.concatenate([phys_data, mnn_data])
        
        merged_coo = coo_matrix((final_data, (final_rows, final_cols)), shape=(n_spot, n_spot))
        
        adj_csr = merged_coo.tocsr()
        from sklearn.preprocessing import normalize
        edge_probabilities = normalize(adj_csr, norm='l1', axis=1)
        
        self.merged_adata.obsm['graph_neigh'] = edge_probabilities.toarray()
        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities.toarray()
        
        print(" GOLDEN Standard Graph built.")


    def run(self):
        self.load_data() 
        
        self.concatenate_slices()
        
        if self.strategy == 'intra':
            self.construct_whole_graph()
            self.calculate_edge_weights()
            
        elif self.strategy == 'inter':
            self.build_hybrid_graph()
        
        return self.merged_adata    

#####复杂形变###
# class LoadBatch10xAdata:
#     def __init__(self, dataset_path: str, file_list: list, 
#                  # --- 基础配置 ---
#                  n_top_genes: int = 3000, 
#                  n_neighbors: int = 5,       # 物理邻居: 5 (Dataset 1/2 通用最优)
#                  image_emb: bool = True, 
#                  label: bool = True, 
#                  filter_na: bool = True,
#                  # --- 核心模型配置 (SOTA Defaults) ---
#                  fusion_output_dim: int = 384,    # 模型容量: 384
#                  pretrain_epochs: int = 500,   
#                  batch_mix_method: str = 'simple',
#                  batch_mix_weight: float = 20.0,  # 混合权重: 20.0 (高压去批次)
                 
#                  # --- 源头对齐配置 (Source-level Harmonization) ---
#                  harmonize_image: bool = True,
#                  image_harmony_theta: float = 10.0, # 图像去噪: 10.0 (核武器级，抹平背景差异)
                 
#                  harmonize_gene: bool = True,
#                  gene_harmony_theta: float = 2.0,   # 基因保留: 2.0 (保留生物学异质性)
                 
#                  # --- 其他 ---
#                  use_gated_fusion: bool = True,
#                  auto_pretrain: bool = True,
#                  cache_model: bool = False,
#                  random_seed: int = 42):
        
#         self._fusion_model_trained = False
#         self.fusion_output_dim = fusion_output_dim
#         self.dataset_path = dataset_path
#         self.file_list = file_list
#         self.n_top_genes = n_top_genes
#         self.n_neighbors = n_neighbors
#         self.image_emb = image_emb
#         self.batch_mix_method = batch_mix_method     
#         self.fusion_output_dim = fusion_output_dim
#         self.pretrain_epochs = pretrain_epochs
#         self.batch_mix_weight = batch_mix_weight
#         self.harmonize_image = harmonize_image
#         self.image_harmony_theta = image_harmony_theta
#         self.harmonize_gene = harmonize_gene
#         self.gene_harmony_theta = gene_harmony_theta
#         self.use_gated_fusion = use_gated_fusion
#         self.auto_pretrain = auto_pretrain
#         self.cache_model = cache_model
#         self.random_seed = random_seed
#         self.label = label
#         self.filter_na = filter_na
#         self.adata_list = []
#         self.merged_adata = None
#         self.model_cache_path = self._get_model_cache_path()

#     def _get_model_cache_path(self):
#         dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
#         file_hash = hash(tuple(sorted(self.file_list))) % 10000
#         return os.path.join('fusion_model_cache', 
#                             f"fusion_{dataset_name}_{file_hash}_d{self.fusion_output_dim}_hg{int(self.harmonize_gene)}.pth")

#     def _should_pretrain(self):
#         if not self.use_gated_fusion or not self.auto_pretrain or not self.image_emb: return False
#         if self.cache_model and os.path.exists(self.model_cache_path):
#             print(f"✓ Found cached model: {self.model_cache_path}")
#             return False
#         return True
    
#     def _pretrain_fusion_model(self):
#         if self._fusion_model_trained: return
#         print(f"\n{'='*60}\nAuto-Pretraining (Clean Input Mode)\n{'='*60}")
        
#         all_gene, all_img, batch_labels = [], [], []
#         for batch_id, adata in enumerate(self.adata_list):
#             all_gene.append(adata.obsm['feat_pca']) # 这里的 feat_pca 已经是 Clean 的了
#             all_img.append(adata.obsm['img_emb'])
#             batch_labels.extend([batch_id] * len(adata))
        
#         all_gene = np.vstack(all_gene)
#         all_img = np.vstack(all_img)
#         batch_labels = np.array(batch_labels)
        
#         # 固定种子
#         import random
#         random.seed(self.random_seed)
#         np.random.seed(self.random_seed)
#         torch.manual_seed(self.random_seed)
#         if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.random_seed)
        
#         model = BatchMixingGatedFusion(
#             gene_dim=all_gene.shape[1],
#             img_dim=all_img.shape[1],
#             hidden_dim=256,
#             output_dim=self.fusion_output_dim,
#         )
        
#         trainer = BatchMixingTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
#         trainer.train(
#             all_gene, all_img, batch_labels,
#             epochs=self.pretrain_epochs,
#             batch_size=512,
#             batch_mix_weight=self.batch_mix_weight,
#             verbose=True
#         )
        
#         if self.cache_model:
#             torch.save(model.state_dict(), self.model_cache_path)
#             print(f"✓ Model cached to: {self.model_cache_path}")
        
#         self._fusion_model = model
#         self._fusion_model_trained = True

#     def construct_interaction(self, input_adata):
#         position = input_adata.obsm['spatial']
#         nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(position)
#         _, indices = nbrs.kneighbors(position)
#         n_spot = input_adata.shape[0]
#         interaction = np.zeros([n_spot, n_spot])
#         for i in range(n_spot):
#             interaction[i, indices[i, 1:]] = 1
#         adj = interaction + interaction.T
#         adj = np.where(adj > 1, 1, adj)
#         input_adata.obsm['local_graph'] = adj
#         return input_adata

#     def load_data(self):
#         for i in self.file_list:
#             print(f'--- Processing slice: {i} ---')
#             load_path = os.path.join(self.dataset_path, i)
#             adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
#             adata.var_names_make_unique()
#             sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
#             sc.pp.normalize_total(adata, target_sum=1e4)
#             sc.pp.log1p(adata)
#             # Label
#             if self.label:
#                 truth_path = os.path.join(load_path, "truth.txt")
#                 if os.path.exists(truth_path):
#                     df_meta = pd.read_csv(truth_path, sep='\t', header=None)
#                     adata.obs['ground_truth'] = df_meta[1].values
#                     if self.filter_na: adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()

#             # Image
#             if self.image_emb:
#                 feat_path = os.path.join("features/dinov2_simsiam512_6", f"{i}.npy")
#                 if not os.path.exists(feat_path): raise FileNotFoundError(f"Missing: {feat_path}")
#                 feat_dict = np.load(feat_path, allow_pickle=True).item()
#                 barcodes, features_raw = feat_dict["barcodes"], feat_dict["features"]
#                 if features_raw.ndim == 3: features_raw = features_raw.squeeze(axis=1)
                
#                 barcode_to_feat = dict(zip(barcodes, features_raw))
#                 features_aligned = np.array([barcode_to_feat.get(bc, np.zeros_like(features_raw[0])) for bc in adata.obs_names])

#                 if 'spatial' in adata.obsm:
#                     nbrs = NearestNeighbors(n_neighbors=20).fit(adata.obsm['spatial'])
#                     _, indices = nbrs.kneighbors(adata.obsm['spatial'])
#                     features_aligned = np.mean(features_aligned[indices], axis=1)

#                 scaler_img = StandardScaler()
#                 img_std = scaler_img.fit_transform(features_aligned)
#                 adata.obsm['img_emb'] = PCA(n_components=256, random_state=42).fit_transform(img_std)

#                 gene_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
#                 target_dim = self.fusion_output_dim if self.fusion_output_dim > 0 else 256
#                 adata.obsm['feat_pca'] = PCA(n_components=target_dim, random_state=42).fit_transform(gene_mat)

#             adata = self.construct_interaction(input_adata=adata)
#             self.adata_list.append(adata)

#         print('\n[SUCCESS] All slices loaded.')
        
#         # (Input-Level Harmony)
#         if self.harmonize_gene and len(self.adata_list) > 1:
#             self._harmonize_features(key='feat_pca', theta=self.gene_harmony_theta, name="Gene")

#         # (Input-Level Harmony)
#         if self.harmonize_image and self.image_emb and len(self.adata_list) > 1:
#             self._harmonize_features(key='img_emb', theta=self.image_harmony_theta, name="Image")

#         if self._should_pretrain():
#             self._pretrain_fusion_model()
        
#         if self.use_gated_fusion and self.image_emb:
#             self._apply_fusion_to_all_slices()
        
#         return self.adata_list
    
#     def _harmonize_features(self, key, theta, name):
#         print(f"\n{'='*40}\nHarmonizing {name} Features (Theta={theta})\n{'='*40}")
#         all_emb = np.vstack([a.obsm[key] for a in self.adata_list])
#         batch_labels = []
#         for i, a in enumerate(self.adata_list): batch_labels.extend([str(i)] * len(a))
#         meta = pd.DataFrame({'batch': batch_labels})
#         import harmonypy as hm
#         harmony_out = hm.run_harmony(
#             all_emb, meta, 'batch',
#             max_iter_harmony=20, theta=theta, sigma=0.1, verbose=False
#         )
        
#         emb_corrected = harmony_out.Z_corr
#         start = 0
#         for adata in self.adata_list:
#             end = start + len(adata)
#             adata.obsm[f'{key}_raw'] = adata.obsm[key].copy()
#             adata.obsm[key] = emb_corrected[start:end]
#             start = end
#         print(f"✓ {name} features harmonized")

#     def _apply_fusion_to_all_slices(self):
#         print(f"\nApplying Trained Fusion Model...")
#         for idx, adata in enumerate(self.adata_list):
#             fused_feat, gate_weights = apply_gated_fusion(
#                 adata.obsm['feat_pca'],
#                 adata.obsm['img_emb'],
#                 model_path=self.model_cache_path if self.cache_model else None,
#                 gene_dim=adata.obsm['feat_pca'].shape[1],
#                 img_dim=adata.obsm['img_emb'].shape[1],
#                 output_dim=self.fusion_output_dim
#             )
#             adata.obsm['con_feat'] = fused_feat
#             adata.obs['gate_weight'] = gate_weights

#     def concatenate_slices(self):
#         if not self.adata_list: return None
#         hv_genes = set(self.adata_list[0].var_names[self.adata_list[0].var['highly_variable']])
#         for adata in self.adata_list[1:]:
#             hv_genes = hv_genes.intersection(set(adata.var_names[adata.var['highly_variable']]))
#         adata = AnnData.concatenate(*self.adata_list, join='outer')
#         feat = adata.X[:, adata.var_names.isin(hv_genes)]
#         if isinstance(feat, (csc_matrix, csr_matrix)): feat = feat.toarray()
#         adata.obsm['feat'] = feat
#         self.merged_adata = adata
#         return self.merged_adata
    
#     def build_hybrid_graph(self):
#         print(f"\n{'='*40}")
#         print("Building GOLDEN Standard Graph (Phys=5.0, MNN=1.0, k=200)...")
        
#         # 1. 物理骨架
#         matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
#         spatial_adj = block_diag(*matrix_list)
        
#         if 'feat_pca' in self.merged_adata.obsm:
#             X_search = self.merged_adata.obsm['feat_pca']
#         else:
#             X_search = self.merged_adata.obsm['feat']
            
#         from sklearn.neighbors import NearestNeighbors
        
#         k_candidates = 200
        
#         nbrs = NearestNeighbors(n_neighbors=k_candidates + 1, metric='euclidean').fit(X_search)
#         distances, indices = nbrs.kneighbors(X_search)
        
#         n_spot = X_search.shape[0]
#         row_indices = []
#         col_indices = []
#         neighbor_sets = [set(indices[i, 1:]) for i in range(n_spot)]
        
#         for i in tqdm(range(n_spot), desc="MNN Check"):
#             candidates = indices[i, 1:]
#             for j in candidates:
#                 if i in neighbor_sets[j]:
#                     row_indices.append(i)
#                     col_indices.append(j)
#         print(f"   Found {len(row_indices)} MNN bridges.")

#         from scipy.sparse import coo_matrix
        
#         phys_rows, phys_cols = np.where(spatial_adj == 1)
#         phys_data = np.full(len(phys_rows), 5.0)
        
#         mnn_data = np.full(len(row_indices), 1.0)
        
#         final_rows = np.concatenate([phys_rows, np.array(row_indices)])
#         final_cols = np.concatenate([phys_cols, np.array(col_indices)])
#         final_data = np.concatenate([phys_data, mnn_data])
        
#         merged_coo = coo_matrix((final_data, (final_rows, final_cols)), shape=(n_spot, n_spot))
        
#         adj_csr = merged_coo.tocsr()
#         from sklearn.preprocessing import normalize
#         edge_probabilities = normalize(adj_csr, norm='l1', axis=1)
        
#         self.merged_adata.obsm['graph_neigh'] = edge_probabilities.toarray()
#         self.merged_adata.obsm['edge_probabilities'] = edge_probabilities.toarray()
        
#         print("✓ GOLDEN Standard Graph built.")
#         print(f"{'='*40}\n")


#     def run(self):
#         self.load_data() 
#         self.concatenate_slices()
        
#         self.build_hybrid_graph()
        
#         print('\n✓ Data processing completed!')
#         return self.merged_adata

#####生物学重复##
# class LoadBatch10xAdata:
#     def __init__(self, dataset_path: str, file_list: list, 
#                  n_top_genes: int = 3000, n_neighbors: int = 5,
#                  image_emb: bool = False, label: bool = True, 
#                  filter_na: bool = True, do_log: bool = True,
#                  use_gated_fusion: bool = True,
#                  fusion_output_dim: int = 384,
#                  auto_pretrain: bool = True,
#                  pretrain_epochs: int = 500,
#                  batch_mix_weight: float = 0.01,         # 批次混合权重
#                  batch_mix_method: str = 'simple',      # 批次混合方法
#                  cache_model: bool = True,
#                  random_seed: int = 42,
#                  harmonize_image: bool = True,
#                  image_harmony_theta: float = 0.5):
        
#         self.dataset_path = dataset_path
#         self.file_list = file_list
#         self.n_top_genes = n_top_genes
#         self.n_neighbors = n_neighbors
#         self.adata_list = []
#         self.adata_len = []
#         self.merged_adata = None
#         self.image_emb = image_emb
#         self.label = label
#         self.filter_na = filter_na
#         self.do_log = do_log
#         self.harmonize_image = harmonize_image
#         self.image_harmony_theta = image_harmony_theta
#         self.use_gated_fusion = use_gated_fusion
#         self.fusion_output_dim = fusion_output_dim
#         self.auto_pretrain = auto_pretrain
#         self.pretrain_epochs = pretrain_epochs
#         self.batch_mix_weight = batch_mix_weight      
#         self.batch_mix_method = batch_mix_method     
#         self.cache_model = cache_model
#         self.random_seed = random_seed
        
#         self.model_cache_path = self._get_model_cache_path()
#         self._fusion_model_trained = False
#         self._fusion_model = None


#     def _get_model_cache_path(self):
#         dataset_name = os.path.basename(self.dataset_path.rstrip('/'))
#         file_hash = hash(tuple(sorted(self.file_list))) % 10000
        
#         cache_dir = 'fusion_model_cache'
#         os.makedirs(cache_dir, exist_ok=True)
        
#         return os.path.join(cache_dir, f"fusion_{dataset_name}_{file_hash}_dim{self.fusion_output_dim}_w{self.batch_mix_weight}.pth")
    
#     def _should_pretrain(self):
#         if not self.use_gated_fusion or not self.auto_pretrain:
#             return False
        
#         if not self.image_emb:
#             return False
                
#         if self.cache_model and os.path.exists(self.model_cache_path):
#             print(f"✓ Found cached model: {self.model_cache_path}")
#             return False
        
#         return True
    
#     def _pretrain_fusion_model(self):
#         if self._fusion_model_trained:
#             return
        
#         print("\n" + "="*60)
#         print("Auto-Pretraining Batch-Mixing Gated Fusion Model")
#         print("="*60)
        
#         print("\n[1/2] Collecting features from loaded slices...")
#         all_gene = []
#         all_img = []
#         batch_labels = []
        
#         for batch_id, adata in enumerate(self.adata_list):
#             all_gene.append(adata.obsm['feat_pca'])
#             all_img.append(adata.obsm['img_emb'])
#             batch_labels.extend([batch_id] * len(adata))
#             print(f"  Batch {batch_id}: {len(adata)} spots")
        
#         all_gene = np.vstack(all_gene)
#         all_img = np.vstack(all_img)
#         batch_labels = np.array(batch_labels)
        
#         gene_dim = all_gene.shape[1]
#         img_dim = all_img.shape[1]
        
#         print(f"\n  Total: {len(all_gene)} spots from {len(self.adata_list)} batches")
#         print(f"  Gene dim: {gene_dim}, Image dim: {img_dim}, Output dim: {self.fusion_output_dim}")
        
#         print(f"\n[2/2] Training fusion model ({self.pretrain_epochs} epochs)...")
#         print(f"  Batch mix weight: {self.batch_mix_weight}")
#         print(f"  Batch mix method: {self.batch_mix_method}")
        
#         import random
#         random.seed(self.random_seed)
#         np.random.seed(self.random_seed)
#         torch.manual_seed(self.random_seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.random_seed)
        
#         model = BatchMixingGatedFusion(
#             gene_dim=gene_dim,
#             img_dim=img_dim,
#             hidden_dim=256,
#             output_dim=self.fusion_output_dim,
#         )
        
#         trainer = BatchMixingTrainer(model, device='cuda')
#         trainer.train(
#             all_gene, 
#             all_img, 
#             batch_labels,
#             epochs=self.pretrain_epochs,
#             batch_size=512,
#             batch_mix_weight=self.batch_mix_weight,    # 🔥 新参数
#             batch_mix_method=self.batch_mix_method,    # 🔥 新参数
#             verbose=True
#         )
        
#         if self.cache_model:
#             torch.save(model.state_dict(), self.model_cache_path)
#             print(f"\n✓ Model cached to: {self.model_cache_path}")
        
#         self._fusion_model = model
#         self._fusion_model_trained = True
        
#         print("="*60)
#         print("Auto-Pretraining Completed!")
#         print("="*60 + "\n")

#     def construct_interaction(self, input_adata):
#         position = input_adata.obsm['spatial']
#         distance_matrix = ot.dist(position, position, metric='euclidean')
#         n_spot = distance_matrix.shape[0]
#         interaction = np.zeros([n_spot, n_spot])
        
#         for i in range(n_spot):
#             vec = distance_matrix[i, :]
#             distance = vec.argsort()
#             for t in range(1, self.n_neighbors + 1):
#                 y = distance[t]
#                 interaction[i, y] = 1

#         adj = interaction + interaction.T
#         adj = np.where(adj > 1, 1, adj)
#         input_adata.obsm['local_graph'] = adj
#         return input_adata

#     def load_data(self):
#         for i in self.file_list:
#             print(f'--- Processing slice: {i} ---')
#             load_path = os.path.join(self.dataset_path, i)
            
#             adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
#             adata.var_names_make_unique()
            
#             sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
#             sc.pp.normalize_total(adata, target_sum=1e4)
#             sc.pp.log1p(adata)
            
#             if self.label:
#                 truth_path = os.path.join(load_path, f"truth.txt")
#                 if os.path.exists(truth_path):
#                     df_meta = pd.read_csv(truth_path, sep='\t', header=None)
#                     adata.obs['ground_truth'] = df_meta[1].values
#                     if self.filter_na:
#                         adata = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
#                         print(f"[{i}] Filtered NA labels.")

#             if self.image_emb:
#                 feat_path = os.path.join("features/dinov2_simsiam512_6", f"{i}.npy")
#                 if not os.path.exists(feat_path):
#                     raise FileNotFoundError(f"Missing image features: {feat_path}")
                
#                 feat_dict = np.load(feat_path, allow_pickle=True).item()
#                 barcodes, features_raw = feat_dict["barcodes"], feat_dict["features"]
                
#                 if features_raw.ndim == 3:
#                     features_raw = features_raw.squeeze(axis=1)

#                 barcode_to_feat = dict(zip(barcodes, features_raw))
#                 features_aligned = []
#                 for spot_id in adata.obs_names:
#                     if spot_id in barcode_to_feat:
#                         features_aligned.append(barcode_to_feat[spot_id])
#                     else:
#                         features_aligned.append(np.zeros_like(features_raw[0]))
#                 features_aligned = np.stack(features_aligned, axis=0)

#                 if 'spatial' in adata.obsm:
#                     k_smooth = 20
#                     nbrs = NearestNeighbors(n_neighbors=k_smooth).fit(adata.obsm['spatial'])
#                     _, indices = nbrs.kneighbors(adata.obsm['spatial'])
#                     features_aligned = np.mean(features_aligned[indices], axis=1)

#                 scaler_img = StandardScaler()
#                 img_std = scaler_img.fit_transform(features_aligned)
#                 adata.obsm['img_emb'] = PCA(n_components=256, random_state=42).fit_transform(img_std)

#                 gene_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
                
#                 adata.obsm['feat_pca'] = PCA(n_components=256, random_state=42).fit_transform(gene_mat)

#                 print(f"[{i}] Gene PCA: {adata.obsm['feat_pca'].shape}, Image emb: {adata.obsm['img_emb'].shape}")

#             adata = self.construct_interaction(input_adata=adata)
#             self.adata_list.append(adata)

#         print('\n[SUCCESS] All slices loaded.')
        
#         if self.harmonize_image and self.image_emb and len(self.adata_list) > 1:
#             self._harmonize_image_features()

#         if self._should_pretrain():
#             self._pretrain_fusion_model()
        

#         if self.use_gated_fusion and self.image_emb:
#             self._apply_fusion_to_all_slices()
        
#         return self.adata_list
    
#     def _harmonize_image_features(self):
#         print("\n" + "="*60)
#         print("Harmonizing Image Features Across Batches")
#         print("="*60)
        
#         print(f"批次数量: {len(self.adata_list)}")
#         print(f"总观测数: {sum(len(adata) for adata in self.adata_list)}")
#         import harmonypy as hm
        
#         all_img_emb = []
#         batch_labels = []
        
#         for batch_id, adata in enumerate(self.adata_list):
#             print(f"批次 {batch_id}: {len(adata)} 个观测, img_emb形状: {adata.obsm['img_emb'].shape}")
#             all_img_emb.append(adata.obsm['img_emb'])
#             batch_labels.extend([str(batch_id)] * len(adata))
        
#         all_img_emb = np.vstack(all_img_emb)
        
#         import pandas as pd
#         meta = pd.DataFrame({'batch': batch_labels})
        
#         harmony_out = hm.run_harmony(
#             all_img_emb,
#             meta,
#             'batch',
#             max_iter_harmony=20,
#             theta=self.image_harmony_theta,           
#             sigma=0.1,
#             verbose=False
#         )
        
#         img_emb_corrected = harmony_out.Z_corr
        
#         start_idx = 0
#         for adata in self.adata_list:
#             end_idx = start_idx + len(adata)
#             slice_data = img_emb_corrected[start_idx:end_idx]
        
#             adata.obsm['img_emb_raw'] = adata.obsm['img_emb'].copy()  # 保存原始
#             adata.obsm['img_emb'] = slice_data
#             start_idx = end_idx
        
#         print("✓ Image features harmonized across batches")
#         print("="*60 + "\n")

#     def _apply_fusion_to_all_slices(self):
#         print("\n" + "="*60)
#         print("Applying Trained Fusion Model to All Slices")
#         print("="*60 + "\n")
        
#         for idx, adata in enumerate(self.adata_list):
#             print(f"Fusing slice {idx+1}/{len(self.adata_list)}...")
            
#             fused_feat, gate_weights = apply_gated_fusion(
#                 adata.obsm['feat_pca'],
#                 adata.obsm['img_emb'],
#                 model_path=self.model_cache_path if self.cache_model else None,
#                 gene_dim=adata.obsm['feat_pca'].shape[1],
#                 img_dim=adata.obsm['img_emb'].shape[1],
#                 output_dim=self.fusion_output_dim
#             )
            
#             adata.obsm['con_feat'] = fused_feat
#             adata.obs['gate_weight'] = gate_weights
            
#             print(f"  ✓ Fused shape: {fused_feat.shape}, Mean α: {gate_weights.mean():.3f}")
        
#         print("\n" + "="*60)
#         print("Fusion Applied to All Slices!")
#         print("="*60 + "\n")

#     def concatenate_slices(self):
#         highly_variable_genes_set = set(self.adata_list[0].var['highly_variable'][self.adata_list[0].var['highly_variable']].index)
        
#         for adata in self.adata_list[1:]:
#             current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
#             highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)

#         adata = AnnData.concatenate(*self.adata_list, join='outer')
#         adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
        
#         if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
#             feat = adata_Vars.X.toarray()[:, ]
#         else:
#             feat = adata_Vars.X[:, ]

#         adata.obsm['feat'] = feat
#         self.merged_adata = adata
#         print('✓ Slices concatenated')
#         return self.merged_adata

#     def construct_whole_graph(self):
#         matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
#         adjacency = block_diag(*matrix_list)
#         self.merged_adata.obsm['graph_neigh'] = adjacency

#         mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
#         mask = block_diag(*mask_list)
#         self.merged_adata.obsm['mask_neigh'] = mask

#     def calculate_edge_weights(self):
#         graph_neigh = self.merged_adata.obsm['graph_neigh']
#         if self.use_gated_fusion and 'con_feat' in self.merged_adata.obsm:
#             node_emb = self.merged_adata.obsm['con_feat']
#             print("✓ Using con_feat for edge weights (fused gene+image features)")
#         elif 'img_emb' in self.merged_adata.obsm:
#             node_emb = self.merged_adata.obsm['img_emb']
#             print("✓ Using img_emb for edge weights (image features only)")
#         else:
#             node_emb = self.merged_adata.obsm['feat']
#             print("✓ Using feat for edge weights (gene features only)")

#         num_nodes = node_emb.shape[0]
#         edge_weights = np.zeros_like(graph_neigh)

#         for i in tqdm(range(num_nodes), desc="Calculating distances"):
#             for j in range(num_nodes):
#                 if graph_neigh[i, j] == 1:
#                     edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

#         edge_probabilities = np.zeros_like(edge_weights)
#         for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
#             non_zero_indices = edge_weights[i] != 0
#             if non_zero_indices.any():
#                 non_zero_weights = np.log(edge_weights[i][non_zero_indices])
#                 softmax_weights = softmax(non_zero_weights)
#                 edge_probabilities[i][non_zero_indices] = softmax_weights

#         self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

#     def calculate_edge_weights_gene(self):
#         """计算基因边权重"""
#         graph_neigh = self.merged_adata.obsm['graph_neigh']
#         node_emb = self.merged_adata.obsm['feat']
#         scaler = StandardScaler()
#         embedding = scaler.fit_transform(node_emb)
#         pca = PCA(n_components=64, random_state=42)
#         embedding = pca.fit_transform(embedding)
#         node_emb = embedding

#         num_nodes = node_emb.shape[0]
#         edge_weights = np.zeros((num_nodes, num_nodes))

#         for i in tqdm(range(num_nodes), desc="Calculating distances"):
#             for j in range(num_nodes):
#                 if graph_neigh[i, j] == 1:
#                     edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

#         edge_probabilities = np.zeros_like(edge_weights)
#         for i in range(num_nodes):
#             non_zero_indices = edge_weights[i] != 0
#             if non_zero_indices.any():
#                 non_zero_weights = edge_weights[i][non_zero_indices]
#                 softmax_weights = softmax(non_zero_weights)
#                 edge_probabilities[i][non_zero_indices] = softmax_weights

#         self.merged_adata.obsm['edge_probabilities'] = edge_probabilities
    
#     def run(self):
#         self.load_data() 
#         self.concatenate_slices()
#         self.construct_whole_graph()        
#         if self.image_emb:
#             self.calculate_edge_weights()
#         else:
#             self.calculate_edge_weights_gene()
        
        
#         print('\n✓ Data processing completed!')
#         return self.merged_adata

####STAIG原方法##
# class LoadBatch10xAdata:
#     def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
#                  image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
#         self.dataset_path = dataset_path  
#         self.file_list = file_list  
#         self.n_top_genes = n_top_genes
#         self.n_neighbors = n_neighbors
#         self.adata_list = []
#         self.adata_len = []
#         self.merged_adata = None
#         self.image_emb = image_emb
#         self.label = label
#         self.filter_na = filter_na
#         self.do_log = do_log

#     def construct_interaction(self, input_adata):
#         position = input_adata.obsm['spatial']
#         distance_matrix = ot.dist(position, position, metric='euclidean')
#         n_spot = distance_matrix.shape[0]
#         interaction = np.zeros([n_spot, n_spot])
#         for i in range(n_spot):
#             vec = distance_matrix[i, :]
#             distance = vec.argsort()
#             for t in range(1, self.n_neighbors + 1):
#                 y = distance[t]
#                 interaction[i, y] = 1

#         adj = interaction + interaction.T
#         adj = np.where(adj > 1, 1, adj)
#         input_adata.obsm['local_graph'] = adj
#         return input_adata


    
#     def load_data(self):
#         for i in self.file_list:
#             print('now load: ' + i)
#             load_path = os.path.join(self.dataset_path, i)
#             adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
#             adata.var_names_make_unique()
#             sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
#             sc.pp.normalize_total(adata, target_sum=1e4)
#             sc.pp.log1p(adata)
#             if self.label:
#                 truth_filename = i + '_truth.txt'
#                 df_meta = pd.read_csv(os.path.join(load_path, truth_filename), sep='\t', header=None)
#                 df_meta_layer = df_meta[1]
#                 adata.obs['ground_truth'] = df_meta_layer.values
#                 print(i + ' load label done')
#                 if self.filter_na:
#                     adata = adata[~pd.isnull(adata.obs['ground_truth'])]
#                     print(i + ' filter NA done')
#             if self.image_emb:
#                 data = np.load(os.path.join(load_path, 'embeddings.npy'))
#                 data = data.reshape(data.shape[0], -1)
#                 scaler = StandardScaler()
#                 embedding = scaler.fit_transform(data)
#                 pca = PCA(n_components=128, random_state=42)
#                 embedding = pca.fit_transform(embedding)
#                 adata.obsm['img_emb'] = embedding
#                 print(i + ' load img embedding done')
#             adata = self.construct_interaction(input_adata=adata)
#             print(i + ' build local graph done')
#             self.adata_list.append(adata)
#             self.adata_len.append(adata.X.shape[0])
#             print(i + ' added to list')
#         print('load all slices done')

#         return self.adata_list
    
#     def concatenate_slices(self):

#         highly_variable_genes_set = set(self.adata_list[0].var['highly_variable'][self.adata_list[0].var['highly_variable']].index)


#         for adata in self.adata_list[1:]:

#             current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
#             highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)

#         adata = AnnData.concatenate(*self.adata_list, join='outer')

#         adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
#         if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
#             feat = adata_Vars.X.toarray()[:, ]
#         else:
#             feat = adata_Vars.X[:, ]

#         adata.obsm['feat'] = feat

#         self.merged_adata = adata
#         print('merge done')
#         return self.merged_adata

#     def construct_whole_graph(self):
#         matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
#         adjacency = block_diag(*matrix_list)
#         self.merged_adata.obsm['graph_neigh'] = adjacency

#         mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
#         mask = block_diag(*mask_list)
#         self.merged_adata.obsm['mask_neigh'] = mask

#     def calculate_edge_weights(self):
#         graph_neigh = self.merged_adata.obsm['graph_neigh']
#         node_emb = self.merged_adata.obsm['img_emb']
#         num_nodes = node_emb.shape[0]
#         edge_weights = np.zeros_like(graph_neigh)  

#         for i in tqdm(range(num_nodes), desc="Calculating distances"):  
#             for j in range(num_nodes):
#                 if graph_neigh[i, j] == 1:  
#                     edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

#         edge_probabilities = np.zeros_like(edge_weights)
#         for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
#             non_zero_indices = edge_weights[i] != 0
#             if non_zero_indices.any():  
#                 non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
#                 softmax_weights = softmax(non_zero_weights)
#                 edge_probabilities[i][non_zero_indices] = softmax_weights

#         self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

#     def calculate_edge_weights_gene(self):

#         graph_neigh = self.merged_adata.obsm['graph_neigh']
#         node_emb = self.merged_adata.obsm['feat']
#         scaler = StandardScaler()
#         embedding = scaler.fit_transform(node_emb)
#         pca = PCA(n_components=64, random_state=42)
#         embedding = pca.fit_transform(embedding)
#         node_emb = embedding

#         num_nodes = node_emb.shape[0]
#         edge_weights = np.zeros((num_nodes, num_nodes))

#         for i in tqdm(range(num_nodes), desc="Calculating distances"):
#             for j in range(num_nodes):
#                 if graph_neigh[i, j] == 1:  
#                     edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

#         edge_probabilities = np.zeros_like(edge_weights)
#         for i in range(num_nodes):
#             non_zero_indices = edge_weights[i] != 0
#             if non_zero_indices.any():
#                 non_zero_weights = edge_weights[i][non_zero_indices]
#                 softmax_weights = softmax(non_zero_weights)
#                 edge_probabilities[i][non_zero_indices] = softmax_weights

#         self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

#     def run(self):
#         self.load_data()
#         self.concatenate_slices()
#         self.construct_whole_graph()
#         if self.image_emb:
#             self.calculate_edge_weights()
#         else:
#             self.calculate_edge_weights_gene()
#         print('merge adata load done')
#         return self.merged_adata    


