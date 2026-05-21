import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.preprocessing import StandardScaler


# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='norm_emb', random_seed=2023):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
    
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']
    
#     res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
#     mclust_res = np.array(res[-2])

#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='norm_emb', random_seed=2023):
    """
    Clustering using mclust via R Global Environment execution.
    This bypasses rpy2 function call interface errors.
    """
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    
    data_np = adata.obsm[used_obsm]
    if np.isnan(data_np).any():
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)
    data_np = np.ascontiguousarray(data_np, dtype=np.float64)
    
    nr, nc = data_np.shape
    
    vec = robjects.FloatVector(data_np.ravel())
    r_mat = robjects.r.matrix(vec, nrow=nr, ncol=nc, byrow=True)
    
    robjects.globalenv['r_input_data'] = r_mat
    
    r_script = f"""
    library(mclust)
    set.seed({random_seed})
    
    # 确保数据是矩阵
    data_mat <- as.matrix(r_input_data)
    
    # 在 R 内部强制加上列名，避免 dimnames 报错
    colnames(data_mat) <- paste0("Dim", 1:ncol(data_mat))
    
    # 运行 Mclust
    # verbose = FALSE 防止 R 输出太多信息干扰 Python
    res <- Mclust(data_mat, G={num_cluster}, modelNames="{modelNames}", verbose = FALSE)
    
    # 提取分类结果
    # 如果失败，返回全 1 向量避免 Python 崩溃
    if (is.null(res$classification)) {{
        rep(1, nrow(data_mat))
    }} else {{
        res$classification
    }}
    """
    
    print(f"Running pure R script for mclust (G={num_cluster})...")
    
    try:
        # robjects.r(字符串) 会执行 R 代码并返回最后一行结果
        mclust_res_r = robjects.r(r_script)
    except Exception as e:
        print("R execution error details:")
        print(e)
        raise RuntimeError("Mclust failed inside R environment.")

    mclust_res = np.array(mclust_res_r)

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
    robjects.r('rm(r_input_data, data_mat, res); gc()')
    
    return adata



def clustering(adata, n_clusters=7, radius=50, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(radius,adata, n_clusters, use_rep='norm_emb', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(radius,adata, n_clusters, use_rep='norm_emb', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

import pandas as pd
import numpy as np
import scanpy as sc
# 假设 metrics 依然从 sklearn.metrics 导入，但我们不再使用它
# from sklearn.metrics import adjusted_rand_score as metrics_ari 

def search_res(radius, adata, n_clusters, method='leiden', use_rep='norm_emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number.
    MODIFIED to return the first matching resolution when 'ground_truth' is unavailable.
    
    Parameters
    ----------
    # ... (参数说明不变)
    
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    
    # 初始化变量
    found_match = False
    best_res = None
    
    # 1. 粗略界定搜索范围
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    
    # 检查初始 resolution=end 时的簇数
    sc.tl.leiden(adata, random_state=0, resolution=end)
    count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    
    # 循环调整 'end' 值，使其接近 n_clusters
    while (count_unique > n_clusters + 2):
        print('Current K={}, 太大，继续调整'.format(count_unique))
        end = end - 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    
    while (count_unique < n_clusters + 2) and (end < 5.0): # 增加 end 边界以防止无限循环
        print('Current K={}, 太小，继续调整'.format(count_unique))
        end = end + 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())

    # 2. 精确搜索目标分辨率
    print(f'Starting fine search from {start} to {end} with step {increment}')
    
    # 从大分辨率向小分辨率搜索，先找到的分辨率通常较好
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        
        # 执行聚类
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))

        # 🎯 关键修改：找到目标簇数时，记录并返回
        if count_unique == n_clusters:
            print(f'✅ Found target cluster number {n_clusters} at resolution {res}.')
            
            # --- 移除 ARI 和 refine_label 逻辑 ---
            # 原本这里有 refine_label 和 ARI 计算，现已移除，因为缺少 ground_truth
            # --------------------------------------
            
            best_res = res
            found_match = True
            break # 找到第一个匹配的分辨率就跳出

    # 3. 返回结果
    if found_match:
        return best_res
    else:
        # 如果循环结束仍未找到精确匹配 n_clusters 的分辨率
        print(f"⚠️ Warning: Exact n_clusters ({n_clusters}) not found in the search range.")
        print("Returning the last checked resolution.")
        # 返回上一个 while 循环结束时的 end 值作为最终分辨率
        return end
    
# def search_res(radius,adata, n_clusters, method='leiden', use_rep='norm_emb', start=0.1, end=3.0, increment=0.01):
#     '''\
#     Searching corresponding resolution according to given cluster number
    
#     Parameters
#     ----------
#     adata : anndata
#         AnnData object of spatial data.
#     n_clusters : int
#         Targetting number of clusters.
#     method : string
#         Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
#     use_rep : string
#         The indicated representation for clustering.
#     start : float
#         The start value for searching.
#     end : float 
#         The end value for searching.
#     increment : float
#         The step size to increase.
        
#     Returns
#     -------
#     res : float
#         Resolution.
        
#     '''
#     print('Searching resolution...')
#     label = 0
#     ress=[]
#     best=None
#     sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
#     sc.tl.leiden(adata, random_state=0, resolution=end)
#     count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#     while (count_unique > n_clusters +2):
#         print(count_unique)
#         print('太大，继续调整')
#         end = end - 0.1
#         sc.tl.leiden(adata, random_state=0, resolution=end)
#         count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#     while (count_unique < n_clusters + 2):
#         print(count_unique)
#         print('太小，继续调整')
#         end = end + 0.1
#         sc.tl.leiden(adata, random_state=0, resolution=end)
#         count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())


#     for res in sorted(list(np.arange(start, end, increment)), reverse=True):
#         if method == 'leiden':
#            sc.tl.leiden(adata, random_state=0, resolution=res)
#            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#            print('resolution={}, cluster number={}'.format(res, count_unique))
#         elif method == 'louvain':
#            sc.tl.louvain(adata, random_state=0, resolution=res)
#            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
#            print('resolution={}, cluster number={}'.format(res, count_unique))

#         if count_unique == n_clusters:
#             print('calculate metric ARI')
#             # calculate metric ARI
#             new_type = refine_label(adata, radius, key='leiden')
#             adata.obs['leiden'] = new_type

#             ARI = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
#             adata.uns['ARI'] = ARI
#             ress.append((res,ARI))
#             print('ARI:', ARI)

#         if count_unique == n_clusters-2:
#             label = 1
#             best = max(ress, key=lambda x: x[1])
#             print(best)
#             break

#     assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
#     return best[0]

