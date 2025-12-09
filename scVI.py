#!/usr/bin/env python3
"""
scVI 批次校正程序 - 完整流程
包含数据加载、预处理和批次校正
输入: GSE132044 原始数据文件
输出: scVI校正后的h5ad文件和可视化结果
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components
from scipy.io import mmread

# 尝试导入scVI
try:
    import scvi
    from scvi.model import SCVI
    SCVI_AVAILABLE = True
except ImportError:
    print("警告: scVI 未安装，将尝试安装...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "scvi-tools"])
        import scvi
        from scvi.model import SCVI
        SCVI_AVAILABLE = True
        print("scVI 安装成功!")
    except:
        SCVI_AVAILABLE = False
        print("scVI 安装失败，请手动安装: pip install scvi-tools")

# 设置绘图风格
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 100
sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(8, 6))

def load_gse132044_data_correct():
    """正确加载 GSE132044 数据集，处理维度不匹配问题"""
    print("正在加载 GSE132044 数据集...")
    
    # 加载细胞元数据
    cells = pd.read_csv('GSE132044_cortex_mm10_cell.tsv.gz', sep='\t', index_col=0)
    print(f"细胞元数据形状: {cells.shape}")
    
    # 加载基因信息
    genes = pd.read_csv('GSE132044_cortex_mm10_gene.tsv.gz', sep='\t', header=None, names=['gene_ids'])
    print(f"基因数量: {len(genes)}")
    
    # 使用 mmread 加载 MTX 文件
    print("正在加载 MTX 表达矩阵...")
    counts = mmread('GSE132044_cortex_mm10_count_matrix.mtx.gz')
    print(f"原始 MTX 矩阵形状: {counts.shape} (基因 × 细胞)")
    
    # 转置为 cells × genes
    counts = counts.T.tocsr()
    print(f"转置后矩阵形状: {counts.shape} (细胞 × 基因)")
    
    # 检查维度是否匹配
    if counts.shape[0] != len(cells):
        print(f"警告: 表达矩阵细胞数 ({counts.shape[0]}) 与元数据细胞数 ({len(cells)}) 不匹配")
        print("将截断到较小的维度...")
        
        # 取较小的维度
        min_cells = min(counts.shape[0], len(cells))
        counts = counts[:min_cells, :]
        cells = cells.iloc[:min_cells]
        print(f"调整后: {min_cells} 个细胞")
    
    # 创建 AnnData 对象
    adata = sc.AnnData(X=counts)
    adata.obs = cells
    adata.var = genes.set_index('gene_ids')
    
    print(f"AnnData 对象创建完成: {adata.shape}")
    print(f"细胞数: {adata.n_obs}, 基因数: {adata.n_vars}")
    
    # 检查批次信息
    print("\n=== 检查批次信息 ===")
    # 从细胞名称中提取测序方法信息
    # 根据文件检查结果，细胞名称格式如: "Cortex1.Smart-seq2.p1_A1"
    method_info = adata.obs.index.str.split('.').str[1]
    adata.obs['method'] = method_info
    print(f"测序方法分布:\n{adata.obs['method'].value_counts()}")
    
    return adata

def preprocess_adata(adata):
    """数据预处理流程"""
    print("开始数据预处理...")
    
    # 复制原始数据
    adata_processed = adata.copy()
    
    # 设置批次信息
    adata_processed.obs['batch'] = adata_processed.obs['method']
    print(f"批次信息: {adata_processed.obs['batch'].value_counts()}")
    
    # 1. 基本QC指标
    print("1. 计算QC指标...")
    # 小鼠线粒体基因通常以'mt-'开头（注意：基因名称格式为 ENSMUSG00000000001_Gnai3）
    # 我们需要从基因名称的第二部分（下划线后）检查是否以'mt-'开头
    gene_symbols = adata_processed.var_names.str.split('_').str[1]
    adata_processed.var['mt'] = gene_symbols.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(
        adata_processed, 
        qc_vars=['mt'], 
        percent_top=None, 
        log1p=False, 
        inplace=True
    )
    
    # 2. 过滤细胞和基因
    print("2. 过滤低质量细胞和基因...")
    print(f"过滤前: {adata_processed.shape}")
    
    # 过滤低表达基因
    sc.pp.filter_genes(adata_processed, min_cells=3)
    
    # 过滤低质量细胞（基于基因数和线粒体比例）
    n_genes_threshold = 200
    mt_threshold = 20  # 线粒体基因比例阈值
    
    cell_filter = (adata_processed.obs['n_genes_by_counts'] >= n_genes_threshold) & \
                  (adata_processed.obs['pct_counts_mt'] <= mt_threshold)
    adata_processed = adata_processed[cell_filter, :]
    
    print(f"过滤后: {adata_processed.shape}")
    
    # 3. 标准化
    print("3. 数据标准化...")
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)
    
    # 4. 识别高变基因
    print("4. 识别高变基因...")
    sc.pp.highly_variable_genes(adata_processed, n_top_genes=2000)
    adata_processed.raw = adata_processed  # 保存原始数据
    
    # 使用高变基因
    adata_processed = adata_processed[:, adata_processed.var.highly_variable]
    
    # 5. 缩放和PCA
    print("5. 缩放数据和PCA...")
    sc.pp.scale(adata_processed, max_value=10)
    sc.tl.pca(adata_processed, svd_solver='arpack')
    
    print("数据预处理完成!")
    return adata_processed

def compute_comprehensive_metrics_simple(adata, batch_key='batch', label_key=None, use_cluster_as_label=True):
    """
    使用简化的评估指标计算批次效应校正效果
    """
    metrics = {}
    
    # 确保有邻居图
    if 'neighbors' not in adata.uns:
        print("计算邻居图...")
        sc.pp.neighbors(adata)
    
    # 如果没有提供标签键，使用聚类结果作为替代
    if label_key is None and use_cluster_as_label:
        # 确保有聚类结果
        if 'leiden' not in adata.obs.columns:
            print("计算Leiden聚类...")
            sc.tl.leiden(adata, resolution=0.5)
        label_key = 'leiden'
        print(f"使用聚类结果作为标签: {label_key}")
    
    # 使用PCA进行计算
    if 'X_pca' in adata.obsm:
        if adata.obsm['X_pca'].shape[1] > 20:
            X_emb = adata.obsm['X_pca'][:, :20]
        else:
            X_emb = adata.obsm['X_pca']
    else:
        # 如果没有PCA，使用scVI嵌入
        if 'X_scVI' in adata.obsm:
            X_emb = adata.obsm['X_scVI']
        else:
            print("警告: 没有找到PCA或scVI嵌入，使用原始数据")
            X_emb = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # 1. 批次轮廓宽度 (ASW batch) - 越小越好
    try:
        if len(adata.obs[batch_key].unique()) > 1:
            batch_silhouette = silhouette_score(X_emb, adata.obs[batch_key])
            metrics['batch_silhouette'] = batch_silhouette
        else:
            metrics['batch_silhouette'] = np.nan
    except Exception as e:
        print(f"批次轮廓宽度计算失败: {e}")
        metrics['batch_silhouette'] = np.nan
    
    # 2. 批次混合的局部逆辛普森指数 (ILISI) - 越大越好
    try:
        # 计算k近邻
        n_neighbors = min(50, X_emb.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_emb)
        distances, indices = nbrs.kneighbors(X_emb)
        
        # 计算每个细胞的LISI分数
        lisi_scores = []
        for i in range(X_emb.shape[0]):
            neighbor_batches = adata.obs[batch_key].iloc[indices[i]].values
            unique, counts = np.unique(neighbor_batches, return_counts=True)
            proportions = counts / len(neighbor_batches)
            simpson = np.sum(proportions ** 2)
            lisi = 1 / simpson if simpson > 0 else 0
            lisi_scores.append(lisi)
        
        metrics['ilisi'] = np.mean(lisi_scores)
    except Exception as e:
        print(f"ILISI计算失败: {e}")
        metrics['ilisi'] = np.nan
    
    # 3. 主成分回归批次方差 (PCR) - 越小越好
    try:
        # 使用前10个PC
        if 'X_pca' in adata.obsm:
            X_pca = adata.obsm['X_pca'][:, :10]
            
            # 编码批次信息
            le = LabelEncoder()
            batch_encoded = le.fit_transform(adata.obs[batch_key])
            
            # 计算每个PC与批次的相关性
            r2_scores = []
            for i in range(X_pca.shape[1]):
                model = LinearRegression()
                model.fit(X_pca[:, i].reshape(-1, 1), batch_encoded)
                y_pred = model.predict(X_pca[:, i].reshape(-1, 1))
                r2 = 1 - np.sum((batch_encoded - y_pred) ** 2) / np.sum((batch_encoded - np.mean(batch_encoded)) ** 2)
                r2_scores.append(r2)
            
            metrics['pcr_batch'] = np.mean(r2_scores)
        else:
            metrics['pcr_batch'] = np.nan
    except Exception as e:
        print(f"PCR计算失败: {e}")
        metrics['pcr_batch'] = np.nan
    
    # 4. 图连通性 (Graph Connectivity) - 越大越好
    try:
        connectivity_scores = []
        for cluster in adata.obs[label_key].unique():
            cluster_mask = adata.obs[label_key] == cluster
            if cluster_mask.sum() > 1:  # 至少有两个细胞
                # 获取该聚类的子图
                sub_adj = adata.obsp['connectivities'][cluster_mask, :][:, cluster_mask]
                # 计算连通分量
                n_components, _ = connected_components(sub_adj)
                # 连通性 = 1 - (连通分量数 - 1) / (细胞数 - 1)
                connectivity = 1.0 - (n_components - 1) / (cluster_mask.sum() - 1)
                connectivity_scores.append(connectivity)
        
        metrics['graph_connectivity'] = np.mean(connectivity_scores) if connectivity_scores else 0
    except Exception as e:
        print(f"图连通性计算失败: {e}")
        metrics['graph_connectivity'] = np.nan
    
    # 5. k近邻批次效应检验 (KBET) - 越小越好
    try:
        # 简化的kBET实现
        # 计算每个细胞的k近邻中不同批次的比例
        n_neighbors = min(30, X_emb.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_emb)
        distances, indices = nbrs.kneighbors(X_emb)
        
        batch_rejection_rates = []
        for i in range(X_emb.shape[0]):
            neighbor_batches = adata.obs[batch_key].iloc[indices[i]].values
            # 计算当前细胞的批次
            cell_batch = adata.obs[batch_key].iloc[i]
            # 计算邻居中相同批次的比例
            same_batch_ratio = np.sum(neighbor_batches == cell_batch) / len(neighbor_batches)
            # 如果比例低于阈值，则拒绝
            threshold = 1 / len(adata.obs[batch_key].unique())
            if same_batch_ratio < threshold:
                batch_rejection_rates.append(1)
            else:
                batch_rejection_rates.append(0)
        
        metrics['kbet_acceptance_rate'] = 1 - np.mean(batch_rejection_rates) if batch_rejection_rates else 0
    except Exception as e:
        print(f"KBET计算失败: {e}")
        metrics['kbet_acceptance_rate'] = np.nan
    
    # 6. 细胞类型轮廓宽度 (ASW Label) - 越大越好
    try:
        if label_key is not None and len(adata.obs[label_key].unique()) > 1:
            label_silhouette = silhouette_score(X_emb, adata.obs[label_key])
            metrics['cell_type_silhouette'] = label_silhouette
        else:
            metrics['cell_type_silhouette'] = np.nan
    except Exception as e:
        print(f"细胞类型轮廓宽度计算失败: {e}")
        metrics['cell_type_silhouette'] = np.nan
    
    # 7. 细胞类型局部逆辛普森指数 (cLISI) - 越大越好
    try:
        if label_key is not None:
            # 计算k近邻
            n_neighbors = min(50, X_emb.shape[0] - 1)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_emb)
            distances, indices = nbrs.kneighbors(X_emb)
            
            # 计算每个细胞的cLISI分数
            clisi_scores = []
            for i in range(X_emb.shape[0]):
                neighbor_labels = adata.obs[label_key].iloc[indices[i]].values
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                proportions = counts / len(neighbor_labels)
                simpson = np.sum(proportions ** 2)
                clisi = 1 / simpson if simpson > 0 else 0
                clisi_scores.append(clisi)
            
            metrics['clisi'] = np.mean(clisi_scores)
        else:
            metrics['clisi'] = np.nan
    except Exception as e:
        print(f"cLISI计算失败: {e}")
        metrics['clisi'] = np.nan
    
    return metrics

def prepare_data_for_scvi(adata):
    """为scVI准备数据"""
    print("\n为scVI准备数据...")
    
    # 创建用于scVI的数据副本
    adata_scvi = adata.copy()
    
    # 确保有原始计数数据
    if 'counts' in adata_scvi.layers:
        print("使用layers['counts']作为原始计数")
        adata_scvi.X = adata_scvi.layers['counts'].copy()
    elif adata_scvi.raw is not None:
        print("使用adata.raw作为原始计数")
        adata_scvi = adata_scvi.raw.to_adata()
    else:
        print("警告: 未找到原始计数数据，使用当前adata.X")
        # 如果数据已经标准化，需要反标准化
        if np.max(adata_scvi.X) < 50:  # 假设标准化后的数据值较小
            print("数据似乎已标准化，尝试反标准化...")
            # 这是一个简化的反标准化，实际应用中可能需要更复杂的方法
            if hasattr(adata_scvi.X, 'toarray'):
                adata_scvi.X = np.expm1(adata_scvi.X.toarray())
            else:
                adata_scvi.X = np.expm1(adata_scvi.X)
    
    # 检查数据是否为整数（计数数据）- 修复稀疏矩阵比较问题
    print("检查数据类型...")
    try:
        # 对于稀疏矩阵，先转换为密集矩阵再检查
        if hasattr(adata_scvi.X, 'toarray'):
            X_dense = adata_scvi.X.toarray()
            is_integer = np.allclose(X_dense, np.round(X_dense))
        else:
            is_integer = np.allclose(adata_scvi.X, np.round(adata_scvi.X))
        
        if not is_integer:
            print("警告: 数据可能不是整数计数，scVI需要原始计数数据")
            print("将数据转换为整数...")
            if hasattr(adata_scvi.X, 'toarray'):
                adata_scvi.X = adata_scvi.X.astype(np.int32)
            else:
                adata_scvi.X = np.round(adata_scvi.X).astype(np.int32)
        else:
            print("数据已经是整数格式")
            
    except Exception as e:
        print(f"数据类型检查失败: {e}")
        print("强制转换为整数...")
        if hasattr(adata_scvi.X, 'toarray'):
            adata_scvi.X = adata_scvi.X.astype(np.int32)
        else:
            adata_scvi.X = np.round(adata_scvi.X).astype(np.int32)
    
    # 确保有批次信息
    if 'batch' not in adata_scvi.obs.columns:
        if 'method' in adata_scvi.obs.columns:
            adata_scvi.obs['batch'] = adata_scvi.obs['method']
        else:
            raise ValueError("数据中未找到批次信息")
    
    print(f"批次信息: {adata_scvi.obs['batch'].nunique()} 个批次")
    print(f"批次分布:\n{adata_scvi.obs['batch'].value_counts()}")
    
    return adata_scvi

def run_scvi_correction(adata, batch_key='batch', n_latent=30, max_epochs=400):
    """运行scVI批次校正"""
    if not SCVI_AVAILABLE:
        raise ImportError("scVI 不可用，请先安装: pip install scvi-tools")
    
    print(f"\n正在运行scVI批次校正...")
    print(f"参数: n_latent={n_latent}, max_epochs={max_epochs}")
    
    # 创建数据副本
    adata_corrected = adata.copy()
    
    try:
        # 设置scVI数据
        scvi.model.SCVI.setup_anndata(adata_corrected, batch_key=batch_key)
        
        # 创建模型
        model = scvi.model.SCVI(adata_corrected, n_latent=n_latent)
        
        # 训练模型
        print("开始训练scVI模型...")
        model.train(max_epochs=max_epochs)
        
        # 获取潜在表示
        latent = model.get_latent_representation()
        adata_corrected.obsm["X_scVI"] = latent
        
        # 使用scVI潜在表示进行后续分析
        sc.pp.neighbors(adata_corrected, use_rep="X_scVI")
        sc.tl.umap(adata_corrected)
        sc.tl.leiden(adata_corrected, resolution=0.5)
        
        print("scVI批次校正完成!")
        return adata_corrected, model
        
    except Exception as e:
        print(f"scVI运行失败: {e}")
        raise

def visualize_results(adata_original, adata_corrected, output_dir):
    """可视化校正结果 - 改进版本，自动处理UMAP坐标"""
    print("\n生成可视化结果...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查并确保UMAP坐标存在
    def ensure_umap(adata, method_name):
        """确保数据有UMAP坐标，如果没有则计算"""
        umap_found = False
        possible_keys = ['X_umap', 'umap', 'UMAP']
        
        for key in possible_keys:
            if key in adata.obsm.keys():
                print(f"{method_name}: Found UMAP in .obsm['{key}']")
                umap_found = True
                # 确保使用标准的键名
                if key != 'X_umap':
                    adata.obsm['X_umap'] = adata.obsm[key]
                break
        
        if not umap_found:
            print(f"{method_name}: No UMAP coordinates found. Computing UMAP...")
            try:
                # 确保有邻居图，如果没有则计算
                if 'neighbors' not in adata.uns:
                    print(f"{method_name}: Computing neighbors for UMAP...")
                    sc.pp.neighbors(adata, random_state=42)
                
                sc.tl.umap(adata, random_state=42)
                print(f"{method_name}: UMAP computed successfully")
            except Exception as e:
                print(f"{method_name}: Failed to compute UMAP - {e}")
                return False
        
        return True
    
    # 确保两个数据集都有UMAP坐标
    orig_umap_ok = ensure_umap(adata_original, "Original Data")
    corr_umap_ok = ensure_umap(adata_corrected, "Corrected Data")
    
    if not orig_umap_ok or not corr_umap_ok:
        print("警告: 无法为某些数据集计算UMAP坐标，可视化可能不完整")
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始数据 - 批次
    if orig_umap_ok:
        sc.pl.umap(adata_original, color=['batch'], ax=axes[0, 0], show=False, 
                   title='Original - Batch', frameon=False)
    else:
        axes[0, 0].text(0.5, 0.5, 'No UMAP\navailable', ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Original - Batch (No UMAP)')
    
    # 原始数据 - 聚类
    if orig_umap_ok and 'leiden' in adata_original.obs.columns:
        sc.pl.umap(adata_original, color=['leiden'], ax=axes[1, 0], show=False, 
                   title='Original - Clusters', frameon=False)
    elif orig_umap_ok:
        # 如果没有leiden聚类，尝试使用其他可用变量
        available_vars = [col for col in adata_original.obs.columns 
                         if col != 'batch' and pd.api.types.is_categorical_dtype(adata_original.obs[col])]
        if available_vars:
            sc.pl.umap(adata_original, color=[available_vars[0]], ax=axes[1, 0], show=False, 
                       title=f'Original - {available_vars[0]}', frameon=False)
        else:
            axes[1, 0].text(0.5, 0.5, 'No clustering\navailable', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Original - No Clusters')
    else:
        axes[1, 0].set_visible(False)
    
    # 校正后数据 - 批次
    if corr_umap_ok:
        sc.pl.umap(adata_corrected, color=['batch'], ax=axes[0, 1], show=False, 
                   title='scVI Corrected - Batch', frameon=False)
    else:
        axes[0, 1].text(0.5, 0.5, 'No UMAP\navailable', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('scVI Corrected - Batch (No UMAP)')
    
    # 校正后数据 - 聚类
    if corr_umap_ok and 'leiden' in adata_corrected.obs.columns:
        sc.pl.umap(adata_corrected, color=['leiden'], ax=axes[1, 1], show=False, 
                   title='scVI Corrected - Clusters', frameon=False)
    elif corr_umap_ok:
        # 如果没有leiden聚类，尝试使用其他可用变量
        available_vars = [col for col in adata_corrected.obs.columns 
                         if col != 'batch' and pd.api.types.is_categorical_dtype(adata_corrected.obs[col])]
        if available_vars:
            sc.pl.umap(adata_corrected, color=[available_vars[0]], ax=axes[1, 1], show=False, 
                       title=f'scVI Corrected - {available_vars[0]}', frameon=False)
        else:
            axes[1, 1].text(0.5, 0.5, 'No clustering\navailable', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('scVI Corrected - No Clusters')
    else:
        axes[1, 1].set_visible(False)
    
    # 批次分布对比
    batch_counts_orig = adata_original.obs['batch'].value_counts()
    batch_counts_corr = adata_corrected.obs['batch'].value_counts()
    
    axes[0, 2].bar(batch_counts_orig.index.astype(str), batch_counts_orig.values, alpha=0.7, label='Original')
    axes[0, 2].bar(batch_counts_corr.index.astype(str), batch_counts_corr.values, alpha=0.7, label='scVI')
    axes[0, 2].set_title('Batch Distribution')
    axes[0, 2].legend()
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 批次混合指标
    if 'batch_silhouette' in adata_original.uns or 'batch_silhouette' in adata_corrected.uns:
        orig_sil = adata_original.uns.get('batch_silhouette', np.nan)
        corr_sil = adata_corrected.uns.get('batch_silhouette', np.nan)
        
        methods = ['Original', 'scVI']
        values = [orig_sil, corr_sil]
        
        bars = axes[1, 2].bar(methods, values, color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[1, 2].set_title('Batch Silhouette Score (Lower is Better)')
        axes[1, 2].set_ylabel('Silhouette Score')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
    else:
        # 如果没有silhouette分数，显示其他统计信息
        n_cells_orig = adata_original.n_obs
        n_cells_corr = adata_corrected.n_obs
        n_genes_orig = adata_original.n_vars
        n_genes_corr = adata_corrected.n_vars
        
        info_text = f"Original:\n{n_cells_orig} cells\n{n_genes_orig} genes\n\nCorrected:\n{n_cells_corr} cells\n{n_genes_corr} genes"
        axes[1, 2].text(0.5, 0.5, info_text, ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].set_title('Dataset Information')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scvi_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 单独保存校正后的UMAP图（如果可用）
    if corr_umap_ok:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 校正后数据 - 批次
        sc.pl.umap(adata_corrected, color=['batch'], ax=axes[0], show=False, 
                   title='scVI Corrected - Batch', frameon=False)
        
        # 校正后数据 - 聚类或其他变量
        if 'leiden' in adata_corrected.obs.columns:
            sc.pl.umap(adata_corrected, color=['leiden'], ax=axes[1], show=False, 
                       title='scVI Corrected - Clusters', frameon=False)
        else:
            # 使用其他可用变量
            available_vars = [col for col in adata_corrected.obs.columns 
                             if col != 'batch' and pd.api.types.is_categorical_dtype(adata_corrected.obs[col])]
            if available_vars:
                sc.pl.umap(adata_corrected, color=[available_vars[0]], ax=axes[1], show=False, 
                           title=f'scVI Corrected - {available_vars[0]}', frameon=False)
            else:
                # 如果没有分类变量，显示样本数
                sc.pl.umap(adata_corrected, color=None, ax=axes[1], show=False, 
                           title='scVI Corrected - UMAP', frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scvi_corrected.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("无法生成校正后的UMAP图 - UMAP坐标不可用")

def save_results(adata_corrected, model, output_dir, input_file):
    """保存结果"""
    print("\n保存结果...")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = "GSE132044"
    output_file = f"{output_dir}/{base_name}_scvi_corrected_{timestamp}.h5ad"
    
    # 保存校正后的数据
    adata_corrected.write(output_file, compression='gzip')
    print(f"校正后的数据已保存: {output_file}")
    
    # 保存模型（如果可能）
    try:
        model_dir = f"{output_dir}/scvi_model_{timestamp}"
        model.save(model_dir)
        print(f"scVI模型已保存: {model_dir}")
    except Exception as e:
        print(f"无法保存模型: {e}")
    
    return output_file

def create_metrics_comparison(original_metrics, corrected_metrics, output_dir):
    """创建指标对比可视化"""
    print("\n创建指标对比图...")
    
    # 筛选出数值型指标
    numeric_metrics = {}
    for key in original_metrics:
        if not pd.isna(original_metrics[key]) and not pd.isna(corrected_metrics.get(key, np.nan)):
            numeric_metrics[key] = {
                'Original': original_metrics[key],
                'scVI': corrected_metrics[key]
            }
    
    if not numeric_metrics:
        print("没有可用的数值指标进行对比")
        return
    
    # 创建指标对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左侧：所有指标的条形图
    metrics_names = list(numeric_metrics.keys())
    original_values = [numeric_metrics[m]['Original'] for m in metrics_names]
    scvi_values = [numeric_metrics[m]['scVI'] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, original_values, width, label='Original', alpha=0.7)
    axes[0].bar(x + width/2, scvi_values, width, label='scVI', alpha=0.7)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Batch Correction Metrics Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, rotation=45, ha='right')
    axes[0].legend()
    
    # 右侧：关键指标的改进雷达图
    key_metrics = ['batch_silhouette', 'cell_type_silhouette', 'graph_connectivity', 'kbet_acceptance_rate']
    available_key_metrics = [m for m in key_metrics if m in numeric_metrics]
    
    if len(available_key_metrics) >= 3:
        # 计算改进百分比
        improvements = {}
        for metric in available_key_metrics:
            orig_val = numeric_metrics[metric]['Original']
            scvi_val = numeric_metrics[metric]['scVI']
            
            # 根据指标类型计算改进
            if metric in ['batch_silhouette']:  # 越小越好
                if orig_val != 0:
                    improvement = (orig_val - scvi_val) / abs(orig_val) * 100
                else:
                    improvement = 0
            else:  # 越大越好
                if orig_val != 0:
                    improvement = (scvi_val - orig_val) / abs(orig_val) * 100
                else:
                    improvement = 0
            
            improvements[metric] = improvement
        
        # 准备雷达图数据
        categories = list(improvements.keys())
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 准备值
        values = [improvements[m] for m in categories]
        values += values[:1]  # 闭合图形
        
        # 绘制雷达图
        ax = axes[1]
        ax.plot(angles, values, 'o-', linewidth=2, label='Improvement (%)')
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
        ax.set_title('Key Metrics Improvement (%)')
        ax.grid(True)
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 5, f'{value:.1f}%', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='scVI批次校正程序 - 完整流程')
    parser.add_argument('--output_dir', '-o', default='scvi_results', 
                       help='输出目录')
    parser.add_argument('--n_latent', type=int, default=30, 
                       help='潜在空间维度')
    parser.add_argument('--max_epochs', type=int, default=400, 
                       help='最大训练轮数')
    parser.add_argument('--batch_key', default='batch', 
                       help='批次信息所在的列名')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("scVI批次校正程序 - 完整流程")
    print("=" * 60)
    
    if not SCVI_AVAILABLE:
        print("错误: scVI 不可用，请先安装: pip install scvi-tools")
        return
    
    try:
        # 1. 加载原始数据
        print("步骤 1/6: 加载原始数据")
        adata_raw = load_gse132044_data_correct()
        
        # 2. 数据预处理（用于指标计算和可视化）
        print("\n步骤 2/6: 数据预处理")
        adata_processed = preprocess_adata(adata_raw)
        
        # 3. 为scVI准备数据（使用原始计数数据）
        print("\n步骤 3/6: 为scVI准备数据")
        # 注意：这里使用原始数据，不是预处理后的数据
        # scVI需要原始计数数据
        adata_scvi = adata_raw.copy()  # 使用原始数据的副本
        
        # 确保有批次信息
        if 'batch' not in adata_scvi.obs.columns:
            if 'method' in adata_scvi.obs.columns:
                adata_scvi.obs['batch'] = adata_scvi.obs['method']
        
        # 过滤低质量细胞和基因（与预处理相同的过滤标准）
        print("为scVI过滤数据...")
        # 计算QC指标
        gene_symbols = adata_scvi.var_names.str.split('_').str[1]
        adata_scvi.var['mt'] = gene_symbols.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(
            adata_scvi, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        # 过滤
        sc.pp.filter_genes(adata_scvi, min_cells=3)
        n_genes_threshold = 200
        mt_threshold = 20
        cell_filter = (adata_scvi.obs['n_genes_by_counts'] >= n_genes_threshold) & \
                      (adata_scvi.obs['pct_counts_mt'] <= mt_threshold)
        adata_scvi = adata_scvi[cell_filter, :]
        
        print(f"scVI数据形状: {adata_scvi.shape}")
        
        # 4. 计算原始数据的批次指标（使用预处理后的数据）
        print("\n步骤 4/6: 计算原始数据指标")
        original_metrics = compute_comprehensive_metrics_simple(adata_processed.copy(), args.batch_key)
        adata_processed.uns['batch_metrics'] = original_metrics
        
        print("原始数据指标:")
        for k, v in original_metrics.items():
            if not pd.isna(v):
                print(f"  {k}: {v:.4f}")
        
        # 5. 运行scVI校正
        print("\n步骤 5/6: 运行scVI批次校正")
        adata_corrected, model = run_scvi_correction(
            adata_scvi,  # 使用为scVI准备的数据
            batch_key=args.batch_key,
            n_latent=args.n_latent,
            max_epochs=args.max_epochs
        )
        
        # 6. 计算校正后的批次指标
        print("\n步骤 6/6: 计算校正后数据指标")
        corrected_metrics = compute_comprehensive_metrics_simple(adata_corrected.copy(), args.batch_key)
        adata_corrected.uns['batch_metrics'] = corrected_metrics
        
        print("scVI校正后指标:")
        for k, v in corrected_metrics.items():
            if not pd.isna(v):
                print(f"  {k}: {v:.4f}")

        # 9. 保存结果
        output_file = save_results(adata_corrected, model, args.output_dir, "GSE132044")   

        # 7. 可视化结果
        visualize_results(adata_processed, adata_corrected, args.output_dir)
        
        # 8. 创建指标对比
        create_metrics_comparison(original_metrics, corrected_metrics, args.output_dir)
        
        # 10. 打印总结
        print("\n" + "=" * 60)
        print("scVI批次校正完成!")
        print("=" * 60)
        print(f"原始数据批次轮廓系数: {original_metrics.get('batch_silhouette', 'N/A')}")
        print(f"校正后批次轮廓系数: {corrected_metrics.get('batch_silhouette', 'N/A')}")
        
        if (not pd.isna(original_metrics.get('batch_silhouette')) and 
            not pd.isna(corrected_metrics.get('batch_silhouette'))):
            improvement = original_metrics['batch_silhouette'] - corrected_metrics['batch_silhouette']
            print(f"改善程度: {improvement:.4f} (正值表示批次效应减小)")
        
        print(f"\n结果文件:")
        print(f"  校正后的数据: {output_file}")
        print(f"  可视化结果: {args.output_dir}/")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()