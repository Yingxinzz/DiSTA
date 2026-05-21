#DiSTA
这是一个用于空间转录组数据分析的 Python 项目，基于 STAIG 框架进行特征提取与融合
Yang, Y., Cui, Y., Zeng, X. et al. STAIG: Spatial transcriptomics analysis via image-aided graph contrastive learning for domain exploration and alignment-free integration. Nat Commun 16, 1067 (2025). https://doi.org/10.1038/s41467-025-56276-0

**本研究提出一种融合H&E染色形态学信息的空间转录组分析框架DiSTA。该框架首先引入自监督孪生网络策略对视觉大模型进行领域自适应微调以提取组织形态特征。单切片分析中，利用形态学信息引导基因表达的局部加权平滑。多切片分析中，依据切片间差异设计了整合策略：对于生物学重复切片，采用一致性联合嵌入策略，结合动态门控机制与参数共享网络实现跨切片一致推断；对于不同形变程度的连续切片，则在门控融合基础上引入互近邻分子锚点与图拓扑隔离策略。最后构建了评估生物学一致性与批次校正效果的多维度空间评价体系，对模型的整体性能进行系统评估

##preparation stage
0.prepare patches
1.train
2.extract feature

##单切片
3_single_slice.py
##多切片
3_integrate_2slices_staig.py
##多切片验证
3_integrate_slice_staig.py

##License
This project is covered under the Apache 2.0 License.
