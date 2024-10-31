# MSC-MVC

- 实验数据

  - NUSWIDEOBJ

  - | method                                                       |  ACC  |  NMI  |  PUR  |
    | ------------------------------------------------------------ | :---: | :---: | :---: |
    | ICLR causal + kmeans                                         | 16.02 | 6.25  | 19.38 |
    | MFLVC                                                        | 18.28 | 15.12 | 23.93 |
    | Dealmvc                                                      | 14.82 | 6.92  | 16.10 |
    | ours                                                         | 22.14 | 17.03 | 28.30 |
    | Ours（batchsize=256，lr=0.0003，epoch=364）                  | 23.03 | 13.02 | 24.72 |
    | Ours（batchsize=512，lr=0.0003，epoch=375）                  | 21.29 | 11.62 | 22.91 |
    | Ours(batchsize=256，lr=0.0003，1epoch=300,2epoch=55,3epoch=150) | 22.94 | 13.87 | 25.47 |
    | --batch_size 256 --learning_rate 0.002 --mse_epochs 300 --con_epochs 100 --tune_epochs 150 | 11.91 |   0   | 11.91 |
    | --batch_size 256 --learning_rate 0.001 --mse_epochs 300 --con_epochs 100 --tune_epochs 150 | 12.32 | 1.84  | 12.32 |
    | --batch_size 256 --learning_rate 0.0003 --mse_epochs 300 --con_epochs 100 --tune_epochs 150 | 22.38 | 14.06 | 25.02 |
    | --batch_size 256 --learning_rate 0.0001 --mse_epochs 300 --con_epochs 55 --tune_epochs 23 | 20.35 | 12.06 | 22.10 |
    | --batch_size 256 --learning_rate 0.002 --mse_epochs 300 --con_epochs 55 --tune_epochs 150 | 11.92 |   0   | 11.92 |
    | --batch_size 256 --learning_rate 0.001 --mse_epochs 300 --con_epochs 55 --tune_epochs 7 | 22.14 | 17.03 | 28.30 |
    
    ```bash
    CUDA_VISIBLE_DEVICES=0 python train6_NUSWIDEOBJ_sota.py --batch_size 256 --learning_rate 0.002 --mse_epochs 300 --con_epochs 55 --tune_epochs 150
    ```


  - CCV

  - | method | ACC  | NMI  | PUR  |
    | ------ | :--: | :--: | :--: |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |
    |        |      |      |      |

    ```bash
    CUDA_VISIBLE_DEVICES=1 python train6_CCV_sota.py --batch_size 256 --learning_rate 0.001 --mse_epochs 300 --con_epochs 55 --tune_epochs 150
    ```



- 目前调研到的一些文章，AAAI 2024正好有两篇因果的图对比学习的文章：
  - Graph Contrastive Invariant Learning from the Causal Perspective
    - [shichuan.org/doc/169.pdf#page=1.29](http://www.shichuan.org/doc/169.pdf#page=1.29)
  - Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective
    - [Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective.pdf](file:///E:/论文/2024年10月/Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective.pdf)
  - Graph Invariant Learning with Subgraph Co-mixup for Out-of-Distribution Generalization
    - 这一篇还未细看，虽然没有写因果相关的字眼，但是粗略的瞄了几眼公式，感觉思路差不多类似
    - [shichuan.org/doc/168.pdf](http://www.shichuan.org/doc/168.pdf)



- 修改
  - abstract
    - 改动前
      - However, most current methods are limited to capturing semantic features and constructing pseudo-labels to capture clustering information, while neglecting the differences among different views and the further extraction of common semantic features.
    - However，most current methods会因view-dependency过多侧重其中部分视角的语义特征而而产生过度依赖，从而忽略了其余视角的有用信息，进而影响聚类的质量。
    - However, most current methods tend to focus excessively on the semantic features of certain views due to high view-dependency, which results in over-reliance on these views while neglecting valuable information from other views, ultimately affecting overall clustering quality.
  - introduction
    - 改动前
      - Among these methods, particularly the deep methods, satisfactory results have been achieved on many datasets by calculating similarity graphs and introducing contrastive learning to align latent representations. However, these methods do not fully capture more essential features, leading to significant performance degradation on some challenging datasets compared to other methods. For instance, \cite{DealMVC} combines contrastive learning and attention mechanisms to fuse features from different views, aiming to pull close different samples under various views to obtain more concentrated feature representations. This approach achieves good results on datasets with small view differences but struggles to produce satisfactory results on datasets with significant view differences. This issue arises because the intrinsic content and differences of features from different views have not been adequately explored, thereby weakening the model's generalization capability.
    - Among these methods, particularly the deep learning approaches, satisfactory results have been achieved on many datasets by calculating similarity graphs and introducing contrastive learning to align latent representations. However, these methods still face the issue of view dependency in multi-view clustering tasks, where the model may overly rely on the information from certain views while neglecting the features from the remaining views. This dependency prevents the model from fully leveraging the potential information across all views, leading to suboptimal performance, especially on datasets with significant view differences. For instance, \cite{DealMVC} combines contrastive learning and attention mechanisms to fuse features from different views, aiming to pull close the representations of samples across views and obtain more concentrated feature representations. While this approach performs well on datasets with small view differences, it struggles on datasets with larger view discrepancies. This is because the method fails to adequately handle the differences between views, causing the model to overly focus on one view and overlook the unique contributions from other views, ultimately weakening the overall generalization capability.
  - related work
    - 改动前
    - 改动后
      - In recent years, an increasing number of scholars have recognized the outstanding performance of deep methods \cite{SSGCNMVC,DFPGNN} in multi-view clustering (MVC). Given the heterogeneous nature of multi-view data, deep learning methods, with their powerful representation capabilities, can adaptively capture complex patterns in multi-view data by learning shared and view-specific features across different perspectives. Among these, autoencoders are widely used \cite{kipf2016variationalgraphautoencoders,Multimodal_deep_learning,Zhang_2019_CVPR} in graph clustering. 很多方法也利用contrasive learning的方法将不同视角下的特征进行对齐。然而由于这些方法大多从输入数据的粗粒度信息入手，对于如何让模型针对于shared and view-specific features更好地学习对于不同视角下的有效、本质、无依赖于其他视图的信息，成为当前的多视角任务下的一大难题。
      - In recent years, an increasing number of scholars have recognized the outstanding performance of deep methods \cite{SSGCNMVC,DFPGNN} in multi-view clustering (MVC). Given the heterogeneous nature of multi-view data, deep learning methods, with their powerful representation capabilities, can adaptively capture complex patterns in multi-view data by learning shared and view-specific features across different perspectives. Among these, autoencoders are widely used \cite{kipf2016variationalgraphautoencoders,Multimodal_deep_learning,Zhang_2019_CVPR} in graph clustering. Furthermore, many approaches leverage contrastive learning to align features from different perspectives. However, as most of these methods rely on coarse-grained information from the input data, the key challenge in multi-view tasks is enabling the model to effectively capture essential, independent, and view-specific information that is informative for each perspective without depending on other views.

# 第二篇Idea开荒

- GCFAgg、DealMVC、MFLVC使用的是一套框架

  - GCFAgg对于不同视角的特征cat后做transformer
  - DealMVC从global和local两个角度做图对比学习（纯暴力）
  - MFLVC伪标签监督

- 目前还不清楚能不能往GNN的一些idea靠一靠，目前找的一些论文：

  - 这两篇目前用的一套框架，同一个实验室做的：
    - Deep Graph Clustering via Dual Correlation Reduction
      - [论文阅读“Deep Graph Clustering via Dual Correlation Reduction”（AAAI2022）-CSDN博客](https://blog.csdn.net/qq_43497436/article/details/124397146)
      - [DCRN.pdf](file:///E:/论文/2024年10月/DCRN.pdf)
    - Mixed Graph Contrastive Network for Semi-Supervised Node Classification
      - 这篇是发在CIKM（CCF b）上的，用的是互信息的思想
      - https://arxiv.org/pdf/2206.02796
  
  

# GNN相关文章收集和阅读

- 重新读了Graph Contrasive Learning
  - [GraphCL.pdf](file:///C:/Users/12895/Desktop/GraphCL.pdf)
- Graph Transformer相关工作
  - Heterogeneous Graph Transformer with Poly-Tokenization
    - [shichuan.org/doc/179.pdf](http://www.shichuan.org/doc/179.pdf)
  - Less is More: on the Over-Globalizing Problem in Graph Transformers
    - [Less is More on the Over-Globalizing Problem in Graph Transformers.pdf](file:///E:/论文/2024年10月/Less is More on the Over-Globalizing Problem in Graph Transformers.pdf)



# MVC via Diff-Transformer idea开荒

- 收集
  - GCFAgg （CVPR 2023）
    - 然而，大多数现有的深度聚类方法通过视角聚合方式从多个视角学习共识表示或视角特定表示，但它们忽略了所有样本之间的结构关系。
    - However, most existing deep clustering methods learn consensus representation or view-specific
      representations from multiple views via view-wise aggregation way, where they ignore structure relationship of all samples.
    - 全局与跨视角特征聚合多视角聚类
  - Differentiable Information Bottleneck for Deterministic Multi-view Clustering（CVPR 2024）
    - ![image-20241021172626936](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241021172626936.png)
  - Integrating Vision-Language Semantic Graphs in Multi-View Clustering
    - 