https://openreview.net/forum?id=5XwylUAmnY&referrer=%5Bthe%20profile%20of%20Zhangqi%20Jiang%5D(%2Fprofile%3Fid%3D~Zhangqi_Jiang1)





- Hierarchical Prompt-Guided Multi-View Clustering
- (Multi-Level) Prompt-Guided Multi-View Clustering via/with Hierarchical Circuit Alignment

- Hierarchical Prompt-Guided Circuit Alignment for Multi-View Clustering

- HITORIE: **HI**erarchical Promp**T** Learning f**O**r Multi-View Clustering with Ci**R**cuit Al**I**gnm**E**nt

- HTRE: **H**ierarchical Promp**T**-Guided Ci**R**cuit Alignm**E**nt for Multi-View Clustering
- HitorieMVC

# Abstract

Multi-view clustering 通过explore 不同视角数据的语义信息以获得更好的表现性能。Recent works 常常通过融合不同视角信息和来考虑跨视角Heterogeneous Features.然而，这种侧重全局特征的对齐与分布匹配，对各维度特征所代表的语义信息缺乏精细的关注与控制。这种仅在隐空间层面进行粗粒度的对齐会导致对部分特征维度的过度依赖或局部最优。为了解决这个问题，我们提出了a new model 名为Hierarchical Prompt-Guided Multi-View Clustering(HiPMVC).为了对于语义信息进行细粒度的精细控制，我们在Multi-view Clustering任务中引入了prompt learning，从input-level和embedding level处理多视图Heterogeneous信息。而为了解决无监督对齐导致的局部最优问题，我们引入了hierarchical circuit alignment这个方法，实现实现更稳定、全局的分布一致性。





# Introduction

In recent years，在多媒体数据激增的背景下Multi-View Clustering 在无监督领域越来越备受关注。很多工作聚焦于如何挖掘多视角数据的共同语义信息，以获得更好的聚类效果。目前现有的方法大致可以分为两类，traditional multi-view clustering和 deep multi-view clustering.

与traditional methods相比，deep methods最大的优势在于能够有效提取输入数据的有效表征。目前现有的deep methods中，对于表征的提取可以大致分为distinct MLP-based methods和GNN-based methods，前者常利用VAE获得有效表征，后者通过对于输入数据利用kNN建图后利用GCN架构提取特征。对于不同视角下的特征对齐，contrasive learning成为在MVC这个任务上广受欢迎的方法。例如，dealmvc通过构建对比损失约束不同样本对于with similar features的一致性。GCFAgg通过structure-guided contrasive learning 来引导聚类结果。此外，很多deep methods通过将不同视角下的特征进行融合的方式获得更好的统一表示。

尽管这些方法在MVC任务中已经取得了显著的improvement，但是，对于一些较难的数据集上，现有方法对各维度特征所代表的语义信息缺乏精细的关注与控制。很多deep methods常常通过融合不同视角信息和来考虑跨视角Heterogeneous Features.并且仅在隐空间层面基于对比学习进行粗粒度的对齐会导致对部分特征维度的相似性产生过度依赖，导致出现局部最优的情况。

为了解决这些问题，我们创新性地提出了a new model 名为Hierarchical Prompt-Guided Circuit Alignment for Multi-View Clustering.我们引入了提示学习的思想，对于输入数据的instance-level和embedding-level的维度特征进行细粒度的控制，对于多视图的Heterogeneous信息的有效聚类信息进行特征层面的提示。为了解决基于对比学习进行粗粒度的对齐会导致对部分特征维度的相似性产生过度依赖这个问题，我们创新性地提出了hierarchical circuit alignment策略，从训练得到的不同层级的伪标签信息进行forward alignment和backward feedback。并且我们的实验证明，对于对于输入数据不同的encoder如distinct MLP encoder 或者GCN encoder，我们的方法都获得了普遍优越性。

我们的贡献点如下：

- 我们提出了Hierarchical Prompt-Guided Circuit Alignment network for Multi-View Clustering。我们的方法利用提示的方式对于输入数据的聚类语义信息进行细粒度掌控，并且实现更稳定、全局的分布一致性。
- 我们分别从instance-level和embedding-level对于输入数据进行特征维度的提示设计。我们设计了通过旁路提示网络训练获得相应的提示，在embedding-level分别从local和global的角度综合获得提示信息。
- 我们提供了一种hierarchical circuit alignment策略。对于不同层次的伪标签信息，我们通过forward alignment和backward feedback进行反馈式动态调节，从而更好对齐各视角的伪标签信息。





Our contributions are summarized as follows:



1.We propose the Hierarchical Prompt-Guided Circuit Alignment network for multi-view clustering. This method leverages prompts to achieve fine-grained control over the clustering semantics of input data, ensuring stable and globally consistent distributions.

这个创新点是属于方法中的东西，不够直接了当，落脚点落小了，需要说更大的优点，请往特征学习、聚类方面靠

2.We design prompt mechanisms at both the primary-level and embedding-level for input features. These mechanisms employ an auxiliary prompt network to train and generate the corresponding prompts. At the embedding level, prompts are derived by integrating local and global perspectives.

这个创新点要讲好处，能够提取细粒度信息，能够有什么好处

3. We develop a hierarchical circuit alignment strategy. This strategy dynamically adjusts pseudo-label information across different levels through forward alignment and backward feedback, effectively aligning pseudo-labels across multiple views.

这个创新点要侧重讲解这么做能够解决聚类中的什么问题，现有的什么不足的地方解决了











# Related Work

目前已有方法中



- prompt learning

- prompt learning 最初发源自NLP任务，旨在输入文本中添加一组可学习的提示参数，使得模型能够在下游任务中适应性微调。在最近几年的研究中，很多工作将提示学习的思想迁移至视觉任务中，旨在用较少的参数量达到类似于文本提示的下游任务适应性效果。Visual prompt旨在通过给输入的图片的周围添加可学习的提示参数后进行预训练模型中训练。Visual Prompt Tuning 通过在ViT上每层backbone前添加token级别的prompt来微调模型。
  而对于Multi-view clustering任务中，我们引入prompt learning的思想，从instance-level和embedding-level同时添加prompt，用于适应性地对于输入数据的有效语义信息进行细粒度的控制。

  





# Methodology

In this section，我们详细展开讲述our model HiPMVC的方法细节。如图1所示，对于输入数据我们从instance-level添加prompt。经过encoder层得到的特征之后添加view-specific embedding prompt和global prompt。其中处理后的特征和global prompt经过mapping encoder之后映射到统一的特征空间。最后通过伪标签分类器产生的多层级标签进行Hierarchical circuit alignment。

- instance-level  prompt learning

为了有效地初步提取输入数据的语义特征，我们对于每一个视角的每一个输入样本，相应地添加可学习的提示。

对于多视角输入数据$\{ \mathbf{X}^v \in \mathbb{R}^{N \times D_v} \}^{V}_{v=1}$, where $N$ denotes the number of samples, $V$ represents the number of views, and $D_v$ is the dimensionality of the $v$-th view，我们设计了对于每个视角的原始输入的prompt encoder的MLP网络$\mathcal{M}_{Local}(\cdot)$，用于提取视图的特征的一种提示信息$\mathbf{P}^{v}_{L}$，表示为：

$$\mathbf{P}^{v}_{L} = \mathcal{M}_{Instance}(\mathbf{X}^{v})$$

where $\mathbf{P}^{v}_{L} \in \mathbb{R}^{N \times D_v}$,在样本数量和维度上和原始输入$\mathbf{X}^v$保持一致，表明一种基于原始数据的一种提示性操作。而为了提取有效语义信息，我们将$\mathbf{X}^v$和$\mathbf{P}^{v}_{L}$一同输入至特征embedding网络，获得preliminary feature $\mathbf{Z}^{v}$，其表示为：

$$\mathbf{Z}^{v} = E_{shallow}(\mathbf{X}^{v}+\mathbf{P}^{v}_{L})$$

，where $\mathbf{Z}^{v}_{L} \in \mathbb{R}^{N \times D}$.为了验证提示操作的普遍性作用，我们对于$E_{shallow}(\cdot)$有两种设计：distinct MLP encoder和GCN encoder，其效果我们会在experiment section中阐述。





- embedding-level prompt learning

目前大多现有的方法中对于提取的特征的处理，大致分为对于单视角的处理和cross-view的多视角融合处理。而仅用类似于注意力机制的融合，由于对于模型规模的限制和无监督任务的特性，在共同语义提取上缺乏细粒度的信号的引导。因此，我们的方法提出了view-specific local prompts $\mathbf{P}^{v}_{L}$和global prompts$\mathbf{P}_{G}$。其中,每个视角下的$\mathbf{P}^{v}_{L}$用于捕捉关键性的embedding-level的聚类信息。$\mathbf{P}_{G}$用于捕捉不同实例间的共同的有效聚类信息。

类似于instance-level prompt learning对于prompt提取的设计，我们分别对于view-specific local prompts$\mathbf{P}^{v}_{L}$和global prompts$\mathbf{P}_{G}$设计相应的轻量级网络$\mathcal{M}_{Local}(\cdot)$和$\mathcal{M}_{Global}(\cdot)$。对于$\mathbf{P}^v_{L}$的提取如下所示：

$$\mathbf{P}^{v}_{L} = \mathcal{M}^{v}_{Local}(\mathbf{X}^{v})$$

where $\mathbf{P}^{v}_{L} \in \mathbb{R}^{N \times D}$。而对于为了获得global prompts$\mathbf{P}_{G}$，我们需要考虑视角之间的融合，因此我们引入了一组可学习的适应性参数weight parameters $\{w^{1}_{\alpha}, w^{2}_{\alpha}, ..., w^{v}_{\alpha} \}, v \in V$, 来动态调整融合信息。其表示如下：

$$\mathbf{P}_{G} = \mathcal{M}_{Global}(\sum ^{V}_{v=1}\dfrac{e^{w^{v}_{\alpha}}}{\sum ^{V}_{k=1}e^{w^{k}_{\alpha}}}\mathbf{X}^v)$$

where $\mathbf{P}^{v}_{L} \in \mathbb{R}^{N \times D}$。构建好了local和global prompts，我们的方法的下一步是将preliminary feature
$\mathbf{Z}^v$与提示一同输入至更高level的聚类语义空间中：

$$\mathbf{H}^{v} = E^{v}_{Mapping}(\mathbf{Z}^{v}+\lambda \mathbf{P}_{G} + (1-\lambda)\mathbf{P}^{v}_{L} )$$

where $\lambda$是超参数。此外，inspired by DealMVC，我们利用基于注意力机制对于各视角的$\mathbf{H}^{v}$进行融合，得到unified feature$\mathbf{P}_{\mathbf{U}}$:

$$\mathbf{W}^v= \texttt{Attn}(\mathbf{H}^{v})\odot \texttt{FFN}(\mathbf{H}^{v})$$

$$\mathbf{H}_{\mathbf{U}}= \sum^{V}_{v=1} \mathbf{H}^v \mathbf{W}^v$$

Where $\mathbf{W}^v \in \mathbb{R}^{D \times D},\mathbf{H}_{\mathbf{U}} \in \mathbb{R}^{N \times D}$,$\texttt{Attn}(\cdot)$为注意力机制网络，$\texttt{FFN}(\cdot)$为非线性MLP网络。

而对于global prompts，由于其本身具有一定的共同聚类语义信息，为了进一步细粒化潜在的聚类信息，我们同样利用与$E^{v}_{Mapping}(\cdot)$具有相同架构的$E^{P}_{Mapping}(\cdot)$，将global prompts一同映射至更高level的聚类语义空间中。其表示如下：

$$\mathbf{H}_{\mathbf{P}} = E^{P}_{Mapping}(\mathbf{P}_{G})$$

- 
- Hierarchical circuit alignment

考虑到多视角输入数据存在特征层面的分布差异，已经对于聚类结果层面各视角的聚类结果的差异，因此我们对于这两个层面都需要进行跨视角对齐。而为了获得聚类结果，我们使用近年来广为接受的pseudo label classifier的策略。对于每个higher-level的特征$\mathbf{H}^{v}$和经过$E^{v}_{Mapping}(\cdot)$映射至统一聚类语义空间中的$\mathbf{H}_{\mathbf{P}}$构建伪标签：

$$\mathbf{Y}^{v} = \mathcal{P}_{Feature}(\mathbf{H}^{v}), \mathbf{Y}_{\mathbf{U}} = \mathcal{P}_{Feature}(\mathbf{H}_{\mathbf{U}})$$

$$\mathbf{Y}_{\mathbf{P}} = \mathcal{P}^{p}_{Feature}(\mathbf{X}^{v})$$

where $\mathcal{P}_{Feature}(\cdot)$和$\mathcal{P}^{p}_{Feature}(\cdot)$为相同的非线性MLP架构，$\mathbf{Y}^{v}, \mathbf{Y}_{\mathbf{U}},\mathbf{Y}_{\mathbf{P}} \in \mathbb{R}^{N \times D_l}$。在获得特征层级和伪标签层级这两个层级之后，我们引入对比学习的方法，分别对于这两个层级进行对齐。我们构建的对比损失为：

$$\mathcal{L}_{\text{Contra}} = - \sum\limits_{1 \leq k < k' \leq V } \sum^{N}_{i=1}  \log(\frac{\exp({D(\mathbf{H}^{k}_{i \cdot},\mathbf{H}^{k'}_{i \cdot})) / \tau_f}}{\sum^{N}_{j=1} \exp({D(\mathbf{H}^{k}_{i \cdot},\mathbf{H}^{k'}_{j \cdot})) / \tau_f}})$$

$$ - \sum\limits_{1 \leq k < k' \leq V } \sum^{C}_{i=1}  \log(\frac{\exp({D((\mathbf{Y}^{k}_{\cdot i})^\top,(\mathbf{Y}^{k'}_{\cdot i})^\top) / \tau_f}}{\sum^{C}_{j=1} \exp({D((\mathbf{Y}^{k}_{\cdot i})^\top,(\mathbf{Y}^{k'}_{\cdot j})^\top)) / \tau_l}})$$

Where $D(\cdot)$表示余弦相似度，$\tau_f,\tau_l$分别表示为特征层级和伪标签层级的温度参数。第一项表示为特征层级的相似性的对齐，第二项表示为伪标签层级的相似性的对齐。

为了统一聚类结果的一致性，并且放大关键聚类标签的信息，我们创新性地提出了一种基于回路反馈的对齐地机制。而对于Circuit的设计，我们分为前向对齐和后向反馈。前向对齐的关键在于将更有代表性的聚类结果作为主要目标，弱化无关甚至是错误的聚类标签的影响，因此对于所有的标签我们考虑取最高值的方式获得$y_{ij}$:

$$y_{ij} = \max\{(\mathbf{Y}_{\mathbf{P}})_{ij}, (\mathbf{Y}^1)_{ij}, (\mathbf{Y}^2)_{ij},...,(\mathbf{Y}^v)_{ij}, (\mathbf{Y}_{\mathbf{U}})_{ij}\}$$

而为了将更具代表性的聚类标签更突出，无关甚至是错误的聚类标签进一步弱化，我们进行如下如下操作，获得全局聚类标签$\mathbf{Y}_{all}$:

$$(\mathbf{Y}_{all})_{·j} = \frac{y^2_{·j}}{\sum^C_{k=1}y^2_{·k}}$$

因此对于前向对齐的优化损失为：

$$\mathcal{L}_{forward} = KL(\mathbf{Y}_{all} \parallel \mathbf{Y_{U}}) - \sum^{V}_{v=1} H(\mathbf{Y}^v) - H(Y_{\mathbf{P}}) $$

where$KL( \cdot \parallel \cdot)$为KL散度，$KL(\mathbf{Y}_{all} \parallel \mathbf{Y_{U}})$具体形式为$$\sum_i \sum_j (\mathbf{Y_{all}})_{ij} \log \frac{(\mathbf{Y_{all}})_{ij}}{(\mathbf{Y_{U}})_{ij}}$$.$H(\cdot)$为entropy，对于$H(\mathbf{Y}^v)$而言具体形式为$-\sum^{C}_{j=1} P((\mathbf{Y}^v)_{\cdot j})logP((\mathbf{Y}^v)_{\cdot j})$。

对于后向反馈部分，我们采用Mediation Mapping的方式，使得模型可以逐渐平衡对全局和局部分布的适应，在中间空间找到相对稳定的表示。对于Mediation Mapping的设计为：

$$\mathbf{Y}^{*}_{\mathbf{P}} = \texttt{softmax}( \sigma(\mathbf{Y}_{\mathbf{P}} W_{f_{\mathbf{P}}}^{(1)} + b_{f_{\mathbf{P}}}^{(1)}) W_{f_{\mathbf{P}}}^{(2)} + b_{f_{\mathbf{P}}}^{(2)} )$$

$$\mathbf{Y}^{*}_{\mathbf{U}} = \texttt{softmax}( \sigma(\mathbf{Y}_{\mathbf{U}} W_{f_{\mathbf{U}}}^{(1)} + b_{f_{\mathbf{U}}}^{(1)}) W_{f_{\mathbf{U}}}^{(2)} + b_{f_{\mathbf{U}}}^{(2)} )$$

where$\sigma(\cdot)$为ReLU。$\mathbf{Y}^{*}_{\mathbf{P}}$和$\mathbf{Y}^{*}_{\mathbf{U}}$是从$\mathbf{Y}_{\mathbf{P}},\mathbf{Y}_{\mathbf{U}}$中学习得来的中间表示，它仍然相同的类概率空间，但参数化后可以灵活地变形、重塑分布。在训练早期， Mediation Mapping的存在缓冲了直接对 $\mathbf{Y}_{\mathbf{P}},\mathbf{Y}_{\mathbf{U}}$的强制分布对齐。因此对于后向反馈的优化损失为：

$$\mathcal{L}_{backward} = KL(\mathbf{Y}^{*}_{\mathbf{U}} \parallel \mathbf{Y}_{\mathbf{P}}) + \sum^V_{v=1} KL(\mathbf{Y}^{*}_{\mathbf{P}} \parallel \mathbf{Y}^{v})$$

整体的Hierarchical circuit alignment如图2所示。对于总体的损失优化目标因此有：
$$\mathcal{L} = \mathcal{L}_{Contra} + \mathcal{L}_{forward} + \mathcal{L}_{backward}$$



- 主图标题
  - 我们通过添加input-level prompts, local embedding-level prompts和global embedding-level prompts，对于输入数据的有效语义信息进行充分挖掘。通过将数据和global embedding-level prompts经过Mapping encoder 映射到统一的空间维度$\mathcal{H}$中。数据和prompts所得到的高层次语义特征分别经过data-level和prompt-level pseudo label classfier得到相应的伪标签。得到的不同level的标签进行Hierarchical circuit alignment。



w/o 1 w/o 2 w/o 3







- experimental results

  For instance, on the BBCSport dataset, our model significantly outperforms others, , which not only exceeds the second-best method by 6-7% but also highlights the robustness of our hierarchical curcuit alignment mechanism. 并且对于大样本数据集Caltech-all，不同level的提示设计使得模型能够更好的捕捉各个样本内的特征维度聚类语义信息，以及样本之间的聚类关联，使得更多类别的数据集也能够更好的鉴别无监督类别信息。此外，我们注意到我们的方法基于GCN的结果优于基于distinct MLP，并且两者均优于其他比较的方法，因此说明我们的方法并不依赖于encoder层的某种具体设计，而更具有普遍优越性。







```python
def plot_all_views_pseudo_label(model, device, cnt):
    colors = ['#0072B2', '#D55E00', '#F0E442', '#56B4E9', '#009E73', '#CC79A7', '#E69F00', '#999999', '#000000']
    cmap = ListedColormap(colors[:class_num])  # 根据聚类数目选择颜色
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    all_hs = []  # 用于存储所有视角拼接后的特征
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])
            all_hs.append(hs[v])

    # 将所有视角的特征拼接在一起
    combined_hs = np.concatenate(all_hs, axis=1)

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []

    # 利用 t-SNE 在拼接后的特征上进行降维
    tsne = TSNE(n_components=2,   init='random',learning_rate='auto',random_state=42)
    tsne_result = tsne.fit_transform(combined_hs)

    # 聚类并生成伪标签
    Pseudo_label = kmeans.fit_predict(combined_hs)
    Pseudo_label = Pseudo_label.reshape(data_size, 1)
    Pseudo_label = torch.from_numpy(Pseudo_label)
    new_pseudo_label.append(Pseudo_label)

    # 绘制 t-SNE 聚类结果散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=Pseudo_label.numpy().flatten(), cmap=cmap)

    # 保存为 PNG 文件
    plt.savefig("tsne_cluster_result"+str(cnt)+".png", dpi=300)
    cnt = cnt + 1
    plt.show()

    return new_pseudo_label

```





Visualization of clustering results on the BBCSport and Caltech7 datasets.左侧为原始特征，右侧为经过HiPMVC训练得到的结果。使用t-SNE进行可视化。



Ablation study results for HTRE-MVC with different components removed. 其中IP表示instance-level prompts，EP表示embedding-level prompts。



- Para sensitivity
- 如图3所示，我们将HTRE-MVC总损失函数中的$\beta$和$\gamma$设置范围为$[0.01,0.1,1,10,100]$。我们在Citeseer和Caltech7这两个数据集上进行实验。对于Caltech7数据集而言，参数$\gamma$在1-100范围区间更为敏感。在更少的视角数的数据集Citeseer上$\beta$和$\gamma$的参数敏感性较为平缓。
- We experimentally evaluate the effect of hyperparameters on the clustering performance of GCFAggMVC, which includes the trade-off coefficient λ (i.e., Lr + λLc) and the temperature parameter τ . Figure 3 shows the NMI of GCFAggMVC when λ is varied from 10−3 to 103 and τ from 0.2 to 0.8. From this figure, the clustering results of the proposed GCFAggMVC are insensitive to both λ and τ in the range 0.1 to 1, and the range 0.3 to 0.5, respectively. Empirically, we set λ and τ to 1.0 and 0.5.



- ablation

  To assess the contribution of each stage in HTRE-MVC, we conduct targeted ablation studies, with the results summarized in Table \ref{table:ablation}.







(icmvc) baosf@ubuntu:/mnt/sda/baosf/ICMVC-main$ python train.py
2024-12-06 00:04:29 - root - INFO: - Dataset:MSRC_v1
K neighbors 10
missing_rate 0.0
2024-12-06 00:04:36 - root - INFO: - Epoch:100/500===>loss=8.2892
{'AMI': 0.7779, 'NMI': 0.7884, 'ARI': 0.7572, 'accuracy': 0.881, 'precision': 0.8792, 'recall': 0.881, 'f_measure': 0.8798}
2024-12-06 00:04:42 - root - INFO: - Epoch:200/500===>loss=8.1636
{'AMI': 0.7779, 'NMI': 0.7884, 'ARI': 0.7572, 'accuracy': 0.881, 'precision': 0.8792, 'recall': 0.881, 'f_measure': 0.8798}
2024-12-06 00:04:48 - root - INFO: - Epoch:300/500===>loss=8.1347
{'AMI': 0.7779, 'NMI': 0.7884, 'ARI': 0.7572, 'accuracy': 0.881, 'precision': 0.8792, 'recall': 0.881, 'f_measure': 0.8798}
2024-12-06 00:04:53 - root - INFO: - Epoch:400/500===>loss=8.1645
{'AMI': 0.7779, 'NMI': 0.7884, 'ARI': 0.7572, 'accuracy': 0.881, 'precision': 0.8792, 'recall': 0.881, 'f_measure': 0.8798}
2024-12-06 00:04:59 - root - INFO: - Epoch:500/500===>loss=8.1221
{'AMI': 0.7779, 'NMI': 0.7884, 'ARI': 0.7572, 'accuracy': 0.881, 'precision': 0.8792, 'recall': 0.881, 'f_measure': 0.8798}
2024-12-06 00:05:05 - root - INFO: - Epoch:100/500===>loss=8.3082
{'AMI': 0.754, 'NMI': 0.7656, 'ARI': 0.729, 'accuracy': 0.8667, 'precision': 0.8657, 'recall': 0.8667, 'f_measure': 0.8657}
2024-12-06 00:05:11 - root - INFO: - Epoch:200/500===>loss=8.1872
{'AMI': 0.7657, 'NMI': 0.7767, 'ARI': 0.7369, 'accuracy': 0.8714, 'precision': 0.8744, 'recall': 0.8714, 'f_measure': 0.8722}
2024-12-06 00:05:16 - root - INFO: - Epoch:300/500===>loss=8.1538
{'AMI': 0.7692, 'NMI': 0.7801, 'ARI': 0.7442, 'accuracy': 0.8762, 'precision': 0.878, 'recall': 0.8762, 'f_measure': 0.8766}
2024-12-06 00:05:22 - root - INFO: - Epoch:400/500===>loss=8.1293
{'AMI': 0.7692, 'NMI': 0.7801, 'ARI': 0.7442, 'accuracy': 0.8762, 'precision': 0.878, 'recall': 0.8762, 'f_measure': 0.8766}
2024-12-06 00:05:27 - root - INFO: - Epoch:500/500===>loss=8.1231
{'AMI': 0.7657, 'NMI': 0.7767, 'ARI': 0.7369, 'accuracy': 0.8714, 'precision': 0.8744, 'recall': 0.8714, 'f_measure': 0.8722}
2024-12-06 00:05:33 - root - INFO: - Epoch:100/500===>loss=8.3113
{'AMI': 0.7583, 'NMI': 0.7696, 'ARI': 0.7421, 'accuracy': 0.8762, 'precision': 0.8758, 'recall': 0.8762, 'f_measure': 0.8756}
2024-12-06 00:05:38 - root - INFO: - Epoch:200/500===>loss=8.1957
{'AMI': 0.7693, 'NMI': 0.7802, 'ARI': 0.7523, 'accuracy': 0.881, 'precision': 0.8796, 'recall': 0.881, 'f_measure': 0.8799}
2024-12-06 00:05:44 - root - INFO: - Epoch:300/500===>loss=8.3096
{'AMI': 0.7693, 'NMI': 0.7802, 'ARI': 0.7523, 'accuracy': 0.881, 'precision': 0.8796, 'recall': 0.881, 'f_measure': 0.8799}
2024-12-06 00:05:50 - root - INFO: - Epoch:400/500===>loss=8.1253
{'AMI': 0.7693, 'NMI': 0.7802, 'ARI': 0.7523, 'accuracy': 0.881, 'precision': 0.8796, 'recall': 0.881, 'f_measure': 0.8799}
2024-12-06 00:05:55 - root - INFO: - Epoch:500/500===>loss=8.0938
{'AMI': 0.7693, 'NMI': 0.7802, 'ARI': 0.7523, 'accuracy': 0.881, 'precision': 0.8796, 'recall': 0.881, 'f_measure': 0.8799}
2024-12-06 00:06:01 - root - INFO: - Epoch:100/500===>loss=8.2776
{'AMI': 0.7909, 'NMI': 0.8008, 'ARI': 0.7741, 'accuracy': 0.8905, 'precision': 0.8895, 'recall': 0.8905, 'f_measure': 0.8894}
2024-12-06 00:06:06 - root - INFO: - Epoch:200/500===>loss=8.1713
{'AMI': 0.7909, 'NMI': 0.8008, 'ARI': 0.7741, 'accuracy': 0.8905, 'precision': 0.8895, 'recall': 0.8905, 'f_measure': 0.8894}
2024-12-06 00:06:12 - root - INFO: - Epoch:300/500===>loss=8.1347
{'AMI': 0.7909, 'NMI': 0.8008, 'ARI': 0.7741, 'accuracy': 0.8905, 'precision': 0.8895, 'recall': 0.8905, 'f_measure': 0.8894}
2024-12-06 00:06:17 - root - INFO: - Epoch:400/500===>loss=8.1062
{'AMI': 0.7909, 'NMI': 0.8008, 'ARI': 0.7741, 'accuracy': 0.8905, 'precision': 0.8895, 'recall': 0.8905, 'f_measure': 0.8894}
2024-12-06 00:06:23 - root - INFO: - Epoch:500/500===>loss=8.0896
{'AMI': 0.7909, 'NMI': 0.8008, 'ARI': 0.7741, 'accuracy': 0.8905, 'precision': 0.8895, 'recall': 0.8905, 'f_measure': 0.8894}
2024-12-06 00:06:28 - root - INFO: - Epoch:100/500===>loss=8.2918
{'AMI': 0.7576, 'NMI': 0.7691, 'ARI': 0.7469, 'accuracy': 0.881, 'precision': 0.8828, 'recall': 0.881, 'f_measure': 0.8812}
2024-12-06 00:06:34 - root - INFO: - Epoch:200/500===>loss=8.2512
{'AMI': 0.78, 'NMI': 0.7904, 'ARI': 0.7682, 'accuracy': 0.8905, 'precision': 0.8899, 'recall': 0.8905, 'f_measure': 0.8897}
2024-12-06 00:06:40 - root - INFO: - Epoch:300/500===>loss=8.1474
{'AMI': 0.78, 'NMI': 0.7904, 'ARI': 0.7682, 'accuracy': 0.8905, 'precision': 0.8899, 'recall': 0.8905, 'f_measure': 0.8897}
2024-12-06 00:06:46 - root - INFO: - Epoch:400/500===>loss=8.1335
{'AMI': 0.78, 'NMI': 0.7904, 'ARI': 0.7682, 'accuracy': 0.8905, 'precision': 0.8899, 'recall': 0.8905, 'f_measure': 0.8897}
2024-12-06 00:06:51 - root - INFO: - Epoch:500/500===>loss=8.2466
{'AMI': 0.78, 'NMI': 0.7904, 'ARI': 0.7682, 'accuracy': 0.8905, 'precision': 0.8899, 'recall': 0.8905, 'f_measure': 0.8897}
2024-12-06 00:06:51 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:06:51 - root - INFO: - [0.881, 0.8714, 0.881, 0.8905, 0.8905]
2024-12-06 00:06:51 - root - INFO: - [0.7884, 0.7767, 0.7802, 0.8008, 0.7904]
2024-12-06 00:06:51 - root - INFO: - [0.7572, 0.7369, 0.7523, 0.7741, 0.7682]
2024-12-06 00:06:51 - root - INFO: -  
                     ACC 88.29 std 0.71
                     NMI 78.73 std 0.84 
                     ARI 75.77 std 1.30
2024-12-06 00:06:51 - root - INFO: - 88.29,0.71;78.73,0.84;75.77,1.3;
acc: 88.29 ,nmi: 78.73 ,ari: 75.77
K neighbors 10
missing_rate 0.3
2024-12-06 00:06:57 - root - INFO: - Epoch:100/500===>loss=8.2815
{'AMI': 0.6311, 'NMI': 0.6485, 'ARI': 0.5727, 'accuracy': 0.7333, 'precision': 0.7282, 'recall': 0.7333, 'f_measure': 0.7299}
2024-12-06 00:07:03 - root - INFO: - Epoch:200/500===>loss=8.1467
{'AMI': 0.6362, 'NMI': 0.6534, 'ARI': 0.5783, 'accuracy': 0.7381, 'precision': 0.7341, 'recall': 0.7381, 'f_measure': 0.7353}
2024-12-06 00:07:08 - root - INFO: - Epoch:300/500===>loss=8.1420
{'AMI': 0.6362, 'NMI': 0.6534, 'ARI': 0.5783, 'accuracy': 0.7381, 'precision': 0.7341, 'recall': 0.7381, 'f_measure': 0.7353}
2024-12-06 00:07:14 - root - INFO: - Epoch:400/500===>loss=8.1079
{'AMI': 0.6362, 'NMI': 0.6534, 'ARI': 0.5783, 'accuracy': 0.7381, 'precision': 0.7341, 'recall': 0.7381, 'f_measure': 0.7353}
2024-12-06 00:07:19 - root - INFO: - Epoch:500/500===>loss=8.0963
{'AMI': 0.6362, 'NMI': 0.6534, 'ARI': 0.5783, 'accuracy': 0.7381, 'precision': 0.7341, 'recall': 0.7381, 'f_measure': 0.7353}
2024-12-06 00:07:25 - root - INFO: - Epoch:100/500===>loss=8.2709
{'AMI': 0.7445, 'NMI': 0.7565, 'ARI': 0.7228, 'accuracy': 0.8667, 'precision': 0.8691, 'recall': 0.8667, 'f_measure': 0.8666}
2024-12-06 00:07:30 - root - INFO: - Epoch:200/500===>loss=8.1676
{'AMI': 0.7558, 'NMI': 0.7673, 'ARI': 0.7339, 'accuracy': 0.8714, 'precision': 0.8724, 'recall': 0.8714, 'f_measure': 0.8708}
2024-12-06 00:07:36 - root - INFO: - Epoch:300/500===>loss=8.1336
{'AMI': 0.7558, 'NMI': 0.7673, 'ARI': 0.7339, 'accuracy': 0.8714, 'precision': 0.8724, 'recall': 0.8714, 'f_measure': 0.8708}
2024-12-06 00:07:41 - root - INFO: - Epoch:400/500===>loss=8.1075
{'AMI': 0.7558, 'NMI': 0.7673, 'ARI': 0.7339, 'accuracy': 0.8714, 'precision': 0.8724, 'recall': 0.8714, 'f_measure': 0.8708}
2024-12-06 00:07:47 - root - INFO: - Epoch:500/500===>loss=8.1042
{'AMI': 0.7558, 'NMI': 0.7673, 'ARI': 0.7339, 'accuracy': 0.8714, 'precision': 0.8724, 'recall': 0.8714, 'f_measure': 0.8708}
2024-12-06 00:07:52 - root - INFO: - Epoch:100/500===>loss=8.2853
{'AMI': 0.7208, 'NMI': 0.7339, 'ARI': 0.6765, 'accuracy': 0.8381, 'precision': 0.8434, 'recall': 0.8381, 'f_measure': 0.839}
2024-12-06 00:07:58 - root - INFO: - Epoch:200/500===>loss=8.1686
{'AMI': 0.7208, 'NMI': 0.7339, 'ARI': 0.6765, 'accuracy': 0.8381, 'precision': 0.8434, 'recall': 0.8381, 'f_measure': 0.839}
2024-12-06 00:08:03 - root - INFO: - Epoch:300/500===>loss=8.1569
{'AMI': 0.7208, 'NMI': 0.7339, 'ARI': 0.6765, 'accuracy': 0.8381, 'precision': 0.8434, 'recall': 0.8381, 'f_measure': 0.839}
2024-12-06 00:08:09 - root - INFO: - Epoch:400/500===>loss=8.1281
{'AMI': 0.7321, 'NMI': 0.7448, 'ARI': 0.6877, 'accuracy': 0.8429, 'precision': 0.8463, 'recall': 0.8429, 'f_measure': 0.843}
2024-12-06 00:08:14 - root - INFO: - Epoch:500/500===>loss=8.1253
{'AMI': 0.7321, 'NMI': 0.7448, 'ARI': 0.6877, 'accuracy': 0.8429, 'precision': 0.8463, 'recall': 0.8429, 'f_measure': 0.843}
2024-12-06 00:08:20 - root - INFO: - Epoch:100/500===>loss=8.2859
{'AMI': 0.6604, 'NMI': 0.6764, 'ARI': 0.6012, 'accuracy': 0.7667, 'precision': 0.7666, 'recall': 0.7667, 'f_measure': 0.766}
2024-12-06 00:08:25 - root - INFO: - Epoch:200/500===>loss=8.1649
{'AMI': 0.6663, 'NMI': 0.682, 'ARI': 0.6026, 'accuracy': 0.7667, 'precision': 0.7678, 'recall': 0.7667, 'f_measure': 0.7666}
2024-12-06 00:08:31 - root - INFO: - Epoch:300/500===>loss=8.1269
{'AMI': 0.6663, 'NMI': 0.682, 'ARI': 0.6026, 'accuracy': 0.7667, 'precision': 0.7678, 'recall': 0.7667, 'f_measure': 0.7666}
2024-12-06 00:08:38 - root - INFO: - Epoch:400/500===>loss=8.1499
{'AMI': 0.6663, 'NMI': 0.682, 'ARI': 0.6026, 'accuracy': 0.7667, 'precision': 0.7678, 'recall': 0.7667, 'f_measure': 0.7666}
2024-12-06 00:08:43 - root - INFO: - Epoch:500/500===>loss=8.1408
{'AMI': 0.6663, 'NMI': 0.682, 'ARI': 0.6026, 'accuracy': 0.7667, 'precision': 0.7678, 'recall': 0.7667, 'f_measure': 0.7666}
2024-12-06 00:08:48 - root - INFO: - Epoch:100/500===>loss=8.3149
{'AMI': 0.704, 'NMI': 0.7179, 'ARI': 0.6282, 'accuracy': 0.8, 'precision': 0.8066, 'recall': 0.8, 'f_measure': 0.8014}
2024-12-06 00:08:54 - root - INFO: - Epoch:200/500===>loss=8.2126
{'AMI': 0.7037, 'NMI': 0.7177, 'ARI': 0.6294, 'accuracy': 0.8, 'precision': 0.8057, 'recall': 0.8, 'f_measure': 0.8016}
2024-12-06 00:09:00 - root - INFO: - Epoch:300/500===>loss=8.1822
{'AMI': 0.7037, 'NMI': 0.7177, 'ARI': 0.6294, 'accuracy': 0.8, 'precision': 0.8057, 'recall': 0.8, 'f_measure': 0.8016}
2024-12-06 00:09:06 - root - INFO: - Epoch:400/500===>loss=8.1382
{'AMI': 0.7001, 'NMI': 0.7142, 'ARI': 0.6222, 'accuracy': 0.7952, 'precision': 0.8025, 'recall': 0.7952, 'f_measure': 0.7972}
2024-12-06 00:09:11 - root - INFO: - Epoch:500/500===>loss=8.1169
{'AMI': 0.7001, 'NMI': 0.7142, 'ARI': 0.6222, 'accuracy': 0.7952, 'precision': 0.8025, 'recall': 0.7952, 'f_measure': 0.7972}
2024-12-06 00:09:11 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:09:11 - root - INFO: - [0.7381, 0.8714, 0.8429, 0.7667, 0.7952]
2024-12-06 00:09:11 - root - INFO: - [0.6534, 0.7673, 0.7448, 0.682, 0.7142]
2024-12-06 00:09:11 - root - INFO: - [0.5783, 0.7339, 0.6877, 0.6026, 0.6222]
2024-12-06 00:09:11 - root - INFO: -  
                     ACC 80.29 std 4.87
                     NMI 71.23 std 4.12 
                     ARI 64.49 std 5.74
2024-12-06 00:09:11 - root - INFO: - 80.29,4.87;71.23,4.12;64.49,5.74;
acc: 80.29 ,nmi: 71.23 ,ari: 64.49
(icmvc) baosf@ubuntu:/mnt/sda/baosf/ICMVC-main$ python train.py
2024-12-06 00:23:19 - root - INFO: - Dataset:LandUse-21
K neighbors 10
missing_rate 0.0
2024-12-06 00:23:29 - root - INFO: - Epoch:100/500===>loss=50.8814
{'AMI': 0.2968, 'NMI': 0.3204, 'ARI': 0.1276, 'accuracy': 0.2743, 'precision': 0.2989, 'recall': 0.2743, 'f_measure': 0.2789}
2024-12-06 00:23:37 - root - INFO: - Epoch:200/500===>loss=50.0097
{'AMI': 0.2909, 'NMI': 0.3145, 'ARI': 0.1323, 'accuracy': 0.2814, 'precision': 0.2941, 'recall': 0.2814, 'f_measure': 0.2837}
2024-12-06 00:23:46 - root - INFO: - Epoch:300/500===>loss=49.5670
{'AMI': 0.2841, 'NMI': 0.3078, 'ARI': 0.1307, 'accuracy': 0.2805, 'precision': 0.2906, 'recall': 0.2805, 'f_measure': 0.2833}
2024-12-06 00:23:56 - root - INFO: - Epoch:400/500===>loss=49.5006
{'AMI': 0.2818, 'NMI': 0.3055, 'ARI': 0.1308, 'accuracy': 0.28, 'precision': 0.288, 'recall': 0.28, 'f_measure': 0.2827}
2024-12-06 00:24:06 - root - INFO: - Epoch:500/500===>loss=49.3480
{'AMI': 0.2766, 'NMI': 0.3005, 'ARI': 0.1295, 'accuracy': 0.2805, 'precision': 0.2866, 'recall': 0.2805, 'f_measure': 0.2829}
2024-12-06 00:24:16 - root - INFO: - Epoch:100/500===>loss=50.8447
{'AMI': 0.3143, 'NMI': 0.3372, 'ARI': 0.1522, 'accuracy': 0.2867, 'precision': 0.3054, 'recall': 0.2867, 'f_measure': 0.2903}
2024-12-06 00:24:27 - root - INFO: - Epoch:200/500===>loss=50.0590
{'AMI': 0.3048, 'NMI': 0.3279, 'ARI': 0.1537, 'accuracy': 0.2938, 'precision': 0.3039, 'recall': 0.2938, 'f_measure': 0.2948}
2024-12-06 00:24:37 - root - INFO: - Epoch:300/500===>loss=49.5113
{'AMI': 0.3033, 'NMI': 0.3264, 'ARI': 0.1523, 'accuracy': 0.2886, 'precision': 0.2977, 'recall': 0.2886, 'f_measure': 0.2898}
2024-12-06 00:24:46 - root - INFO: - Epoch:400/500===>loss=49.2954
{'AMI': 0.2993, 'NMI': 0.3225, 'ARI': 0.1506, 'accuracy': 0.2843, 'precision': 0.2874, 'recall': 0.2843, 'f_measure': 0.2846}
2024-12-06 00:24:56 - root - INFO: - Epoch:500/500===>loss=49.0322
{'AMI': 0.2935, 'NMI': 0.3168, 'ARI': 0.1487, 'accuracy': 0.2838, 'precision': 0.286, 'recall': 0.2838, 'f_measure': 0.2843}
2024-12-06 00:25:05 - root - INFO: - Epoch:100/500===>loss=50.8026
{'AMI': 0.3042, 'NMI': 0.3274, 'ARI': 0.148, 'accuracy': 0.2781, 'precision': 0.293, 'recall': 0.2781, 'f_measure': 0.2783}
2024-12-06 00:25:15 - root - INFO: - Epoch:200/500===>loss=50.0322
{'AMI': 0.2972, 'NMI': 0.3206, 'ARI': 0.1487, 'accuracy': 0.28, 'precision': 0.2921, 'recall': 0.28, 'f_measure': 0.2807}
2024-12-06 00:25:25 - root - INFO: - Epoch:300/500===>loss=49.7247
{'AMI': 0.294, 'NMI': 0.3174, 'ARI': 0.1482, 'accuracy': 0.279, 'precision': 0.2812, 'recall': 0.279, 'f_measure': 0.2777}
2024-12-06 00:25:34 - root - INFO: - Epoch:400/500===>loss=49.3451
{'AMI': 0.2923, 'NMI': 0.3156, 'ARI': 0.1485, 'accuracy': 0.2771, 'precision': 0.2784, 'recall': 0.2771, 'f_measure': 0.2766}
2024-12-06 00:25:44 - root - INFO: - Epoch:500/500===>loss=49.2919
{'AMI': 0.2901, 'NMI': 0.3135, 'ARI': 0.1488, 'accuracy': 0.2776, 'precision': 0.2765, 'recall': 0.2776, 'f_measure': 0.2765}
2024-12-06 00:25:53 - root - INFO: - Epoch:100/500===>loss=50.8329
{'AMI': 0.3093, 'NMI': 0.3323, 'ARI': 0.1532, 'accuracy': 0.2929, 'precision': 0.3009, 'recall': 0.2929, 'f_measure': 0.2929}
2024-12-06 00:26:03 - root - INFO: - Epoch:200/500===>loss=50.0493
{'AMI': 0.307, 'NMI': 0.3299, 'ARI': 0.1543, 'accuracy': 0.2943, 'precision': 0.3011, 'recall': 0.2943, 'f_measure': 0.2948}
2024-12-06 00:26:13 - root - INFO: - Epoch:300/500===>loss=49.5866
{'AMI': 0.3052, 'NMI': 0.3282, 'ARI': 0.1543, 'accuracy': 0.2919, 'precision': 0.2927, 'recall': 0.2919, 'f_measure': 0.2907}
2024-12-06 00:26:22 - root - INFO: - Epoch:400/500===>loss=49.2462
{'AMI': 0.2967, 'NMI': 0.3199, 'ARI': 0.1499, 'accuracy': 0.2867, 'precision': 0.2871, 'recall': 0.2867, 'f_measure': 0.2861}
2024-12-06 00:26:32 - root - INFO: - Epoch:500/500===>loss=49.1524
{'AMI': 0.2962, 'NMI': 0.3195, 'ARI': 0.1506, 'accuracy': 0.2871, 'precision': 0.2873, 'recall': 0.2871, 'f_measure': 0.2866}
2024-12-06 00:26:41 - root - INFO: - Epoch:100/500===>loss=50.7569
{'AMI': 0.3124, 'NMI': 0.3354, 'ARI': 0.1497, 'accuracy': 0.2838, 'precision': 0.2947, 'recall': 0.2838, 'f_measure': 0.2834}
2024-12-06 00:26:51 - root - INFO: - Epoch:200/500===>loss=49.9944
{'AMI': 0.3097, 'NMI': 0.3327, 'ARI': 0.1526, 'accuracy': 0.2881, 'precision': 0.296, 'recall': 0.2881, 'f_measure': 0.2877}
2024-12-06 00:27:00 - root - INFO: - Epoch:300/500===>loss=49.6477
{'AMI': 0.302, 'NMI': 0.3252, 'ARI': 0.1496, 'accuracy': 0.2843, 'precision': 0.2894, 'recall': 0.2843, 'f_measure': 0.2841}
2024-12-06 00:27:10 - root - INFO: - Epoch:400/500===>loss=49.4084
{'AMI': 0.2989, 'NMI': 0.3221, 'ARI': 0.1479, 'accuracy': 0.2819, 'precision': 0.2868, 'recall': 0.2819, 'f_measure': 0.2823}
2024-12-06 00:27:20 - root - INFO: - Epoch:500/500===>loss=49.2209
{'AMI': 0.2955, 'NMI': 0.3187, 'ARI': 0.1481, 'accuracy': 0.2852, 'precision': 0.2876, 'recall': 0.2852, 'f_measure': 0.2852}
2024-12-06 00:27:20 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:27:20 - root - INFO: - [0.2805, 0.2838, 0.2776, 0.2871, 0.2852]
2024-12-06 00:27:20 - root - INFO: - [0.3005, 0.3168, 0.3135, 0.3195, 0.3187]
2024-12-06 00:27:20 - root - INFO: - [0.1295, 0.1487, 0.1488, 0.1506, 0.1481]
2024-12-06 00:27:20 - root - INFO: -  
                     ACC 28.28 std 0.34
                     NMI 31.38 std 0.70 
                     ARI 14.51 std 0.79
2024-12-06 00:27:20 - root - INFO: - 28.28,0.34;31.38,0.7;14.51,0.79;
acc: 28.28 ,nmi: 31.38 ,ari: 14.51
K neighbors 10
missing_rate 0.3
2024-12-06 00:27:31 - root - INFO: - Epoch:100/500===>loss=50.3420
{'AMI': 0.2788, 'NMI': 0.3028, 'ARI': 0.1292, 'accuracy': 0.2581, 'precision': 0.2664, 'recall': 0.2581, 'f_measure': 0.2573}
2024-12-06 00:27:41 - root - INFO: - Epoch:200/500===>loss=49.5393
{'AMI': 0.2682, 'NMI': 0.2925, 'ARI': 0.1271, 'accuracy': 0.2543, 'precision': 0.263, 'recall': 0.2543, 'f_measure': 0.2561}
2024-12-06 00:27:50 - root - INFO: - Epoch:300/500===>loss=49.2437
{'AMI': 0.2643, 'NMI': 0.2886, 'ARI': 0.1264, 'accuracy': 0.2552, 'precision': 0.259, 'recall': 0.2552, 'f_measure': 0.2559}
2024-12-06 00:28:00 - root - INFO: - Epoch:400/500===>loss=49.0504
{'AMI': 0.2624, 'NMI': 0.2867, 'ARI': 0.1269, 'accuracy': 0.2543, 'precision': 0.254, 'recall': 0.2543, 'f_measure': 0.2537}
2024-12-06 00:28:09 - root - INFO: - Epoch:500/500===>loss=49.1453
{'AMI': 0.2625, 'NMI': 0.2868, 'ARI': 0.1268, 'accuracy': 0.2562, 'precision': 0.2558, 'recall': 0.2562, 'f_measure': 0.2558}
2024-12-06 00:28:19 - root - INFO: - Epoch:100/500===>loss=50.3937
{'AMI': 0.2827, 'NMI': 0.3066, 'ARI': 0.1299, 'accuracy': 0.2576, 'precision': 0.2758, 'recall': 0.2576, 'f_measure': 0.2616}
2024-12-06 00:28:29 - root - INFO: - Epoch:200/500===>loss=49.5739
{'AMI': 0.2817, 'NMI': 0.3054, 'ARI': 0.1303, 'accuracy': 0.2557, 'precision': 0.2618, 'recall': 0.2557, 'f_measure': 0.2569}
2024-12-06 00:28:39 - root - INFO: - Epoch:300/500===>loss=49.2233
{'AMI': 0.2789, 'NMI': 0.3027, 'ARI': 0.1288, 'accuracy': 0.2557, 'precision': 0.2586, 'recall': 0.2557, 'f_measure': 0.2564}
2024-12-06 00:28:48 - root - INFO: - Epoch:400/500===>loss=49.0601
{'AMI': 0.2767, 'NMI': 0.3005, 'ARI': 0.1282, 'accuracy': 0.2529, 'precision': 0.2544, 'recall': 0.2529, 'f_measure': 0.2531}
2024-12-06 00:28:58 - root - INFO: - Epoch:500/500===>loss=49.0122
{'AMI': 0.2717, 'NMI': 0.2957, 'ARI': 0.1261, 'accuracy': 0.251, 'precision': 0.2524, 'recall': 0.251, 'f_measure': 0.2513}
2024-12-06 00:29:08 - root - INFO: - Epoch:100/500===>loss=50.4797
{'AMI': 0.3063, 'NMI': 0.3295, 'ARI': 0.1476, 'accuracy': 0.2871, 'precision': 0.2985, 'recall': 0.2871, 'f_measure': 0.2857}
2024-12-06 00:29:18 - root - INFO: - Epoch:200/500===>loss=49.5297
{'AMI': 0.2926, 'NMI': 0.3161, 'ARI': 0.1447, 'accuracy': 0.2848, 'precision': 0.2949, 'recall': 0.2848, 'f_measure': 0.2847}
2024-12-06 00:29:27 - root - INFO: - Epoch:300/500===>loss=49.0879
{'AMI': 0.2822, 'NMI': 0.3059, 'ARI': 0.1426, 'accuracy': 0.2824, 'precision': 0.2876, 'recall': 0.2824, 'f_measure': 0.2825}
2024-12-06 00:29:37 - root - INFO: - Epoch:400/500===>loss=48.8813
{'AMI': 0.2741, 'NMI': 0.298, 'ARI': 0.1404, 'accuracy': 0.2819, 'precision': 0.283, 'recall': 0.2819, 'f_measure': 0.2816}
2024-12-06 00:29:47 - root - INFO: - Epoch:500/500===>loss=48.8148
{'AMI': 0.2727, 'NMI': 0.2967, 'ARI': 0.1395, 'accuracy': 0.2805, 'precision': 0.2792, 'recall': 0.2805, 'f_measure': 0.2793}
2024-12-06 00:29:56 - root - INFO: - Epoch:100/500===>loss=50.3814
{'AMI': 0.2921, 'NMI': 0.3157, 'ARI': 0.1356, 'accuracy': 0.271, 'precision': 0.2815, 'recall': 0.271, 'f_measure': 0.2718}
2024-12-06 00:30:06 - root - INFO: - Epoch:200/500===>loss=49.4972
{'AMI': 0.2947, 'NMI': 0.3181, 'ARI': 0.1382, 'accuracy': 0.2724, 'precision': 0.2804, 'recall': 0.2724, 'f_measure': 0.2729}
2024-12-06 00:30:16 - root - INFO: - Epoch:300/500===>loss=49.1292
{'AMI': 0.2855, 'NMI': 0.3091, 'ARI': 0.1342, 'accuracy': 0.269, 'precision': 0.2718, 'recall': 0.269, 'f_measure': 0.2692}
2024-12-06 00:30:26 - root - INFO: - Epoch:400/500===>loss=49.0744
{'AMI': 0.282, 'NMI': 0.3057, 'ARI': 0.1332, 'accuracy': 0.2695, 'precision': 0.2716, 'recall': 0.2695, 'f_measure': 0.2698}
2024-12-06 00:30:35 - root - INFO: - Epoch:500/500===>loss=48.7854
{'AMI': 0.279, 'NMI': 0.3027, 'ARI': 0.1318, 'accuracy': 0.269, 'precision': 0.2698, 'recall': 0.269, 'f_measure': 0.269}
2024-12-06 00:30:45 - root - INFO: - Epoch:100/500===>loss=50.3075
{'AMI': 0.2834, 'NMI': 0.3073, 'ARI': 0.1316, 'accuracy': 0.2724, 'precision': 0.2827, 'recall': 0.2724, 'f_measure': 0.2723}
2024-12-06 00:30:55 - root - INFO: - Epoch:200/500===>loss=49.5690
{'AMI': 0.2722, 'NMI': 0.2963, 'ARI': 0.1297, 'accuracy': 0.2686, 'precision': 0.2739, 'recall': 0.2686, 'f_measure': 0.2691}
2024-12-06 00:31:05 - root - INFO: - Epoch:300/500===>loss=49.0704
{'AMI': 0.2667, 'NMI': 0.2909, 'ARI': 0.1275, 'accuracy': 0.2638, 'precision': 0.2689, 'recall': 0.2638, 'f_measure': 0.2653}
2024-12-06 00:31:15 - root - INFO: - Epoch:400/500===>loss=48.8185
{'AMI': 0.2605, 'NMI': 0.2849, 'ARI': 0.1256, 'accuracy': 0.2638, 'precision': 0.267, 'recall': 0.2638, 'f_measure': 0.2647}
2024-12-06 00:31:24 - root - INFO: - Epoch:500/500===>loss=48.8004
{'AMI': 0.2594, 'NMI': 0.2838, 'ARI': 0.126, 'accuracy': 0.2629, 'precision': 0.2643, 'recall': 0.2629, 'f_measure': 0.2633}
2024-12-06 00:31:24 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:31:24 - root - INFO: - [0.2562, 0.251, 0.2805, 0.269, 0.2629]
2024-12-06 00:31:24 - root - INFO: - [0.2868, 0.2957, 0.2967, 0.3027, 0.2838]
2024-12-06 00:31:24 - root - INFO: - [0.1268, 0.1261, 0.1395, 0.1318, 0.126]
2024-12-06 00:31:24 - root - INFO: -  
                     ACC 26.39 std 1.03
                     NMI 29.31 std 0.69 
                     ARI 13.00 std 0.52
2024-12-06 00:31:24 - root - INFO: - 26.39,1.03;29.31,0.69;13.0,0.52;
acc: 26.39 ,nmi: 29.31 ,ari: 13.0
(icmvc) baosf@ubuntu:/mnt/sda/baosf/ICMVC-main$ python train.py
2024-12-06 00:32:31 - root - INFO: - Dataset:LandUse-21
K neighbors 10
missing_rate 0.0
2024-12-06 00:32:41 - root - INFO: - Epoch:100/200===>loss=50.9058
{'AMI': 0.2994, 'NMI': 0.3228, 'ARI': 0.1301, 'accuracy': 0.2595, 'precision': 0.2759, 'recall': 0.2595, 'f_measure': 0.262}
2024-12-06 00:32:49 - root - INFO: - Epoch:200/200===>loss=50.0970
{'AMI': 0.2948, 'NMI': 0.3182, 'ARI': 0.1303, 'accuracy': 0.2586, 'precision': 0.267, 'recall': 0.2586, 'f_measure': 0.259}
2024-12-06 00:32:57 - root - INFO: - Epoch:100/200===>loss=50.8287
{'AMI': 0.302, 'NMI': 0.3252, 'ARI': 0.1462, 'accuracy': 0.2676, 'precision': 0.28, 'recall': 0.2676, 'f_measure': 0.2689}
2024-12-06 00:33:05 - root - INFO: - Epoch:200/200===>loss=50.1586
{'AMI': 0.3021, 'NMI': 0.3253, 'ARI': 0.1489, 'accuracy': 0.2748, 'precision': 0.2894, 'recall': 0.2748, 'f_measure': 0.2781}
2024-12-06 00:33:13 - root - INFO: - Epoch:100/200===>loss=50.8017
{'AMI': 0.3029, 'NMI': 0.3261, 'ARI': 0.1442, 'accuracy': 0.2805, 'precision': 0.3006, 'recall': 0.2805, 'f_measure': 0.2851}
2024-12-06 00:33:21 - root - INFO: - Epoch:200/200===>loss=49.9813
{'AMI': 0.3037, 'NMI': 0.3268, 'ARI': 0.1432, 'accuracy': 0.2857, 'precision': 0.3043, 'recall': 0.2857, 'f_measure': 0.2906}
2024-12-06 00:33:30 - root - INFO: - Epoch:100/200===>loss=50.8820
{'AMI': 0.2959, 'NMI': 0.3194, 'ARI': 0.1461, 'accuracy': 0.281, 'precision': 0.2877, 'recall': 0.281, 'f_measure': 0.2794}
2024-12-06 00:33:38 - root - INFO: - Epoch:200/200===>loss=49.9660
{'AMI': 0.2924, 'NMI': 0.3158, 'ARI': 0.1469, 'accuracy': 0.289, 'precision': 0.2988, 'recall': 0.289, 'f_measure': 0.2909}
2024-12-06 00:33:46 - root - INFO: - Epoch:100/200===>loss=50.8323
{'AMI': 0.3035, 'NMI': 0.327, 'ARI': 0.1286, 'accuracy': 0.2548, 'precision': 0.2877, 'recall': 0.2548, 'f_measure': 0.2617}
2024-12-06 00:33:54 - root - INFO: - Epoch:200/200===>loss=50.0145
{'AMI': 0.2943, 'NMI': 0.3179, 'ARI': 0.1265, 'accuracy': 0.2562, 'precision': 0.279, 'recall': 0.2562, 'f_measure': 0.2615}
2024-12-06 00:33:54 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:33:54 - root - INFO: - [0.2586, 0.2748, 0.2857, 0.289, 0.2562]
2024-12-06 00:33:54 - root - INFO: - [0.3182, 0.3253, 0.3268, 0.3158, 0.3179]
2024-12-06 00:33:54 - root - INFO: - [0.1303, 0.1489, 0.1432, 0.1469, 0.1265]
2024-12-06 00:33:54 - root - INFO: -  
                     ACC 27.29 std 1.35
                     NMI 32.08 std 0.44 
                     ARI 13.92 std 0.91
2024-12-06 00:33:54 - root - INFO: - 27.29,1.35;32.08,0.44;13.92,0.91;
acc: 27.29 ,nmi: 32.08 ,ari: 13.92
K neighbors 10
missing_rate 0.3
2024-12-06 00:34:03 - root - INFO: - Epoch:100/200===>loss=50.3745
{'AMI': 0.2746, 'NMI': 0.2988, 'ARI': 0.1204, 'accuracy': 0.2605, 'precision': 0.2829, 'recall': 0.2605, 'f_measure': 0.2668}
2024-12-06 00:34:10 - root - INFO: - Epoch:200/200===>loss=49.4572
{'AMI': 0.2772, 'NMI': 0.3012, 'ARI': 0.1249, 'accuracy': 0.261, 'precision': 0.2775, 'recall': 0.261, 'f_measure': 0.2658}
2024-12-06 00:34:19 - root - INFO: - Epoch:100/200===>loss=50.3449
{'AMI': 0.2837, 'NMI': 0.3076, 'ARI': 0.127, 'accuracy': 0.2614, 'precision': 0.2843, 'recall': 0.2614, 'f_measure': 0.2654}
2024-12-06 00:34:29 - root - INFO: - Epoch:200/200===>loss=49.5699
{'AMI': 0.281, 'NMI': 0.3048, 'ARI': 0.1299, 'accuracy': 0.2671, 'precision': 0.2663, 'recall': 0.2671, 'f_measure': 0.2651}
2024-12-06 00:34:39 - root - INFO: - Epoch:100/200===>loss=50.2875
{'AMI': 0.3094, 'NMI': 0.3324, 'ARI': 0.1527, 'accuracy': 0.2895, 'precision': 0.3037, 'recall': 0.2895, 'f_measure': 0.2908}
2024-12-06 00:34:49 - root - INFO: - Epoch:200/200===>loss=49.4835
{'AMI': 0.3066, 'NMI': 0.3296, 'ARI': 0.1535, 'accuracy': 0.2914, 'precision': 0.3016, 'recall': 0.2914, 'f_measure': 0.2924}
2024-12-06 00:34:59 - root - INFO: - Epoch:100/200===>loss=50.2769
{'AMI': 0.3022, 'NMI': 0.3254, 'ARI': 0.1496, 'accuracy': 0.2757, 'precision': 0.2849, 'recall': 0.2757, 'f_measure': 0.2755}
2024-12-06 00:35:08 - root - INFO: - Epoch:200/200===>loss=49.4865
{'AMI': 0.2981, 'NMI': 0.3214, 'ARI': 0.1493, 'accuracy': 0.2771, 'precision': 0.2825, 'recall': 0.2771, 'f_measure': 0.2769}
2024-12-06 00:35:18 - root - INFO: - Epoch:100/200===>loss=50.3362
{'AMI': 0.2848, 'NMI': 0.3085, 'ARI': 0.1235, 'accuracy': 0.2562, 'precision': 0.2769, 'recall': 0.2562, 'f_measure': 0.2623}
2024-12-06 00:35:28 - root - INFO: - Epoch:200/200===>loss=49.4733
{'AMI': 0.2881, 'NMI': 0.3117, 'ARI': 0.1269, 'accuracy': 0.26, 'precision': 0.2748, 'recall': 0.26, 'f_measure': 0.2641}
2024-12-06 00:35:28 - root - INFO: - --------------------Training over--------------------
2024-12-06 00:35:28 - root - INFO: - [0.261, 0.2671, 0.2914, 0.2771, 0.26]
2024-12-06 00:35:28 - root - INFO: - [0.3012, 0.3048, 0.3296, 0.3214, 0.3117]
2024-12-06 00:35:28 - root - INFO: - [0.1249, 0.1299, 0.1535, 0.1493, 0.1269]
2024-12-06 00:35:28 - root - INFO: -  
                     ACC 27.13 std 1.17
                     NMI 31.37 std 1.05 
                     ARI 13.69 std 1.20
2024-12-06 00:35:28 - root - INFO: - 27.13,1.17;31.37,1.05;13.69,1.2;
acc: 27.13 ,nmi: 31.37 ,ari: 13.69







