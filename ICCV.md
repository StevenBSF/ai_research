虚假链接改成噪声说法



#  Cross-view Content Consistency

由于同样的样本的representation在不同视角下可能存在语义上的共享信息和视图差异，即对应Eq.(2)中的content和style的因果建模。针对Multi-view clustering的任务而言，在保证所有视角对样本的内容达成一致的同时，允许每个视角保留各自的风格变化。已有的工作中(GCFAgg)对于跨视角信息使用了transformer架构式的global cross-view 融合，导致representation缺乏有效语义层面的过滤，无法保证融合的特征均对于聚类任务有意义。因此，有效一致化跨视角content representation的同时均衡且多样化 content 信息，并且保证不同视图下的风格的多样性成为难点。因此，我们引入了$\mathcal{L}_{\text{CrossView}}$损失，写为：

\begin{equation}
\label{eq:Crossloss}
\begin{split}
\mathcal{L}_{\text{CrossView}}  :=&  \sum\limits_{1 \leq k < k' \leq V } \mathbb{E}_{(\mathbf{Z}_c^k, \mathbf{Z}_c^{k'}) \sim p_{\mathbf{Z}_c^k, \mathbf{Z}_c^{k'}}} \left[ \left\| \mathbf{Z}_c^k - \mathbf{Z}_c^{k'} \right\|_2^2 \right]\\
 &- \sum^{V}_{v=1} \mathtt{Entropy}(\mathbf{Z}_c^{v}) - \sum^{V}_{v=1} \mathtt{Entropy}(\mathbf{Z}_s^{v}).
\end{split}
\end{equation}

其中第一项促使相同样本在不同视角下的内容特征尽可能相似，使得每个样本都能拥有一致的跨视角内容表示。该项对于聚类的鲁棒性至关重要：它确保了聚类（内容）是基于数据的本质特征，而非某个特定视角的特性。本质上，该项最大化了视角间的内容一致性，确保模型学到的内容表示仅包含跨视角共享的因子，而非特定视角的伪特征。从直觉上来看，确保如果两个数据点在一个视角中被归为同一簇，它们在其他视角中也会被分到相同的簇。$- \sum^{V}_{v=1} \mathtt{Entropy}(\mathbf{Z}_c^{v})$通过最大化每个视角的内容熵，该损失项鼓励均衡且多样化的聚类分配,防止该视角的内容表示不会塌缩至单一类别或无意义的结果。本质上，其使得内容表示具有较高的信息量，并确保簇是分离清晰、且具有实际意义的。$- \sum^{V}_{v=1} \mathtt{Entropy}(\mathbf{Z}_s^{v})$保证了内容表示 $\mathbf{Z}_c$ 只编码跨视角稳定的特征，而样本间的细微差异由风格表示 $\mathbf{Z}_s$ 负责捕捉。通过风格熵项，模型会自发地学习到内容信息是共享的，而风格信息是个性化的。