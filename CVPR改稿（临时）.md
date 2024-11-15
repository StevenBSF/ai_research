

- View dependency，是指对于在多视角聚类任务下由于不同视角的信息源存在独特的内容、噪声和偏差，导致模型过度依赖某个视角的聚类信息，使得特定视角的独特噪声或偏差可能被过度放大，并且忽略其他视角中的补充信息，进而导致各视角的差异信息无法有效整合，其所带来额外的互补信息无法得到充分挖掘，而无法综合不同视角特征，使得多视角数据优势难以体现。

- View dependency, refers to a phenomenon in multi-view clustering tasks where, due to unique content, noise, and biases within information source in each view, the model becomes overly reliant on the clustering information from some particular views. This dependency can lead to an amplification of specific view-dependent noise or bias while disregarding supplementary information from other views. Consequently, the model struggles to integrate the distinct information across views effectively, failing to fully exploit the complementary information embedded in the multi-view data. This limitation hinders the potential advantages of multi-view data and reduces the overall effectiveness of the clustering task.

- This issue arises because deep methods在进行跨视角融合的时候未能周全地考虑到视图之间的公共内容信息，将各个的视角特征投射到统一的特征空间会导致将所有视角对齐至some dominant views。这使得supplementary information from other views丢失。在real-world的setting下过于模型依赖于数据的某些特定视角，忽略数据的本身原貌，与multi-view的特性背道而离。

- To address this issue, we introduce causal mechanisms \cite{Scholkopf2019CausalityFM} into the deep multi-view clustering task, aiming to extract causality-based features and mitigate the challenges posed by view dependency. This approach filters out non-causal associations, allowing the model to concentrate on features that genuinely represent underlying patterns across views. Moreover, to reduce view dependency, we construct causal content-style features from both intra-view and cross-view perspectives. Unlike existing methods that focus solely on cross-view content, our approach captures and balances both shared content across views and unique content and style within each view, preventing over-reliance on dominant views.

- 介绍多视角的任务
- 阐述多视角特别是deep的方法的目前面临的问题
- 提出方法的动机，引入因果机制并且进行简单讲解
- 我们提出的方法的简单介绍



在多视角聚类中，**view-dependency** 是指模型过于依赖某一个视角（或多个特定视角），从而影响聚类效果的现象。具体来说，由于多视角数据来自不同的信息源，每个视角可能包含独特的内容、噪声或偏差，如果模型对某个视角的特征过度依赖，则可能导致以下问题：

1. **信息丢失**：模型过度依赖一个主视角，可能忽略其他视角中的补充信息，使得聚类结果不完整，无法充分利用多视角的多样性。
2. **噪声放大**：特定视角的独特噪声或偏差可能被过度放大，从而导致模型拟合不准确，影响聚类的准确性。
3. **互补信息融合困难**：各视角的差异可能带来额外的互补信息，但过于依赖一个视角会使模型难以有效地综合不同视角的特征，无法充分挖掘多视角数据的优势。



The dashed arrow表示statistical dependence。The solid arrow表示causal influence。在因果层面既有共同信息$c$又有视角风格信息$s$对于目标任务$Y^c$的影响，但是无关信息$u$不应当作用于$Y^c$。



In this paper, to more effectively model multi-view clustering from a causal perspective, we propose a novel model named \qi{Causality-based Multi-View Clustering (CausalMVC).} We introduce a Structural Causal Model tailored specifically for multi-view clustering tasks. Our model enables the extraction of clustering information from both \textit{content} and \textit{style}. We build causal \textit{content}-\textit{style}features from intra-view and cross-view perspectives. It effectively captures and balances the shared content across views, along with the unique content and style specific to each view, thus mitigating over-reliance on dominant views.



However, these methods often fall short in isolating genuine clustering signals from spurious patterns and view-specific noise due to statistical correlation. This reliance on correlated but non-informative features can 导致模型出现对某些视角中的特定噪声信号或伪模式产生过拟合。此外，模型可能在不同视角间建立spurious connections,从而误将不相关的数据点聚合在同一类中，降低了聚类的准确性。进而模型难以有效提取出真正的聚类结构，导致聚类效果的显著制约。

Meanwhile, view dependency, refers to a phenomenon in multi-view clustering tasks. In this phenomenon, each view contains unique content, noise, and biases within its information source. As a result, the model becomes overly reliant on the clustering information from certain dominant views. This, in turn, causes the model to struggle to integrate the distinct information across views effectively, failing to fully exploit the complementary information embedded in the multi-view data. This limitation hinders the potential advantages of multi-view data and reduces the overall effectiveness of the clustering task.



# 创新点

- We propose a novel multi-view clustering model based on causality mechanism. 我们针对多视角聚类任务设计了相应的Structural Causal Model to alleviate statistical spurious connections.
- 我们effectively mitigate view dependency.from intra-view and cross-view perspectives，我们挖掘了causal content and style for clustering.我们对于intra-view，在去除无关信息的基础上对齐了内容信息，并且对style进行差异化处理。我们对cross-view进行了内容consistency的约束。
- Building on content-style features, we establish a causally adaptive unified feature to enable contrastive learning tailored for multi-view clustering tasks,平衡各视角consistent content和差异化 style。

# 方法部分

- where Query Mapping表示对于当前view下所有包含的未经处理的信息的投射，Invariant \textit{noise} Mapping表示对于biases or noise信息的处理，\textit{Content} Value和\textit{Style} value分别表示content和style的信息的投射。我们通过先构建出视角内部所有信息的映射之后，再对所存在无关信息投射之后进行差分消除，所得到的结果再与对应的Value值进行加权计算，得到目标的causal feature。
- 我们提出的主要方法步骤由图2所示。对于输入数据，我们会先进行preliminary feature alignment并用重构损失加以约束，得到我们的初步特征，之后我们会将初步特征从intraview和crossview两个角度进行因果层面提取出各个视图下的content和style特征，对于得到的content和style特征，我们在构建 unified causal content-style features基础上进一步通过对比学习的方法 extract and supervise the clustering information 。
- 由于我们要对于输入的特征进行因果层面的分析以期望能够达到对于content、style、无关信息的解耦，我们需要对输入的特征和目标任务进行logical reformulation。

- 考虑到对于多视角特征对于视角内部的unique noise和不同视角下由于视角之间的差异造成的内容信息的差异，并且对于视角之间，模型可能过度强调某个主视角而导致难以有效整合互补信息，因此为了进一步解决这些现有的难点，我们从intra-view和cross-view两个角度考虑以下两个问题：
- Considering the unique noise within each view's features and the content discrepancies arising from differences across views, the model may overemphasize a dominant view, making it challenging to effectively integrate complementary information. To further address these existing challenges, we examine the following two aspects from both intra-view and cross-view perspectives:



- This is particularly challenging in multi-view clustering,  这是因为spurious connections会导致模型aggregate unrelated data points into the same cluster。



- According to the \textit{Common Cause Principle}, which states that \textit{if two events are correlated, there exists a common cause that explains the correlation, rather than one event directly causing the other}, 如 Fig 2所示，we reformulate $\mathbf{H}^v$ and $Y^C$ as follows:









- Here, the mean and standard deviation of $\mathbf{H}^{v}$ follow $\mathcal{N}(\mu(\mathbf{H}^{v}), \Sigma^2_\mu(\mathbf{H}^{v}))$ and $\mathcal{N}(\sigma(\mathbf{H}^{v}), \Sigma^2_\sigma(\mathbf{H}^{v}))$ respectively. The parameters $\beta(\mathbf{H}^{v})$ and $\gamma(\mathbf{H}^{v})$ represent the perturbed mean and standard deviation, and $\widetilde{\mathbf{H}}^{v} $ is the transformed noise-intervened feature.因此，$\widetilde{\mathbf{H}}^{v}$也可以从因果角度reformulate成$\widetilde{\mathbf{H}}^v := g(\tilde c^v, \tilde s^v,\tilde u^v)$.

- \textbf{Problem 1}: How can we ensure the consistency of the intra-view \textit{content} $c^v := p^v_{1:|c^v|}(\mathbf{H}^{v})$ and $\tilde c^v := p^v_{1:|c^v|}(\widetilde {\mathbf{H}}^{v})$, i.e., $c^v = \tilde c^v$,并且挖掘出intra-view的style$s^v:=p^v(\mathbf{H}^{v})_{(|c^v|+1):(|c^v+s^v|)}$?
- \textbf{Problem 2}: How can we ensure the consistency of the cross-view \textit{content} $c^k = p^k_{1:|c^k|}(\mathbf{H}^{k})$ and $c^{k'} = p^{k'}_{1:|c^{k'}|}(\mathbf{H}^{k'})$, i.e., $c^k = c^{k'}$, 并且保证视角之间\textit{style}$s^k$,$s^{k'}$的差异性，where $k,k' \in V$?



- 对于Problem 1，为了进行视角内的内容的consistency约束并且考虑real world下的视角内的差异，我们需要对于初步特征进行扰动。

- 为了处理视角内的内容特征和风格变化，我们需要对于在Eq(3)已有的因果形式化建模的基础上，进行相应的特征处理和损失约束。由于$\mathbf{Z}_c^v =\ p^v_{1:|c^v|}(\mathbf{H}^{v})$和$\mathbf{Z}_s^v =\ p^v_{(|c^v|+1):(|c^v+s^v|)}(\mathbf{H}^{v})$是logical structural mapping，要从理论转至model层面，我们需要相应过渡到深度学习可学习的结构映射网络。因此我们引入了基于差分思想的Dual content-style提取的特征投影$\texttt{DiffMapping}(\cdot)$。差分网络的motivation在于，考虑视角内部所有semantic information的基础上，去除潜在的noise和bias，进而各自同时捕捉content和style。对于目前已有的preliminary feature$\mathbf{Z}^{v}$ 和经过扰动的feature$\widetilde{\mathbf{Z}}^{v}$，我们构建如下差分网络以获得content feature$\mathbf{Z}_c^{v}$，$\widetilde{\mathbf{Z}}_c^{v}$和style feature$\mathbf{Z}_s^{v}$​ :

- To handle intra-view content features and style variations, we build on the causal formalization presented in Eq(3) by introducing specific feature processing and loss constraints. Given that $\mathbf{Z}_c^v =\ p^v_{1:|c^v|}(\mathbf{H}^{v})$ and $\mathbf{Z}_s^v =\ p^v_{(|c^v|+1):(|c^v+s^v|)}(\mathbf{H}^{v})$ represent logical structural mappings, transitioning from theoretical formalism to a model-level implementation requires a learnable deep structural mapping network. We therefore introduce a differential approach for dual content-style feature projection, denoted as $\texttt{DiffMapping}(\cdot)$. The motivation behind the differential network lies in its ability to capture content and style by considering all semantic information within the view, while removing potential noise and bias. For the preliminary feature $\mathbf{Z}^{v}$ and its perturbed counterpart $\widetilde{\mathbf{Z}}^{v}$, we construct the following differential network to obtain the content features $\mathbf{Z}_c^{v}$, $\widetilde{\mathbf{Z}}_c^{v}$ and style features $\mathbf{Z}_s^{v}$:

- For intra-view content alignment, we need to constrain the dual \textit{content-style} feature projection $\texttt{DiffMapping}(\cdot)$, ensuring that, regardless of whether the feature is perturbed or not, the \textit{content} remains consistent under the same view并且去除无关noise $u^v$​. Therefore, we formalize this constraint as:



- Considering that in we performed a content-style component analysis and deconstruction of the processed causal features, the key insight of causal mechanisms, as compared to other deep multi-view clustering methods, lies in the handling of features. Some previous works primarily focused on how to better leverage attention mechanisms or other MLP architectures to integrate and fuse semantic and label features across different views and align features between them. However, as previously discussed, black-box fusion mechanisms can lead to significant challenges on complex datasets due to strong statistical dependence $P(\mathbf{H}^v,Y^C)$, which 会导致有较强的statistical spurious connections and 明显的view dependency。 Therefore, building on the foundation laid before, we introduce a simple yet effective strategy for exploiting the complementary information. We incorporate learnable weight parameters $\{w^{1}_{\alpha},w^{2}_{\alpha},...,w^{v}_{\alpha} \}, v \in V$, to construct self-adaptive high-quality unified causal content-style features:







而目前多视角聚类任务下content-style信息挖掘的难点在于如何更好保证content consistency的前提下提取更加有用的聚类信息。

- 同时，$\widetilde{\mathbf{Z}}_c^{v}$也可通过$\widetilde{\mathbf{Z}}_c^{v}=\text{DiffMapping}(\widetilde{\mathbf{H}}_c^{v})$获得。

- 同时，为了保证视角内\textit{content}和\textit{style}的差异性，我们需要对其进行解耦。因此我们引入以下约束：

- 为了保证对于cross view情形下不同视角之间的\textit{style}的invariance，我们需要对于

  



$$\mathbf{Z}_c^v =&\ p^v_{1:|c^v|}(\mathbf{H}^{v}) \\
    =&\ \text{DiffMapping}(\mathbf{H}^v) \\
    =&\ \bigl( \mathtt{Softmax}\bigl(  \mathbf{H}^v \mathbf{W}_{Q_c} (\mathbf{H}^v \mathbf{W}_{K_c})^\top   \bigr) \\
    &\quad -\ \mathtt{Softmax}\bigl( \mathbf{H}^v \mathbf{W}_{Q_u}  ( \mathbf{H}^v \mathbf{W}_{K_u} )^\top \bigr) \bigr) \mathbf{H}^v \mathbf{W}_c^v $$









- （limitation1）虚假connection deep learning based 方法，我们建立causal-based建立因果连接
- （limitation2）在建立因果连接的基础上，已有方法对于某些view、dominant。。。，
- （）
- （limitation2原因）因为只考虑cross-view，忽略了其他view的聚类信息，
- （limitation2结果）从而导致聚类结果过拟合于某些试图并且存在性能抖动
- （我们对于limitation2改进的方法）我们在intra-view构建causal content-style feature
- （我们对于limitation2改进的原因）
- （limitation2改进的记过）

为了消除潜在的spurious connections（针对limitation1的方法动机），

我们通过causal-based方法构建基于多视角数据的Structural Causal Model （我们对于limitation1改进的方法）。

进而而达到消除biases or noise带来的影响（对于limitation1改进的结果）。

为了mitigating View Dependency（针对limitation1的方法动机），

我们考虑在cross-view角度挖掘聚类信息的基础上，进一步从intra-view角度构建causal content-style feature。我们根据这两个角度，不仅对于causal content进行对齐，而且对于causal style的进行差异性约束（我们对于limitation1改进的方法）。

进而达到适应性地权衡各个视图的common和有差别聚类信息，达到大幅缓解模型性能抖动的效果（对于limitation2改进的结果）。



For the first limitation, we establish causal-based connections between the data and the clustering task, leveraging  Structural Causal Model to infer essential connections（我们对于limitation2改进的方法） and thus alleviate the impact of biases or noise（limitation2改进的结果）. For the second limitation, we construct causal content-style features from an intra-view perspective while considering cross-view clustering information. This dual approach aligns causal content from both perspectives and simultaneously accounts for causal style discrepancies, enabling an adaptive balance between common and differential clustering information across views, thereby significantly alleviating performance jitter and effectively mitigating view dependency.

