

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





# 算法复杂度分析

- 在本节中，我们系统地分析了所提出的多视角聚类算法的计算复杂性。我们从Intra-view Content-Style Extraction、Cross-view Content Consistency、Causality-based Contrasive Learning三个阶段逐步分析。

- 在视角内内容与风格特征提取阶段，特征扰动步骤用于对初步特征引入微小随机变化，此过程涉及均值和标准差计算、噪声生成以及特征变换，其复杂度为 \(O(Nd)\)。由于复杂度对样本数 \(N\) 呈线性增长，这一部分计算成本相对较低。其次，DiffMapping()操作捕获样本之间的关联关系，以进一步解耦内容和风格特征。具体而言，该过程需要进行 \(N \times d\) 和 \(d \times N\) 矩阵乘法以计算权重，随后生成大小为 \(N \times N\) 的注意力矩阵并进行Softmax归一化操作，最终计算差异特征。由于涉及到 \(N\) 样本间的成对交互，矩阵乘法的复杂度为 \(O(N^2d)\)。此外，为了进一步约束内容和风格特征的解耦，算法通过正交性约束强制两类特征在向量空间中彼此独立。这一过程需要计算内容特征和风格特征矩阵的内积，即 \(N \times d\) 矩阵与 \(d \times N\) 矩阵的乘法，复杂度为 \(O(N^2d)\)。所以视角内内容与风格特征提取阶段的总复杂度由差分映射的二次复杂度主导。在单视角情况下，复杂度为 \(O(N^2d)\)，当扩展到 \(V\) 个视角时，总复杂度为 \(O(VN^2d)\)。

- 在跨视角内容一致性阶段，首先，特征映射用于生成跨视角一致性的标签，其复杂度为 \( O(Nd) \)。随后，通过计算所有视角对的内容一致性差异来完成跨视角对齐。由于视角对的数量为 \( \frac{V(V - 1)}{2} \)，因此该步骤的复杂度为 \( O(V^2Nd) \)。

- 在因果对比学习阶段，算法通过多个关键步骤进一步优化特征表示。首先，通过加权与拼接操作生成统一的特征表示，其复杂度为 \( O(VNd) \)。接着，计算样本间的相似度矩阵，其复杂度为 \( O(N^2d) \)。基于相似度矩阵构建伪标签图时，复杂度进一步上升至 \( O(N^2C) \)，其中 \( C \) 表示聚类类别数。最后，计算伪标签图上的对比损失，其复杂度为 \( O(N^2) \)。整体而言，对比学习阶段的单视角复杂度为 \( O(N^2d) \)，扩展到 \( V \) 个视角时，总复杂度为 \( O(VN^2d) \)。

  综合上述分析，算法的总体复杂度由视角内特征提取和因果对比学习阶段主导，均对样本数 \( N \) 呈现二次增长特性，因此总体复杂度为 \( O(VN^2d) \),而V通常远小于N，所以总体复杂度为O(N^2d).





# 证明

以下是提取的文字内容：

以下是提取的文字内容：

---

**Definition 4.1 (Block-identifiability).** We say that the true content partition \( c = f^{-1}(x)_{1:n_c} \) is block-identified by a function \( g : \mathcal{X} \to \mathcal{Z} \) if the inferred content partition \( \hat{c} = g(x)_{1:n_c} \) contains all and only information about \( c \), i.e., if there exists an invertible function \( h : \mathbb{R}^{n_c} \to \mathbb{R}^{n_c} \) such that \( \hat{c} = h(c) \).

Theorem 4.2 (Identifying content with a generative model). Consider the data generating process described in § 3, i.e., the pairs (x, x̃) of original and augmented views are generated according to (2) and (3) with \( p_{z̃|z} \) as defined in Assumptions 3.1 and 3.2. Assume further that:

（i）\( f : Z \to X \) is smooth and invertible with smooth inverse (i.e., a diffeomorphism);

  (ii)\( p_z \) is a smooth, continuous density on \( Z \) with \( p_z(z) > 0 \) almost everywhere;

  (iii) For any \( l \in \{1, ..., n_s\} \), there exists \( A \subseteq \{1, ..., n_s\} \) such that \( l \in A \), \( p_A(A) > 0 \); \( p_{s_A|s_A} \) is smooth with respect to both \( s_A \) and \( \tilde{s}_A \); and for any \( s_A \), \( p_{\tilde{s}_A|s_A}(\cdot | s_A) > 0 \) in some open, non-empty subset containing \( s_A \).

If, for a given \( n_s \) (\( 1 \leq n_s < n \)), a generative model \( (\hat{p}_z, \hat{p}_A, \hat{p}_{\tilde{s}_A | s_A}, \hat{f}) \) assumes the same generative process (§ 3), satisfies the above assumptions (i)-(iii), and matches the data likelihood,
\[
p_{x, \tilde{x}}(x, \tilde{x}) = \hat{p}_{x, \tilde{x}}(x, \tilde{x}), \quad \forall (x, \tilde{x}) \in \mathcal{X} \times \mathcal{X},
\]
then it block-identifies the true content variables via \( g = \hat{f}^{-1} \) in the sense of Defn. 4.1.



**Proof.** The proof consists of two main steps.

In the first step, we use assumption (i) and the matching likelihoods to show that the representation \( \hat{z} = g(x) \) extracted by \( g = \hat{f}^{-1} \) is related to the true latent \( z \) by a smooth invertible mapping \( h \), and that \( \hat{z} \) must satisfy invariance across \( (x, \tilde{x}) \) in the first \( n_c \) (content) components almost surely (a.s.) with respect to (w.r.t.) the true generative process.

In the second step, we then use assumptions (ii) and (iii) to prove (by contradiction) that \( \hat{c} := \hat{z}_{1:n_c} = h(z)_{1:n_c} \) can, in fact, only depend on the true content \( c \) and not on the true style \( s \), for otherwise the invariance established in the first step would have been violated with probability greater than zero.

To provide some further intuition for the second step, the assumed generative process implies that \( (c, s, \tilde{s}) | A \) is constrained to take values (a.s.) in a subspace \( \mathcal{R} \) of \( C \times S \times \tilde{S} \) of dimension \( n_c + n_s + |A| \) (as opposed to dimension \( n_c + 2n_s \) for \( C \times S \times \tilde{S} \)). In this context, assumption (iii) implies that \( (c, s, \tilde{s}) | A \) has a density with respect to a measure on this subspace equivalent to the Lebesgue measure on \( \mathbb{R}^{n_c + n_s + |A|} \). This equivalence implies, in particular, that this "subspace measure" is strictly positive: it takes strictly positive values on open sets of \( \mathcal{R} \) seen as a topological subspace of \( C \times S \times \tilde{S} \). These open sets are defined by the induced topology: they are the intersection of the open sets of \( C \times S \times \tilde{S} \) with \( \mathcal{R} \). An open set \( B \) of \( \mathcal{V} \) on which \( p(c, s, \tilde{s} | A) > 0 \) then satisfies \( P(B | A) > 0 \). We look for such an open set to prove our result.

**Step 1.** From the assumed data generating process described in § 3—in particular, from the form of the model conditional \( \hat{p}_{\tilde{z} | z} \) described in Assumptions 3.1 and 3.2—it follows that:
\[
g(x)_{1:n_c} = g(\tilde{x})_{1:n_c} \tag{6}
\]
a.s., i.e., with probability one, w.r.t. the model distribution \( \hat{p}_{x, \tilde{x}} \).

Due to the assumption of matching likelihoods, the invariance in (6) must also hold (a.s.) w.r.t. the true data distribution \( p_{x, \tilde{x}} \).

Next, since \( f, \hat{f} : \mathcal{Z} \to \mathcal{X} \) are smooth and invertible functions by assumption (i), there exists a smooth and invertible function \( h = g \circ f : \mathcal{Z} \to \mathcal{Z} \) such that:
\[
g = h \circ f^{-1}. \tag{7}
\]

Substituting (7) into (6), we obtain (a.s. w.r.t. \( p \)):
\[
\hat{c} := \hat{z}_{1:n_c} = g(x)_{1:n_c} = h(f^{-1}(x))_{1:n_c} = h(f^{-1}(\tilde{x}))_{1:n_c}. \tag{8}
\]

Substituting \( z = f^{-1}(x) \) and \( \tilde{z} = f^{-1}(\tilde{x}) \) into (8), we obtain (a.s. w.r.t. \( p \)):
\[
\hat{c} = h(z)_{1:n_c} = h(\tilde{z})_{1:n_c}. \tag{9}
\]

It remains to show that \( h(\cdot)_{1:n_c} \) can only be a function of \( c \), i.e., does not depend on any other (style) dimension of \( z = (c, s) \).

**Step 2.** Suppose for a contradiction that \( h_c(c, s) := h(c, s)_{1:n_c} = h(z)_{1:n_c} \) depends on some component of the style variable \( s \):
\[
\exists l \in \{1, ..., n_s\}, (c^*, s^*) \in C \times S, \quad \text{s.t.} \quad \frac{\partial h_c}{\partial s_l}(c^*, s^*) \neq 0, \tag{10}
\]
that is, we assume that the partial derivative of \( h_c \) w.r.t. some style variable \( s_l \) is non-zero at some point \( z^* = (c^*, s^*) \in \mathcal{Z} = C \times S \).

Since \( h \) is smooth, so is \( h_c \). Therefore, \( h_c \) has continuous (first) partial derivatives.

By continuity of the partial derivative,
\[
\frac{\partial h_c}{\partial s_l}
\]
must be non-zero in a neighbourhood of \( (c^*, s^*) \), i.e.,
\[
\exists \eta > 0 \quad \text{s.t.} \quad s_l \mapsto h_c(c^*, (s_{-l}^*, s_l)) \quad \text{is strictly monotonic on} \quad (s_l^* - \eta, s_l^* + \eta). \tag{11}
\]
where \( s_{-l} \in S_{-l} \) denotes the vector of remaining style variables except \( s_l \).

Next, define the auxiliary function \( \psi : C \times S \times S \to \mathbb{R}_{\geq 0} \) as follows:
\[
\psi(c, s, \tilde{s}) := |h_c(c, s) - h_c(c, \tilde{s})| \geq 0. \tag{12}
\]
To obtain a contradiction to the invariance condition (9) from Step 1 under assumption (10), it remains to show that \( \psi \) from (12) is strictly positive with probability greater than zero (w.r.t. \( p \)).

First, the strict monotonicity from (11) implies that:
\[
\psi(c^*, (s_{-l}^*, s_l), (s_{-l}^*, \tilde{s}_l)) > 0, \quad \forall (s_l, \tilde{s}_l) \in (s_l^* - \eta, s_l^* + \eta) \times (s_l^* - \eta, s_l^* + \eta). \tag{13}
\]

Note that in order to obtain the strict inequality in (13), it is important that \( s_l \) and \( \tilde{s}_l \) take values in disjoint open subsets of the interval \( (s_l^* - \eta, s_l^* + \eta) \) from (11).

Since \( \psi \) is a composition of continuous functions (absolute value of the difference of two continuous functions), \( \psi \) is continuous.

Consider the open set \( \mathbb{R}_{>0} \), and recall that, under a continuous function, pre-images (or inverse images) of open sets are always open.

Applied to the continuous function \( \psi \), this pre-image corresponds to an open set:
\[
U \subseteq C \times S \times S \tag{14}
\]
in the domain of \( \psi \) on which \( \psi \) is strictly positive.

Moreover, due to (13):
\[
\{c^*\} \times \left(\{s_{-l}^*\} \times (s_l^*, s_l^* + \eta)\right) \times \left(\{s_{-l}^*\} \times (s_l^* - \eta, s_l^*)\right) \subseteq U. \tag{15}
\]
so \( U \) is non-empty.

Next, by assumption (iii), there exists at least one subset \( A \subseteq \{1, ..., n_s\} \) of changing style variables such that \( l \in A \) and \( p_A(A) > 0 \); pick one such subset and call it \( A \).

Then, also by assumption (iii), for any \( s_A \in S_A \), there is an open subset \( \mathcal{O}(s_A) \subseteq S_A \) containing \( s_A \), such that \( p_{\tilde{s}_A | s_A}(\cdot | s_A) > 0 \) within \( \mathcal{O}(s_A) \).

Define the following space:
\[
\mathcal{R}_A := \{(s_A, \tilde{s}_A) : s_A \in S_A, \tilde{s}_A \in \mathcal{O}(s_A)\}, \tag{16}
\]
and, recalling that \( A^c = \{1, ..., n_s\} \setminus A \) denotes the complement of \( A \), define:
\[
\mathcal{R} := C \times S_{A^c} \times \mathcal{R}_A, \tag{17}
\]
which is a topological subspace of \( C \times S \times S \).

By assumptions (ii) and (iii), \( p_z \) is smooth and fully supported, and \( p_{\tilde{s}_A | s_A}(\cdot | s_A) \) is smooth and fully supported on \( \mathcal{O}(s_A) \) for any \( s_A \in S_A \). Therefore, the measure \( \mu_{(c, s_{A^c}, s_A, \tilde{s}_A) | A} \) has fully supported, strictly-positive density on \( \mathcal{R} \) w.r.t. a strictly positive measure on \( \mathcal{R} \). In other words, \( p_z \times p_{\tilde{s}_A | s_A} \) is fully supported (i.e., strictly positive) on \( \mathcal{R} \).

Now consider the intersection \( U \cap \mathcal{R} \) of the open set \( U \) with the topological subspace \( \mathcal{R} \).

Since \( U \) is open, by the definition of topological subspaces, the intersection \( U \cap \mathcal{R} \subseteq \mathcal{R} \) is open in \( \mathcal{R} \), (and thus has the same dimension as \( \mathcal{R} \) if non-empty).

Moreover, since \( \mathcal{O}(s_A^*) \) is open containing \( s_A^* \), there exists \( \eta' > 0 \) such that \( \{s_{-l}^*\} \times (s_l^* - \eta', s_l^*) \subseteq \mathcal{O}(s_A^*) \). Thus, for \( \eta'' = \min(\eta, \eta') > 0 \),
\[
\{c^*\} \times \{s_{-l}^*\} \times (\{s_A^*\} \times (s_l^*, s_l^* + \eta)) \times (\{s_A^*\} \times (s_l^* - \eta'', s_l^*)) \subseteq \mathcal{R}. \tag{18}
\]

In particular, this implies that:
\[
\{c^*\} \times (\{s_{-l}^*\} \times (s_l^*, s_l^* + \eta)) \times (\{s_{-l}^*\} \times (s_l^* - \eta'', s_l^*)) \subseteq \mathcal{R}. \tag{19}
\]

Now, since \( \eta'' < \eta \), the LHS of (19) is also in \( U \) according to (15), so the intersection \( U \cap \mathcal{R} \) is non-empty.

In summary, the intersection \( U \cap \mathcal{R} \subseteq \mathcal{R} \):

- is non-empty (since both \( U \) and \( \mathcal{R} \) contain the LHS of (15));
- is an open subset of the topological subspace \( \mathcal{R} \) of \( C \times S \times S \) (since it is the intersection of an open set, \( U \), with \( \mathcal{R} \));
- satisfies \( \psi > 0 \) (since this holds for all of \( U \));
- is fully supported w.r.t. the generative process (since this holds for all of \( \mathcal{R} \)).

As a consequence,
\[
\mathbb{P}(\psi(c, s, \tilde{s}) > 0 | A) \geq \mathbb{P}(U \cap \mathcal{R}) > 0, \tag{20}
\]
where \( \mathbb{P} \) denotes probability w.r.t. the true generative process \( p \).

Since \( p_A(A) > 0 \), this is a contradiction to the invariance (9) from Step 1.

Hence, assumption (10) cannot hold, i.e., \( h_c(c, s) \) does not depend on any style variable \( s_l \). It is thus only a function of \( c \), i.e., \( \hat{c} = h_c(c) \).

Finally, smoothness and invertibility of \( h_c : C \to C \) follow from smoothness and invertibility of \( h \), as established in Step 1.

This concludes the proof that \( \hat{c} \) is related to the true content \( c \) via a smooth invertible mapping.





---

Theorem 4.3 (Identifying content with an invertible encoder). Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let \( g : \mathcal{X} \to \mathcal{Z} \) be any smooth and invertible function which minimizes the following functional:

\[
\mathcal{L}_{\text{Align}}(g) := \mathbb{E}_{(x, \tilde{x}) \sim p_{x, \tilde{x}}} \left[ \left\| g(x)_{1:n_c} - g(\tilde{x})_{1:n_c} \right\|_2^2 \right] \quad (4)
\]

Then \( g \) block-identifies the true content variables in the sense of Definition 4.1.

---

**Theorem 4.4 (Identifying content with discriminative learning and a non-invertible encoder).** Assume the same data generating process (§ 3) and conditions (i)-(iv) as in Thm. 4.2. Let \( g : \mathcal{X} \to (0, 1)^{n_c} \) be any smooth function which minimizes the following functional:

\[
\mathcal{L}_{\text{AlignMaxEnt}}(g) := \mathbb{E}_{(x, \tilde{x}) \sim p_{x, \tilde{x}}} \left[ \left\| g(x) - g(\tilde{x}) \right\|_2^2 \right] - H(g(x)) \quad (5)
\]

where \( H(\cdot) \) denotes the differential entropy of the random variable \( g(x) \) taking values in \( (0, 1)^{n_c} \).

Then \( g \) block-identifies the true content variables in the sense of Defn. 4.1.









\textbf{Proposition 1} (Proposition 5 of Zimmermann et al. (2021)). Let $\mathcal{M}, \mathcal{N}$ be simply connected and oriented $C^1$ manifolds without boundaries and $h : \mathcal{M} \to \mathcal{N}$ be a differentiable map. Further, let the random variable $\mathbf{z} \in \mathcal{M}$ be distributed according to $\mathbf{z} \sim p(\mathbf{z})$ for a regular density function $p$, i.e., $0 < p < \infty$. \textbf{If the push-forward $p_{\# h}(\mathbf{z})$ through $h$ is also a regular density, i.e., $0 < p_{\# h} < \infty$, then $h$ is a bijection.} \vspace{1em} \textbf{Theorem 3.2} (Identifiability from a Set of Views). Consider a set of views $\mathbf{x}_V$ satisfying Asm. 2.1, and let $\mathcal{G}$ be a set of content encoders (Defn. 3.1) that minimizes the following objective \[ \mathcal{L}(\mathcal{G}) = \sum_{k < k' \in V} \mathbb{E} \left[\|g_k(\mathbf{x}_k) - g_{k'}(\mathbf{x}_{k'})\|_2\right] - \sum_{k \in V} H(g_k(\mathbf{x}_k)), \] where the expectation is taken w.r.t. $p(\mathbf{x}_V)$ and $H(\cdot)$ denotes differential entropy. \textbf{Then the shared content variable $\mathbf{z}_C := \{\mathbf{z}_j : j \in C\}$ is block-identified (Defn. 2.3) by $g_k \in \mathcal{G}$ for any $k \in V$.}









# 视频demo稿子

- Multi-view clustering is a task that uses different types of features from multiple views or data sources to find patterns and relationships in the data, while combining information from each view to improve clustering results. A central challenge in multi-view clustering involves effectively utilizing the semantic information from different views to self-supervisedly partition the data into distinct and unrelated groups. Recent works are primarily based on deep learning, which, despite their strengths, have notable limitations. First, they often rely on correlated patterns from a statistical perspective, which might be spurious connections. Second, due to variations across views, these models may overemphasize some dominant views or struggle to integrate complementary information effectively, resulting in view dependency. In real-world settings, this approach causes models to overly rely on certain specific views of the data, disregarding the original nature of the multi-view data and ultimately diverging from the essence of multi-view learning.
- 如图所示，c、s、u表示causal factors。c表示content，s表示style，u表示noise。
- 对于图中的例子，content表示共同的聚类信息，比如Dog，对于第一个view中的style可以表示为real-world，以及狗的品种Golden Retriever，对于第二个view中的style可以表示为Cartoon-style以及狗的品种Poodle。直观上理解而言，他们各自保留的风格信息对聚类仍保留一定语义信息，因此需要对其进行提取语义特征。
- 接下来请看我们提出的Causal Structural Model for Multi-view Clustering。
- 为了消除statistical spurious connections，我们需要进行transition from the statistical dependence $P(\mathbf{H}^v,Y^C)$ between the input and clustering clusters towards minimizing the gap between $P(\mathbf{H}^v)$ and $P(\mathbf{H}^v|Y^C)$, \ie,
  \begin{equation}
  \label{eq:minP}
     \min \sum_{v=1}^V \|P(\mathbf{H}^v) - P(\mathbf{H}^v|Y^C) \|,
  \end{equation}
  where $Y^C$ denotes the clustering clusters.
- To eliminate statistical spurious connections, we aim to transition from the statistical dependence between P of H superscript v and Y superscript C —representing the input features and clustering assignments—towards minimizing the difference between P of H superscript v and P of H superscript v given Y superscript C. Specifically, this is achieved by minimizing the sum of the distances across all views, expressed as 右侧的公式。 Here, Y superscript C refers to the clustering assignments. 
- 从任意两个视角回到我们的所有视角的情形，可以看到对于所有视角我们都进行构建Structual Causal Model。
- 在建立好对于Multi-View Clustering这个任务的Structual Causal Model之后，我们引入总体方法的模型框架。我们的方法整体分为四个部分，其中我们的重点阐述对于Intra-view Content-Style Extraction、Cross-view Content Consistency和Causality-based Contrasive Learning这三部分的解释。
- 对于Intra-view Content-Style Extraction，我们将初步获得的特征 H 进行视角内的Gaussian Perturbation，用以获得Perturbed H。

- 之前的SCM是一种逻辑上的建模。为了实现 transitioning from theoretical formalism to a model-level implementation， 我们需要构建 a learnable deep structural mapping network. We therefore introduce a differential approach for dual content-style feature projection, denoted as Differential Mapping. 
- 在Cross-view Content Consistency阶段， we adopt the approach of using causal content labels to supervise cross-view content consistency. Specifically, we construct the projection function zeta to obtain the label mathbf Y^v 
-  the key advantage of causal mechanisms, compared to other deep multi-view clustering methods, lies in the handling of features.所以我们from intra-view and cross-view perspectives提取出causal content and style feature之后，we 能够propose a simple yet effective strategy to exploit complementary information by introducing learnable weight parameters  to construct self-adaptive, high-quality unified causal content-style features U。并且我们依次根据Causally unified feature U构建相应的similarity matrix和pseudo label graph用于聚类监督。
- 在实验方面，Across all nine datasets, our model outperforms all other methods in terms of all three metrics. Notably, for more challenging datasets such as NUS wide, and Caltech-all, our model consistently achieves superior performance across all three metrics. This demonstrates that for more difficult tasks, our model effectively extracts the most essential content features while eliminating irrelevant information, thus enabling a causal understanding from intra-view and cross-view perspectives. It is also noteworthy that for the Caltech datasets, as the number of views increases, CausalMVC shows steady improvements across all three metrics.
- We propose a novel multi-view clustering model CausalMVC, which introduces causal inference in multi-view clustering. In the Causal Content-Style Feature stage, we analyze multi-view features using a causal mechanism from both intra-view and cross-view perspectives, aiming to achieve consistency in clustering content within and across views while preserving style differentiation. At the Causality-based Contrastive Learning stage, we construct similarity and pseudo-label graphs for contrastive learning by utilizing an adaptive unified feature composed of multi-view causal content-style features. Experimental results validate the advantages of our method over existing multi-view clustering methods.
- 目前Deep Multi-View Clustering的方法，如MFLVC, DealMVC, GCFAgg,对于样本数$N$的复杂度均在$O(N^2)$,我们的方法也保持相同的复杂度$O(N^2)$.

