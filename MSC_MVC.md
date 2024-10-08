## Abstract

​	Multi-view clustering aims to extract effective semantic information from multiple views to obtain better features for clustering. However, most current methods are limited to capturing semantic features and constructing pseudo-labels to capture clustering information, while neglecting the differences among different views and the further extraction of common semantic features. As a result, for some more challenging task data, the presence of irrelevant information and differences in each view makes it difficult to effectively assess the actual semantic quality of different views, leading to insufficient extraction of the common semantic information across views. To solve the above problems, our goal is to extract causal semantic features and generalize the inputs of different views to obtain content that truly lacks view dependency. The challenge lies in how to handle the inputs, extract causal essential features from different views, and apply them to the task of multi-view clustering. To address this, we propose a novel model named **M**ulti-**S**tage **C**ausal Feature Learning for **M**ulti-**V**iew **C**lustering(MSC-MVC).We introduce a causal content label that enables the alignment of content features from different perspectives, thereby eliminating view dependency,and based on this,we assess view quality based on the composition of content features across different views. Additionally, we perform feature augmentation through Gaussian distribution sampling, using similarity graph constraints to mitigate the impact of differences between views. By leveraging causal content features, we can further construct pseudo-label graphs to supervise clustering information. Extensive experiments on seven datasets are performed to illustrate the effectiveness and efficiency of the proposed model.

## Introduction

在过去几十年里，在机器学习和数据挖掘领域里Multi-View Clustering一直是一项火热并且长期处于关注焦点的任务。而多视角聚类的关键之处在于如何充分利用不同视图的有效语义信息将数据自监督地划分为一些各不相关地组别之中。对此，现阶段的对于多视角聚类这一任务的处理方法大致可以分为：基于non-negative matrix factorization,基于subspace learning, graph-based learning.

在这些方法中，特别是graph-based method虽然能够在较多的数据集上通过计算相似图和引入对比学习的方法来对齐latent representation获得了较为满意的结果，但是对于更深、更本质的层次的特征未能得到很好提取，从而导致了对于某些较难的数据集上出现了相较于其他方法在某些指标上面大幅崩溃的结果。For instance，DealMVC 通过结合对比学习和注意力机制融合不同视图的特征，pull close不同视图下不同样本以期望获得更集中的特征表示，在一些视图差异不大的一些数据集上获得了较好的结果，但是对于在一些视图差异较大的数据集上则出现了难以获得理想结果的情况，这是因为对于不同视图各自特征本身的内容和差异未能得到充分的挖掘，而弱化了模型的泛化能力。

为了解决这个问题，我们创新性地将因果机制引入graph-based multi-view clustering这个任务上，以期能够挖掘representation更为本质的特征并且减弱视图依赖性，从而增强各个视图之间的可解释性(identity)和泛化能力。换句话说，我们通过构建新的模型实现了形式化的、更逐步挖掘有效内容信息而非其他无关信息。解决这个问题的难点仍然在于，如何识别不同视图各自的有效特征并且解决视图依赖性。为了方便问题的描述，我们沿用yon中"content-style"的术语。简单来说，例如对于真实世界的图片视角的金毛犬和卡通风格的图片视角下的贵宾犬，由于这两个样本都有相同的狗这一个品种而可形成簇，因此"狗"可以作为因果推断中的内容，而真实世界与卡通风格、金毛犬和贵宾犬这两个品种则可成为"风格"。先前的其他任务的研究中，CIRL将风格视为无关因素而尽可能地保留内容因素而提升泛化能力。由于任务的不同，我们的研究中发现，对于"风格"中的某些因素仍然可能保留某些聚类信息。以刚才的例子，风格中的金毛犬和贵宾犬这两个品种仍保留一定的聚类信息上的相关性，而我们的模型能保证同时挖掘"content"和"style"中的聚类信息。

Moreover，我们分多阶段（multi-stage）提取特征，并且在causal "content-style"的基础上构建图对比学习网络。

## Related Work

### Graph-based MVC

In recent years，越来越多的学者关注到gragh-based method在MVC中的出色表现。考虑到多视图数据的异构性质，Graph-based方法可以灵活地处理异构信息，通过构建不同视图的图结构来综合考虑不同视图的数据，并且通过图的结构信息增强对于多视图聚类结果的鲁棒性和一致性。其中autoencoder被广泛地使用到图聚类中。更具体而言，对于来自$v$-view地输入数据${X^1,X^2,...,X^v}$，经过encoder层$E^v()$和decoder层$D^v()$​得到重构数据，对应地重构损失为：

### Causal Mechanism

当我们谈论到Causal Mechanism，我们不妨先回顾到machine learning以及Statistical Learning Theory。根据Statistical Learning Theory，我们知道对于目前很多方法都满足Statistical Dependence关系，在统计学习理论之下，统计依赖关系是指两个或多个变量之间存在某种统计上的关联，即一个变量的信息可以用来推测另一个变量的信息，我们形式化的表述为$f:X→Y$。但是随着目前各个领域和任务模型逐渐出现瓶颈，比如对于对于一组数据，模型虽然能识别出鞋码与数学成绩具有统计上的相关性，但事实告诉我们其背后共同作用的潜在因素可能是年龄，换句话说，虽然统计学习理论可以帮助我们找到变量之间的关联，但它并不能解释这些关联背后的因果机制。

而如果从Structural causal models对于这个问题建模，则有对于原本的$X \rightarrow Y$ ,遵循Common Cause Principle，应有$X \leftarrow Z \rightarrow Y$,这是因为我们将潜在因素的影响考虑在内。对于causal representation learning,(Scholkopf et al. 2016; Gresele et al. 2020; von Kugelgen et al. 2021)等人展开了详细的工作;(Liu et al. 2024)受causal inference的启发，将因果机制迁移到Text-Based Person Retrieval这个任务中。而具体到对于MVC这个任务上的建模，我们在后续的Methodology中具体展开。



From the perspective of Structural Causal Models, modeling this problem suggests that for the original relationship $X \rightarrow Y$ , and following the Common Cause Principle, we should instead have $X \leftarrow Z \rightarrow Y$. This is because we take into account the influence of latent factors.For causal representation learning, detailed work has been conducted by scholars such as Schölkopf et al. (2016), Gresele et al. (2020), and von Kügelgen et al. (2021). Inspired by causal inference, Liu et al. (2024) have extended causal mechanisms to the task of Text-Based Person Retrieval. Specifically, for modeling the task of multi-view clustering, we will elaborate further in the subsequent Methodology section.

## methodology

### Stage 1

对于多视角聚类任务而言，很多工作的实验验证了autoencoder对于初步提取不同视角的有效信息和去除冗余信息起到明显作用，因此我们的模型中同样吸取了先前的工作经验，并将其作为我们的第一阶段的特征学习。考虑对于输入数据$\{ X^v \in \mathbb{R}^{N \times D_v} \}^{V}_{v=1}$,我们通过encoder层获得初步特征$H^v = E^v(X^v ; \theta^v)\in \mathbb{R}^{N \times D_v}$，where $\theta^v$记作encoder层参数，并且将初步特征进一步经过decoder层获得重构output$\widetilde{X}^v = D^v(H^v ; \psi^v)\in \mathbb{R}^{N \times D_v}$，where $\psi^v$​​记作decoder层参数。重构损失我们采用Eq(1).

### Stage 2

对于Stage 1所利用的autoencoder提取了preliminary features，具有了一定程度上聚类任务所需要的有效信息，不过保留的噪声仍然很多。特别是对于multi-view clustering，我们如何能够让模型鉴别有效信息？而我们的目的之一是为了让feature更加的identify，为了达到这个目的，我们引入因果推断，即从统计层面的输入到聚类簇之间的统计依赖性$P(H^v,C)$过渡至最小化$P(H^v)$和$P(H^v|C)$之间的差距，即$min \|P(H^v) - P(H^v|C) \|$,where $C$表示聚类簇。

而根据Common Cause Principle，我们对于$H^v$和$C$重新形式化为：

$$H^v := g(Z^v) = g(\delta(c^v,s^v,u^v))$$

$$C := h(Z^v) = h(\zeta(c^v,s^v))$$

where  $g()$和$h()$为view-specific function，$\delta()$和$\zeta()$为 noise corruption function，$c^v$、$s^v$、$u^v$分别为各个视图下的content、style、view-specific的冗余信息,且$c^v$、$s^v$、$u^v$满足$c^v \perp\!\!\!\perp s^v \perp\!\!\!\perp u^v$。而事实上对于$c^v$、$s^v$、$u^v$而言在我们目前的多视角聚类任务是未给定的，我们需要依次构建函数$\{p^v:\mathcal{H}^v \rightarrow (0,1)^{|c^v|} \}$和$\{ q^v:\mathcal{H}^v \rightarrow (0,1)^{|s^v|} \}$,where $p^v$、$q^v$ 分别记作content、style投影函数，$\mathcal{H}^v$、$\mathcal{Z}^v$分别为preliminary features $H^v$、$Z^v$的 latent sub-space，$\mathcal{Z}^v$为preliminary features $Z^v$的 latent sub-space,$|c^v|$、$|s^v|$分别记作content、style的维度大小。

那么，多个视角下latent space造成的域差异，i.e.,对于冗余信息$u^v$我们如何处理呢？我们引入高斯分布采样的策略，将$\mathcal{H}^v$下的$H^v$进行域变换：

$$\beta(H^{v}) = \mu(H^{v}) + \epsilon_\mu \Sigma_\mu(H^{v}), \quad \epsilon_\mu \sim \mathcal{N}(0, 1),$$

$$\gamma(H^{v}) = \sigma(H^{v}) + \epsilon_\sigma \Sigma_\sigma(H^{v}), \quad \epsilon_\sigma \sim \mathcal{N}(0, 1),$$​

$$\widetilde H^{v} = \gamma(H^{v}) \left( \frac{H^{v} - \mu(H^{v})}{\sigma(H^{v})} \right) + \beta(H^{v}).$$

where，$H^{v}$的均值和标准差分别遵循$\mathcal{N}(\mu(H^{v}), \Sigma^2_\mu(H^{v}))$、$\mathcal{N}(\sigma(H^{v}), \Sigma^2_\sigma(H^{v}))$，而$\beta(H^{v})$和$\gamma(H^{v})$分别表示为经过$\epsilon_\mu$扰动后的均值和标准差，而$\widetilde H^{v} $​为转变后的noise-intervened feature,因此有：

$$\widetilde H^v := g(Z^v) = g(\delta(\widetilde c^v, \widetilde s^v,\widetilde u^v))$$

至此，我们进一步需要解决两个问题：

- Problem1:如何保证视图内部内容$c^v = p^v(H^{v})$和$\widetilde c^v = p^{v}(\widetilde H^{v})$的uniformity,换句话说，$c^v = \widetilde c^v$?
- Problem2:如何保证cross-view的内容$ c^k = p^k(H^{k})$和$c^{k'} = p^{k'}(H^{k'})$的uniformity,换句话说，$c^v = \widetilde c^v$,where $k,k' \in V$?

对于problem1，我们需要对于content投影函数$p^v$进行约束，即对于不论是未经扰动的$H^{v}$还是$\widetilde H^{v} $，经过同一个视角下的$p^v$都应满足内容不变，对此构造一致损失：

$$   \arg \min_{p^v}  \mathbb{E}_{(\mathbf{H}^v, \widetilde{\mathbf{H}}^v) \sim p_{\mathbf{H}^v, \widetilde{\mathbf{H}}^v}} \left[ \left\| p^v(\mathbf{H}^v) - p^v(\widetilde{\mathbf{H}}^v) \right\|_2^2 \right]$$

而为了保证content feature确实与multi-view clustering的任务有关，我们应当保证对于所提取的$c^v = p^v(H^{v})$与自监督信号聚类信息$C$的互信息$I(c^v,C)$最大化，形式化为：

$$\arg \max_{p^v} I(p^v(\mathbf{H}^{v}),C) = \arg \max_{p^v} I(p^v(\mathbf{H}^{v}),p^v(\mathbf{H}^{v})) = \arg \max_{p^v} H(p^v(\mathbf{H}^{v}))$$​

where $H()$表示熵。

统一起来，我们有损失函数：

$$\mathcal{L}_{\text{IntraViewAlign}}(p^v) :=  \mathbb{E}_{(\mathbf{H}^v, \widetilde{\mathbf{H}}^v) \sim p_{\mathbf{H}^v, \widetilde{\mathbf{H}}^v}} \left[ \left\| p^v(\mathbf{H}^v) - p^v(\widetilde{\mathbf{H}}^v) \right\|_2^2 \right] - H(p^v(\mathbf{H}^{v}))$$​

$\mathcal{L}_{\text{IntraViewAlign}}$的收敛性证明会在附录中给出。

对于problem2，我们采用causal content label的方式，监督cross-view的内容一致性，即构建投影函数 $\{ Mix^v: \mathcal{Z}^v \rightarrow \mathcal{Y}^{v} \}$, 且$\mathcal{Y}^{v} \subseteq \mathbb{R}^{V \times N \times |c^v|}  $,获得标签$Y^v = Mix(Z^v) = Mix(p(\mathbf{H}^{v}))$。类似对于Problem1对于问题的建模，但是不同的是对于标签$Y^k = [y^k_1,y^k_2,...,y^k_v ],k \in V,y^k_i \in \mathbb{R}^{N \times |c^v|}$,我们考虑对于每个视图都能依次反向猜测所有视图的content，并且每两个视图所反向猜测的视图应当相应对齐，因此内容不变构造的约束形式化为：

$$   \arg \min_{Mix} \sum_{1 \leq k \lt k' \leq V } \mathbb{E}_{(Z^k, Z^{k'}) \sim p_{Z^k, Z^{k'}}} \left[ \left\| Mix(Z^k) - Mix(Z^{k'}) \right\|_2^2 \right]$$

相应地，为保证互信息最大化，形式化为：

$$\arg \max_{Mix} \sum^{V}_{k=1} H(Mix(Z^{k}))$$

统一起来，我们有损失函数：

$$\mathcal{L}_{\text{CrossViewAlign}} := \sum_{1 \leq k \lt k' \leq V } \mathbb{E}_{(Z^k, Z^{k'}) \sim p_{Z^k, Z^{k'}}} \left[ \left\| Mix(Z^k) - Mix(Z^{k'}) \right\|_2^2 \right]$$

$$ - \sum^{V}_{k=1} H(Mix(Z^{k}))$$​

$\mathcal{L}_{\text{CrossViewAlign}}$​的收敛性证明同样会在附录中给出。

综合Problem1和Problem2的上述步骤，我们有因果损失：

$$\mathcal{L}_{\text{Causal}} = \mathcal{L}_{\text{IntraViewAlign}} + \mathcal{L}_{\text{CrossViewAlign}}$$

In Stage 1, the autoencoder extracts preliminary features that contain some degree of useful information for the clustering task, but a significant amount of noise still remains. This is particularly challenging for multi-view clustering, where it is crucial for the model to discern relevant information. One of our objectives is to enhance the identifiability of features. To achieve this, we introduce causal inference, transitioning from the statistical dependence $P(H^v,C)$ between the input and clustering clusters to minimizing the gap between $P(H^v)$ and $P(H^v|C)$, i.e.,$min \|P(H^v) - P(H^v|C) \|$ , where $C$ denotes the clustering clusters.

According to the Common Cause Principle, we reformulate $H^v$ and $C$ as follows:

$$H^v := g(Z^v) = g(\delta(c^v,s^v,u^v))$$

$$C := h(Z^v) = h(\zeta(c^v,s^v))$$

where $g()$ and $h()$ are view-specific functions, $\delta()$ and $\zeta()$ are noise corruption functions, and $c^v$, $s^v$ and $u^v$ represent the content, style, and view-specific redundant information for each view, respectively. Additionally, $c^v$, $s^v$ and $u^v$ satisfy the conditions $c^v \perp\!\!\!\perp s^v \perp\!\!\!\perp u^v$.In our current multi-view clustering task, the content $c^v$, style $s^v$, and noise $u^v$ are not explicitly provided. Thus, we need to sequentially construct the functions $\{p^v:\mathcal{H}^v \rightarrow (0,1)^{|c^v|} \}$ and $\{ q^v:\mathcal{H}^v \rightarrow (0,1)^{|s^v|} \}$, where $p^v$ and $q^v$  are the content and style projection functions, respectively, and $\mathcal{H}^v$ is the latent sub-space of the preliminary features $H^v$. The dimensions of content and style, $|c^v|$ and $|s^v|$, are also denoted accordingly.

Now, considering the domain discrepancies caused by latent spaces across multiple views, i.e., how do we handle the redundant information $u^v$? We introduce a Gaussian distribution sampling strategy to perform domain transformation on $H^v$ within $\mathcal{H}^v$:



$$\beta(H^{v}) = \mu(H^{v}) + \epsilon_\mu \Sigma_\mu(H^{v}), \quad \epsilon_\mu \sim \mathcal{N}(0, 1),$$

$$\gamma(H^{v}) = \sigma(H^{v}) + \epsilon_\sigma \Sigma_\sigma(H^{v}), \quad \epsilon_\sigma \sim \mathcal{N}(0, 1),$$​

$$\widetilde H^{v} = \gamma(H^{v}) \left( \frac{H^{v} - \mu(H^{v})}{\sigma(H^{v})} \right) + \beta(H^{v}).$$​

Here, the mean and standard deviation of $\mathbf{H}^{v}$ follow $\mathcal{N}(\mu(\mathbf{H}^{v}), \Sigma^2_\mu(\mathbf{H}^{v}))$ and $\mathcal{N}(\sigma(H^{v}), \Sigma^2_\sigma(H^{v}))$ respectively. The parameters $\beta(\mathbf{H}^{v})$ and $\gamma(\mathbf{H}^{v})$ represent the perturbed mean and standard deviation, and $\widetilde{\mathbf{H}}^{v} $ is the transformed noise-intervened feature. Thus, we have:

$$\widetilde{\mathbf{H}}^v := g(Z^v) = g(\delta(\widetilde c^v, \widetilde s^v,\widetilde u^v))$$

At this point, we further need to address two problems:

**Problem 1:** How can we ensure the uniformity of the intra-view content $c^v = p^v(H^{v})$ and $\widetilde c^v = p^{v}(\widetilde H^{v})$, i.e., $c^v = \widetilde c^v$?

**Problem 2:** How can we ensure the cross-view content uniformity between $c^k = p^k(H^{k})$ and $c^{k'} = p^{k'}(H^{k'})$, i.e., $c^v = \widetilde c^v$, where $k,k' \in V$?

For **Problem 1**, we need to constrain the content projection function $p^v$, ensuring that, regardless of whether the feature is perturbed ($H^{v}$) or not ($\widetilde H^{v} $​), the content remains invariant under the same view. Therefore, we formalize this constraint as:

$$   \arg \min_{p^v}  \mathbb{E}_{(\mathbf{H}^v, \widetilde{\mathbf{H}}^v) \sim p_{\mathbf{H}^v, \widetilde{\mathbf{H}}^v}} \left[ \left\| p^v(\mathbf{H}^v) - p^v(\widetilde{\mathbf{H}}^v) \right\|_2^2 \right]$$

Moreover, to ensure that the content features are indeed relevant to the multi-view clustering task, we should maximize the mutual information $I(c^v,C)$ between the extracted content features $c^v = p^v(H^{v})$ and the self-supervised clustering information $C$​. This can be formalized as:

$$\arg \max_{p^v} I(p^v(\mathbf{H}^{v}),C) = \arg \max_{p^v} I(p^v(\mathbf{H}^{v}),p^v(\mathbf{H}^{v})) = \arg \max_{p^v} H(p^v(\mathbf{H}^{v}))$$

where $H()$​ denotes entropy. Combining these, we have the loss function:

$$\mathcal{L}_{\text{IntraViewAlign}}(p^v) :=  \mathbb{E}_{(\mathbf{H}^v, \widetilde{\mathbf{H}}^v) \sim p_{\mathbf{H}^v, \widetilde{\mathbf{H}}^v}} \left[ \left\| p^v(\mathbf{H}^v) - p^v(\widetilde{\mathbf{H}}^v) \right\|_2^2 \right] - H(p^v(\mathbf{H}^{v}))$$​

The convergence proof for $\mathcal{L}_{\text{CrossViewAlign}}$ will be provided in the appendix.

For **Problem 2**, we adopt the approach of using a causal content label to supervise cross-view content consistency. Specifically, we construct the projection function $\{ Mix^v: \mathcal{Z}^v \rightarrow \mathcal{Y}^{v} \}$, where $\mathcal{Y}^{v} \subseteq \mathbb{R}^{V \times N \times |c^v|}$, to obtain the label $Y^v = Mix(Z^v) = Mix(p(\mathbf{H}^{v}))$. This approach is similar to the modeling of **Problem 1**, but with the difference that for the labels $Y^k = [y^k_1,y^k_2,...,y^k_v ],k \in V,y^k_i \in \mathbb{R}^{N \times |c^v|}$, we consider that each view should be able to infer the content of all other views, and the inferred content between any two views should be aligned accordingly. Therefore, the constraint to ensure content invariance is formalized as:

$$   \arg \min_{Mix} \sum_{1 \leq k \lt k' \leq V } \mathbb{E}_{(Z^k, Z^{k'}) \sim p_{Z^k, Z^{k'}}} \left[ \left\| Mix(Z^k) - Mix(Z^{k'}) \right\|_2^2 \right]$$

Correspondingly, to ensure mutual information maximization, it is formalized as:

$$\arg \max_{Mix} \sum^{V}_{k=1} H(Mix(Z^{k}))$$

Combining these, we obtain the loss function:

$$\mathcal{L}_{\text{CrossViewAlign}} := \sum_{1 \leq k \lt k' \leq V } \mathbb{E}_{(Z^k, Z^{k'}) \sim p_{Z^k, Z^{k'}}} \left[ \left\| Mix(Z^k) - Mix(Z^{k'}) \right\|_2^2 \right]$$

$$ - \sum^{V}_{k=1} H(Mix(Z^{k}))$$

The convergence proof for $\mathcal{L}_{\text{CrossViewAlign}}$ will also be provided in the appendix.

By integrating the steps outlined for **Problem 1** and **Problem 2**, we obtain the causal loss:

$$\mathcal{L}_{\text{Causal}} = \mathcal{L}_{\text{IntraViewAlign}} + \mathcal{L}_{\text{CrossViewAlign}}$$

### Stage 3

考虑到我们在Stage2处理好的因果特征进行了content-style的成分分析和解构，而因果机制相较于Gragh-based multi-view clustering所启发之处在于，对于特征的处理，先前的工作如DealMVC仅从如何更好的利用注意力机制或者其他MLP架构更好融合不同视角下的语义特征和标签特征，并且将不同视角之间的特征作对齐。实际上，前文也提到，对于黑箱式的融合机制对于较为困难的数据集会因为较强的statistical dependence $P(H^v,C)$而导致对于某些有用的语义信息被忽略并且同时错误地强化某些无关的信息。因此，我们在Stage 2的铺垫下进行简单但有效的视图质量判断策略。我们引入可学习的权重参数$\{w^{1}_{\alpha},w^{2}_{\alpha},...,w^{v}_{\alpha} \}, \{ \tilde w^{1}_{\alpha}, \tilde w^{2}_{\alpha},..., \tilde w^{v}_{\alpha} \}, v \in V$，构造高质量统一因果content-style特征：

$$U=\sum ^{V}_{v=1}\dfrac{e^{w^{v}_{\alpha}}}{\sum ^{V}_{k=1}e^{w^{k}_{\alpha}}}H^{v}, \widetilde U=\sum ^{V}_{v=1}\dfrac{e^{\tilde w^{v}_{\alpha}}}{\sum ^{V}_{k=1}e^{ \tilde w^{k}_{\alpha}}} \widetilde H^{v }$$

基于高质量统一因果content-style特征，我们进一步构造伪标签：

$$P = \sigma(HW+b), \widetilde P = \sigma(\widetilde{H}W+b)$$

where $W \in \mathbb{R}^{D \times C}, b \in \mathbb{R}^{N \times C}$, $\sigma()$为激活函数ReLU。

接着我们构造相似性矩阵：

$$S = \frac{\langle U,\widetilde U^T \rangle }{\| U \|_2 \| \widetilde U^T \|_2}$$

对于相似性矩阵$S \in \mathbb{R}^{N \times N}$而言，对于任意两个样本对$i,j$,若$i \neq j$则构成负样本对，因为我们要尽可能去除不同视图下的view-specific redundant information$u^v$,所以在$S$中应尽可能地约束至0;若$i = j$即在对角线上则构成正样本对，同样因为因为我们要尽可能一致化不同视图下的content$c^v$，所以尽可能相似度约束至1，因此我们构造损失函数：

$$\mathcal{L}_{f\_S} = \frac{1}{2} \| S - I \|_{2}^{2}$$​

where $I \in \mathbb{R}^{N \times N}$表示单位矩阵。接着根据$P=[p_1,p_2,...,p_N], \widetilde P =[\tilde{p}_1,\tilde{p}_2,...,\tilde{p}_N]$我们构造伪标签图:

$$Q =  \begin{cases}  1 &  i = j, \\ p_{i} \cdot \tilde{p}_{j} &  i \neq j \text{ and } p_{i} \cdot \tilde{p}_{j} \geq \tau, \\ 0 & i \neq j \text{ and } p_{i} \cdot \tilde{p}_{j} < \tau \end{cases}$$

类似地，我们对于伪标签图本身进行约束：

$$\mathcal{L}_{f\_Q} = \frac{1}{2} \| Q - I \|_{2}^{2}$$​

之后我们再用构造地相似性矩阵和伪标签图构造对比损失：

$$L = - \sum^{N}_{i=1} Q_{\cdot i} \log(\frac{e^{S_{\cdot i}/ \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} /\tau_c}})$$

$$= -  Q_{ii} \log(\frac{e^{S_{ii} / \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} / \tau_c}}) - \sum^{N}_{k=1,k \neq i} Q_{ik} \log(\frac{e^{S_{ik} / \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} / \tau_c}})$$​

其中第一项pull close了相同样本在不同视图下的特征，第二项pull close了拥有相似content但不同样本的特征。

Considering that in Stage 2 we performed a content-style component analysis and deconstruction of the processed causal features, the key insight of causal mechanisms, as compared to graph-based multi-view clustering, lies in the handling of features. Previous works, such as DealMVC, primarily focused on how to better leverage attention mechanisms or other MLP architectures to integrate semantic and label features across different views and align features between them. However, as previously discussed, black-box fusion mechanisms can lead to significant challenges on complex datasets due to strong statistical dependence $P(H^v,C)$, which may result in the neglect of useful semantic information while simultaneously reinforcing irrelevant information. Therefore, building on the foundation laid in Stage 2, we introduce a simple yet effective strategy for assessing view quality. We incorporate learnable weight parameters $\{w^{1}_{\alpha},w^{2}_{\alpha},...,w^{v}_{\alpha} \}, \{ \tilde w^{1}_{\alpha}, \tilde w^{2}_{\alpha},..., \tilde w^{v}_{\alpha} \}, v \in V$, to construct high-quality unified causal content-style features:

$$U=\sum ^{V}_{v=1}\dfrac{e^{w^{v}_{\alpha}}}{\sum ^{V}_{k=1}e^{w^{k}_{\alpha}}}H^{v}, \widetilde U=\sum ^{V}_{v=1}\dfrac{e^{\tilde w^{v}_{\alpha}}}{\sum ^{V}_{k=1}e^{ \tilde w^{k}_{\alpha}}} \widetilde H^{v }$$

Based on these high-quality unified causal content-style features, we further construct pseudo-labels:

$$P = \sigma(HW+b), \widetilde P = \sigma(\widetilde{H}W+b)$$

where $W \in \mathbb{R}^{D \times C}, b \in \mathbb{R}^{N \times C}$, and $\sigma()$ represents the ReLU activation function.

Next, we construct a similarity matrix:

$$S = \frac{\langle U,\widetilde U^T \rangle }{\| U \|_2 \| \widetilde U^T \|_2}$$

For the similarity matrix $S \in \mathbb{R}^{N \times N}$, any pair of samples $i,j$ (where $i \neq j$) constitutes a negative sample pair, and their similarity in $S$ should be constrained to 0. Conversely, if $i = j$, meaning the pair lies on the diagonal, it constitutes a positive sample pair, and their similarity should be constrained to 1. Therefore, we construct the following loss function:

$$\mathcal{L}_{f\_S} = \frac{1}{2} \| S - I \|_{2}^{2}$$

where $I \in \mathbb{R}^{N \times N}$ denotes the identity matrix. Subsequently, based on $P=[p_1,p_2,...,p_N]$ and $\widetilde P =[\tilde{p}_1,\tilde{p}_2,...,\tilde{p}_N]$, we construct the pseudo-label graph:

$$Q =  \begin{cases}  1 &  i = j, \\ p_{i} \cdot \tilde{p}_{j} &  i \neq j \text{ and } p_{i} \cdot \tilde{p}_{j} \geq \tau, \\ 0 & i \neq j \text{ and } p_{i} \cdot \tilde{p}_{j} < \tau \end{cases}$$

Similarly, we impose constraints on the pseudo-label graph itself:

$$\mathcal{L}_{f\_Q} = \frac{1}{2} \| Q - I \|_{2}^{2}$$

 Then we construct the contrastive loss using the similarity matrix and the pseudo-label graph:

$$L = - \sum^{N}_{i=1} Q_{\cdot i} \log(\frac{e^{S_{\cdot i}/ \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} /\tau_c}})$$

$$= -  Q_{ii} \log(\frac{e^{S_{ii} / \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} / \tau_c}}) - \sum^{N}_{k=1,k \neq i} Q_{ik} \log(\frac{e^{S_{ik} / \tau_c}}{\sum^{N}_{j=1} e^{S_{\cdot j} / \tau_c}})$$

通过上述分析，在Stage2我们obtain了损失函数：
$$\mathcal{L}_{\text{Gragh}} = \mathcal{L}_{\text{Contra}} + \mathcal{L}_{\text{f\_S} }+ \mathcal{L}_{\text{f\_Q} }$$

综上所有三个阶段，我们模型的总损失可以统计为：

$$\mathcal{L} = \mathcal{L}_{\text{Rec}} + \lambda_1 \mathcal{L}_{\text{Causal} }+ \lambda_2 \mathcal{L}_{\text{Graph} }$$

其中$\lambda_1$和$\lambda_1$是hyper-parameters，which will 在experiments中展开讨论 in details.

## experiments

### experiment setup

#### datasets and baselines

为了评估我们的模型的有效性，我们在八个多视角数据集上展开实验，这八个数据集分别为MSRC、Nuswide、Citeseer、Wiki、Caltech-2V、Caltech-3V、Caltech-4V、Caltech-5V.对于数据集的简要介绍在table 1中。我们选取了15个SOTA Multi-view clustering方法，包括了COMIC、CoMVC、SiMVC、MFLVC、MVD、DSMVC、DealMVC、DFPGNN、SC、BMVC、RMVC、MVC-LFA、EAMC、OPLFMVC、CGFAgg。

####  Evaluation metrics

为了评估我们的模型的有效性，我们选取了最通用的The clustering performance的评估指标：clustering accuracy (ACC), normalized mutual information (NMI), and purity (PUR).

####  Implementation details

在我们提出的方法MSC-MVC中，我们将Stage1、Stage2和Stage3的learning rate分别设置在0.001、0.001、0.003,训练的batch size设置在32。$\lambda_1$和$\lambda_2$设置在$[0.01,0.1,1,10,100]$这个范围中。在Stage3中的$\tau_Q$和$\tau_c$​分别设置为0.8、1.



### Experimental Results and Analysis
#### Performance Comparison

Table 2 和Table 3展示了我们的模型我们选取的baseline methods在不同数据集的下的性能表现，其中best value用红色标注，第二好的value用蓝色标注。从表格数据结果中，我们可以看到：

- 在ACC指标上，我们的模型在八个数据集上均优于其他所有baseline methods。并且对于较难的数据集比如Nuswide和Wiki，均能有在ACC、NMI、PUR三个指标全优于其他模型的表现。这说明对于较难的任务而言，MSC-MVC能够从聚类信息中挖掘最为本质的content feature，并且将无关信息进行剔除，因此做到了从因果本质层面解析Cross-view和Intra-View的聚类特征。
- 特别注意到对于Caltech数据集而言，随着数据集视角数增加，MSC-MVC能够做到在三个指标上的稳步提升。而相较于其他模型而言，有的模型如CGFAgg、EAMC会出现视角数增加过程中性能却有下滑的情况。这一点同样说明了MSC-MVC能够正确捕捉跨视角聚类信息并且有效建立相似图。

#### Ablation Study

为了验证MSC-MVC的各阶段的有效性，我们设计了相应的消融实验。实验结果如Table 4所示，可以看到在MSRC数据集上，我们很有意思地发现到MSC-MVC存在某种路径依赖的性质——1）在缺失$\mathcal{L}_{\text{Causal} }$的情况下，相较于$\mathcal{L}_{\text{Causal} } +\mathcal{L}_{\text{Graph} }$的情况之下有一定的性能损失，因为缺少对于特征本质内容的有效挖掘而导致聚类效果有一定下降；2）在缺失$\mathcal{L}_{\text{Graph} }$的情况下，由于即便从causal角度挖掘了本质特征但是由于缺少视图质量的判断和相似图的计算而导致性能大打折扣。从这个角度而言，也反映了我们考虑multi-stage策略的合理性，也就是说从multi-stage角度逐步挖掘有效聚类信息从而提升模型性能。

#### Visualization of Clustering Results

为了更好地展示MSC-MVC在不同视角下以及所有视角下经过视图质量判别后对于特征空间的有效聚类，我们采用了t-SNE算法对于聚类结果进行可视化。从figure2的上方的原始特征和下方的经过MSC-MVC后所挖掘的聚类特征可以看出，在各个视角下以及所有视角下对于同一类别的聚类特征有效且正确地进行聚合。

#### Parameter sensitivity analysis

为了进一步讨论参数$\lambda_1$和$\lambda_2$的选取和分析其敏感性，我们继续进一步进行Parameter sensitivity analysis实验。如Figure 4所示，可以看到对于Wiki数据集而言，对于参数大小的选取从评估角度而言相对缓和；对于MSRC数据集而言，对于参数$\lambda_2$的选取比较敏感，对于样本数量较小的数据集而言Stage3的Graph-based method对于聚类结果的影响更明显。







### 聚类散点图

在数据集Caltech-5V上的聚类结果。从左到右依次为第一至第五个视角和所有视角合并至一起的结果，上方为原始特征，下方为MSC-MVC训练后所得的结果，并且使用的方法是t-SNE。



We propose a novel multi-view clustering method termed MSC-MVC, pioneering the application of causal inference in multi-view clustering tasks.
 we introduce a causal mechanism-based approach to view quality assessment from both intra-view and cross-view perspectives, utilizing causal content labels to supervise feature effectiveness.
Through multi-stage feature processing, we employ causal essential features for clustering tasks and alignment, while generalizing at the feature level to achieve consistency in similarity graphs.





## 创新点

 

我们提出了一个新颖的多视角聚类模型，名为multi-stage causal feature learning for multi-view clustering,利用多阶段训练的方式将因果机制引入到多视角聚类的任务中。

对于从原始数据提取出的特征，我们从视角内部(intra-view)和跨视角(cross-view)两个角度进行因果层面的分析，对于挖掘这两方面的聚类内容一致性展开了理论建模推导和并且构建相应的统一的损失函数。在视角内部角度我们采用高斯分布采样的方式获得约束内容一致性，在跨视角角度我们采用构建因果内容标签(causal content label)的方式约束来自不同视角的聚类内容。

我们采用基于图的对比学习的方式去除在因果阶段建模所展开阐述的聚类无关信息，同时利用相似图拉近有相似聚类内容和视图特定因果风格的样本，并且采用伪标签图的方式监督相似图的因果内容-风格的聚类程度。

The illustration of 我们提出的多视角 structural causal model.对于intra-view和cross-view的causal content我们都进行invariance约束。

The overview of MSC-MVC.我们的模型框架分为三个阶段进行训练，对于输入数据会经过第一阶段的autoencoder提取出Preliminary Feature,对于第一阶段提取的特征会经过Gaussian distribution sampling进行视角内部特征增强。接着在第二阶段中，输入特征从intra-view和cross-view角度分别提取causal content.最后在第三阶段我们通过基于因果的视图质量判断和graph-based contrasive learning对于前两个阶段处理好的特征进行聚类信息的提取和监督.





我们的whole  multi-stage process of  MSC-MSC is summarized in Algorithm 1，并且我们最终使用k-means的方法进行微调。



Figure1：在Caltech-kV数据集上MSC-MVC与其他模型的聚类表现结果,其中k表示为视角数。评估指标分别为clustering
accuracy (ACC), normalized mutual information (NMI), and Purity (PUR).
