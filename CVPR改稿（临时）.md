# 方法部分

- 我们提出的主要方法步骤由图2所示。对于输入数据，我们会先进行preliminary feature alignment并用重构损失加以约束，得到我们的初步特征，之后我们会将初步特征从intraview和crossview两个角度进行因果层面提取出各个视图下的content和style特征，对于得到的content和style特征，我们在构建 unified causal content-style features基础上进一步通过对比学习的方法 extract and supervise the clustering information 。
- 由于我们要对于输入的特征进行因果层面的分析以期望能够达到对于content、style、无关信息的解耦，我们需要对输入的特征和目标任务进行logical reformulation。

- 考虑到对于多视角特征对于视角内部的unique noise和不同视角下由于视角之间的差异造成的内容信息的差异，并且对于视角之间，模型可能过度强调某个主视角而导致难以有效整合互补信息，因此为了进一步解决这些现有的难点，我们从intra-view和cross-view两个角度考虑以下两个问题：
- Considering the unique noise within each view's features and the content discrepancies arising from differences across views, the model may overemphasize a dominant view, making it challenging to effectively integrate complementary information. To further address these existing challenges, we examine the following two aspects from both intra-view and cross-view perspectives:



- Here, the mean and standard deviation of $\mathbf{H}^{v}$ follow $\mathcal{N}(\mu(\mathbf{H}^{v}), \Sigma^2_\mu(\mathbf{H}^{v}))$ and $\mathcal{N}(\sigma(\mathbf{H}^{v}), \Sigma^2_\sigma(\mathbf{H}^{v}))$ respectively. The parameters $\beta(\mathbf{H}^{v})$ and $\gamma(\mathbf{H}^{v})$ represent the perturbed mean and standard deviation, and $\widetilde{\mathbf{H}}^{v} $ is the transformed noise-intervened feature.因此，$\widetilde{\mathbf{H}}^{v}$也可以从因果角度reformulate成$\widetilde{\mathbf{H}}^v := g(\tilde c^v, \tilde s^v,\tilde u^v)$.

- \textbf{Problem 1}: How can we ensure the consistency of the intra-view \textit{content} $c^v := p^v_{1:|c^v|}(\mathbf{H}^{v})$ and $\tilde c^v := p^v_{1:|c^v|}(\widetilde {\mathbf{H}}^{v})$, i.e., $c^v = \tilde c^v$,并且挖掘出intra-view的style$s^v:=p^v(\mathbf{H}^{v})_{(|c^v|+1):(|c^v+s^v|)}$?
- \textbf{Problem 2}: How can we ensure the consistency of the cross-view \textit{content} $c^k = p^k_{1:|c^k|}(\mathbf{H}^{k})$ and $c^{k'} = p^{k'}_{1:|c^{k'}|}(\mathbf{H}^{k'})$, i.e., $c^k = c^{k'}$, 并且保证视角之间\textit{style}$s^k$,$s^{k'}$的差异性，where $k,k' \in V$?



- 对于Problem 1，为了进行视角内的内容的consistency约束并且考虑real world下的视角内的差异，我们需要对于初步特征进行扰动。

- For视角内的内容特征和风格变化，我们需要对于在已有的因果形式化建模的基础上，进行相应的特征处理和损失约束。对于多视角聚类任务下content-style信息挖掘的难点在于如何更好保证content consistency的前提下提取更加有用的聚类信息。因此我们引入了基于差分思想的特征投影。对于目前已有的preliminary feature$\mathbf{Z}^{v}$ 和经过扰动的feature$\widetilde{\mathbf{Z}}^{v}$，我们构建如下差分网络以获得content feature$\mathbf{Z}_c^{v}$，$\widetilde{\mathbf{Z}}_c^{v}$和style feature$\mathbf{Z}_s^{v}$ :

- 同时，$\widetilde{\mathbf{Z}}_c^{v}$也可通过$\widetilde{\mathbf{Z}}_c^{v}=\text{DiffMapping}(\widetilde{\mathbf{H}}_c^{v})$获得。

- 同时，为了保证视角内\textit{content}和\textit{style}的差异性，我们需要对其进行解耦。因此我们引入以下约束：

- 为了保证对于cross view情形下不同视角之间的\textit{style}的invariance，我们需要对于

  



$$\mathbf{Z}_c^v =&\ p^v_{1:|c^v|}(\mathbf{H}^{v}) \\
    =&\ \text{DiffMapping}(\mathbf{H}^v) \\
    =&\ \bigl( \mathtt{Softmax}\bigl(  \mathbf{H}^v \mathbf{W}_{Q_c} (\mathbf{H}^v \mathbf{W}_{K_c})^\top   \bigr) \\
    &\quad -\ \mathtt{Softmax}\bigl( \mathbf{H}^v \mathbf{W}_{Q_u}  ( \mathbf{H}^v \mathbf{W}_{K_u} )^\top \bigr) \bigr) \mathbf{H}^v \mathbf{W}_c^v $$