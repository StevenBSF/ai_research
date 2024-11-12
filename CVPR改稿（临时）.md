# 方法部分

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