根据 Fano 不等式，分类（或聚类）的错误率 P_e满足

$P_e \geq 1 - \frac{I\Bigl(Y; H^{(1)}, \dots, {H}^{(V)}\Bigr) + \log 2}{\log K}$,

其中 K 是类别数；因此，聚类准确率 ACC = 1 - P_e 满足

$ACC \leq \frac{I\Bigl(Y; H^{(1)}, \dots, {H}^{(V)}\Bigr) + \log 2}{\log K}$.

$ACC \leq \frac{\sum I(Y; c, s_v) + \log 2}{\log K}$.

但在存在强视图依赖的情况下，虽然每个视图 X^{(v)} 单独可能携带较多信息 I\bigl(Y; X^{(v)}\bigr)，但由于视图之间高度相关，联合互信息 I(Y; X) 实际上远小于各视图互信息的简单累加：

$I(Y; X) < \sum_{v=1}^{V} I\bigl(Y; H^{(v)}\bigr).$



假设每个视图的数据不仅由共享的内容因子 $c$ 生成，还包含一个视图特定的风格 $s^{(v)}$ 和一些残余噪声 $u^{(v)}$。那么，可以将数据生成过程写为

$H^{(v)} = g^{(v)}(c, s^{(v)}, u^{(v)}),$

经过因果解耦后，我们实际上提取出的是 c 和有益的 s_v 两部分信息，而将冗余噪声 u_v 尽可能去除。理想情况下，我们希望：

$I(Y; c, s_v) \approx I\Bigl(Y; H^{(1)}, \dots, {H}^{(V)}\Bigr) + \Delta$,

其中 $\Delta > 0$ 表示通过解耦去除了噪声带来的冗余后增加的有效信息。此时，基于 $c$ 与 $s_v$ 进行聚类，其理论上界变为

$ACC \leq \frac{I(Y; c, s_v) + \log 2}{\log K}$.

由于 $\Delta > 0$，有效信息量提升，从而使得ACC的上界高于传统方法的上界。



NVD:

补充现有方法的上界



$I\Bigl(Y; H^{(1)}, \dots, H^{(V)}\Bigr) = \sum_{v=1}^{V} I\bigl(Y; c, s_v, u_v\bigr) - R,$

这里的交互互信息项正是针对不同视图中的噪声部分的重复贡献而引入的。我们可以将

$R = \sum_{i<j} I\bigl(Y; u_i, u_j\bigr) - \sum_{i<j<k} I\bigl(Y; u_i, u_j, u_k\bigr) + \cdots,$

从而明确表示噪声 $u_v$ 在不同视图中引起的冗余信息损失。

$I\Bigl(Y; H^{(1)}, \dots, H^{(V)}\Bigr)  = \sum_{v=1}^{V} I\bigl(Y; c, s_v, u_v\bigr) - R$

$\leq \sum_{v=1}^{V} I\bigl(Y; c, s_v, \tilde u_v\bigr) - \tilde R$.



DVD:

在多视图融合中，我们通常期望各视图的信息能够互补叠加，提升联合互信息 I\Bigl(Y; X^{(1)}, \dots, X^{(V)}\Bigr)。但如果某个视图的 s_v 部分贡献过大，即 I(Y; s_v) 在各视图中极不平衡，那么联合互信息往往会受到主导视图的限制：

$I\Bigl(Y; H^{(1)}, \dots, H^{(V)}\Bigr) \approx I(Y; c) + \max_{v} I\bigl(Y; s_v\bigr) + \text{(NVD)}.$

为了解决 Dominant View Dependency，即避免某个视图的信息过于主导，许多多视角聚类方法会引入视图加权或一致性约束，从而平衡各视图的贡献。即使没有将 s_v 进一步明确拆分为两个部分，我们也可以通过设计损失函数或引入权重因子 w_v 来约束各视图的信息贡献，使得

$I\Bigl(Y; X^{(1)}, \dots, X^{(V)}\Bigr) \approx I(Y; c) + \sum_{v=1}^{V} w_v \, I\bigl(Y; s_v\bigr),$

其中希望权重 w_v 不会让某一视图的信息占比过大，从而使整体融合后的信息量能够更好地反映数据的共性和互补性。



对于权重的解释：

我们可以用信息链式法则来解释这两者之间的关系。对单个视图来说，有：

$I(Y; c, s_v) = I(Y; c) + I(Y; s_v \mid c) = I(Y; c) + \frac{I(Y; s_v \mid c)}{I(Y; s_v )} I(Y; s_v ).$

而一般来说，由于条件熵的性质，我们有

$I(Y; s_v) \ge I(Y; s_v \mid c).$

因此，如果我们取 $w_v=1$，则

$I(Y; c) + I(Y; s_v) \ge I(Y; c) + I(Y; s_v \mid c) = I(Y; c, s_v).$

也就是说，直接把 I(Y; s_v) 加上去会给出一个上界。如果我们希望更精确地描述经过共享信息 c 后，各视图特定信息对 Y 的贡献，可以设置一个权重

$w_v = \frac{I(Y; s_v \mid c)}{I(Y; s_v)},$

那么有

$I(Y; c, s_v) = I(Y; c) + w_v I(Y; s_v).$

在多视角情形中，如果我们有 V 个视图，且共享变量 c 是全局的，那么联合互信息可以写为

$I\Bigl(Y; c, \{s_v\}{v=1}^{V}\Bigr) = I(Y; c) + I\Bigl(Y; \{s_v\}_{v=1}^{V} \mid c\Bigr).$

如果在给定 c 后各视图的 s_v 近似条件独立，那么

$I\Bigl(Y; \{s_v\}{v=1}^{V} \mid c\Bigr) \approx \sum_{v=1}^{V} I(Y; s_v \mid c).$

因此，一个合理的表达就是

$I\Bigl(Y; c, \{s_v\}{v=1}^{V}\Bigr) \approx I(Y; c) + \sum{v=1}^{V} w_v\, I(Y; s_v),$

其中每个权重 w_v 满足

$w_v \approx \frac{I(Y; s_v \mid c)}{I(Y; s_v)},$

这就保证了如果条件下 $s_v$ 与 $Y$ 的信息损失较大（即 $I(Y; s_v \mid c)$ 较小），相应地 $w_v$ 会较小，使得加权后的总和不会超过实际的 $I(Y; c, s_v)$ 值。











Fano 不等式的推导利用了条件熵和互信息的基本不等式。

​	1.	**条件熵的不等式**

设 Y 为类别变量，其取值个数为 K；\hat{Y} 是基于观察 X 得到的估计。令分类错误率为

$P_e = \Pr\{\hat{Y} \neq Y\}$.

根据 Fano 不等式的经典形式，有

$H(Y \mid X) \leq H(P_e) + P_e \log(K - 1)$,

其中 $H(Y \mid X)$ 表示 $Y$ 的条件熵，而 $H(P_e) = -P_e \log P_e - (1 - P_e) \log(1 - P_e)$ 是二元熵函数。

​	2.	**利用互信息的定义**

根据互信息的定义，有

$I(Y; X) = H(Y) - H(Y \mid X)$.

如果我们假设 Y 均匀分布，则

$H(Y) = \log K$.

因此可以写成

$H(Y \mid X) = \log K - I(Y; X).$

​	3.	**结合两式**

将上面两式结合，我们有

$\log K - I(Y; X) \leq H(P_e) + P_e \log(K - 1)$.

当 $P_e \leq 1/2$ 时，二元熵 $H(P_e)$ 的上界为

$H(P_e) \leq \log 2$.

同时，对于 K 较大时，可以近似认为 $\log(K - 1) \leq \log K$。因此，上式可进一步放宽为

$\log K - I(Y; X) \leq \log 2 + P_e \log K$.

​	4.	**求解** $P_e$

整理上述不等式：

$\log K - I(Y; X) - \log 2 \leq P_e \log K$.

两边除以 $\log K$（注意 $\log K > 0$），得到

$P_e \geq 1 - \frac{I(Y; X) + \log 2}{\log K}.$