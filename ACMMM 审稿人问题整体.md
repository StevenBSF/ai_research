- 计算复杂度和泛化能力
  - （b7E9_Q1）双差分网络和稀疏性正则化带来计算成本，尚未验证在非视觉数据上的性能。
  - （3taB_Q1）请提供关于计算复杂度的详细分析，并与现有方法进行实验对比，以展示所提出方法的可扩展性。
  - （2bNa_Q2)论文中未对 CausalMVC 的计算复杂度进行详细分析。特别是该网络结构是否会相较于基线方法增加整体复杂度？希望作者提供量化的比较分析。

- 超参数
  - （b7E9_Q2）模型对λ₁、λ₂、β敏感，但只在部分数据集上给出结果，其他数据集如何设置仍不清晰
  - （3taB_Q3）建议提供关键超参数（如 λ₁、λ₂、β 等）的设定准则，并说明这些参数对模型性能的影响。
  - （4SxA_Q1）伪标签掩码图的阈值 和 对比学习中的温度参数 被设为固定常数。请作者说明模型对这些参数是否敏感，以及它们是否影响最终性能与收敛性？
  - （4SxA_Q2）第4.2节中对特征均值和方差加入了高斯噪声进行扰动，但并未解释扰动方差参数（记作 和 ）的设定原则，也未说明这些参数如何影响结果的波动性。

- 内容风格可解释性
  - （b7E9_Q3）复杂的因果内容-风格机制使模型决策过程难以理解。
  - （QS4h_Q1）本文对内容/风格的可识别性分析仍然不足。作者提出的损失函数能如何确保语义因素的有效解耦？
  - （2bNa_Q3）CausalMVC 将多视角表示分解为内容与风格两个部分。这两个部分是否都对最终聚类结果有益？它们各自的贡献程度如何？请提供进一步分析与解释。
- 方法可能存在的问题
  - （b7E9_Q4）易对无噪声视角过拟合，可能忽略带噪视角中的风格信息，影响实际场景下的表现。
  - （4SxA_Q3）文中使用的多视角数据集在聚类规模上是否均衡？若在现实场景下存在高度不平衡的簇分布，所提出方法是否依然有效？
  - （3taB_Q4）在获取最终聚类结果时，作者是否评估了其他方法（如谱聚类）？这些方法在处理非凸簇结构方面可能优于K-means。
  - （QS4h_Q3）所提出方法的聚类结果依赖于由多视角表示生成的伪标签。如果这些伪标签在初始化阶段不准确，是否会对最终的聚类性能产生不利影响？
- 方法设计性质解释
  - （QS4h_Q2）图4中的实验结果显示，随着视角数量的增加，CausalMVC的性能有所提升。作者能否解释该方法具备哪些内在性质，从而导致这种性能增长趋势？
  - （2bNa_Q1）双差分内容-风格网络通过计算内容与噪声的 Query-Key 差异来实现噪声抑制。请作者进一步说明该设计与已有差分Transformer架构的区别，以及它带来了哪些具体优势？
  - （b7E9_Q5）正则化影响未解释清楚，例如LSparseCov的具体优势与影响分析不足
- 实验相关
  - （QS4h_Q4）本文在MVC任务中展示了优秀的实验效果，因此我对其可复现性很感兴趣。希望作者能在答辩信中提供匿名代码链接，展示如何复现表1、2、3、4中的实验结果，并给出十个数据集的对应超参数配置。
  - （3taB_Q2）作者应量化内容风格感受野相较于标准融合方法在性能上的提升，并通过消融实验予以验证。





下面是关于“计算复杂度和泛化能力”三位审稿人问题的回复草稿，已对理论复杂度、实测开销以及非视觉数据上的初步验证进行了补充说明。



**回复 b7E9_Q1 / 3taB_Q1 / 2bNa_Q2：计算复杂度分析与可扩展性验证**



1. **理论复杂度分析**

   - 对于每个视角，DiffMapping(·) 包含两次 Query–Key 映射和一次 Value 变换，整体计算量为

     $$O\bigl(ND^2 + ND^2\bigr)=O(ND^2), $$

     其中 $N$ 为样本数，$D$ 为统一的潜在维度。与常见的基于注意力的 MVC 方法（如 GCFAgg、CVCL）在同阶的 $O(ND^2)$ 复杂度一致。

   - 稀疏协方差正则化需要对每个视角计算 $D\times D$ 的协方差矩阵并施加 L1 与低秩约束，其复杂度同样为 $O(ND^2)$，但由于协方差矩阵维度仅 $D\times D$。



总回答写在前面

某个问题看G1，G2.。。。

之后写单独的审稿人







请你仔细阅读我们的CausalMVC这篇文章现在是ACM MM会议rebuttal阶段，我需要给审稿人的Rebuttal Questions写相应的回复。首先出于礼貌要感谢审稿人提出的问题。
对于Model Interpretability你需要从原理上进行合理的解释，不是敷衍地说我们会补充某个section或者会写在补充材料里。
对于Overfitting to Noise-Free Views这个提问，你要委婉的说明审稿人理解偏了，审稿人可能理解的是，对于noisy view dependency这个现象，如果说模型suffer from这些noisy view，那么模型可能会更关注于那些noise-free的view，导致Overfitting to Noise-Free Views。实际上我们已经已经将问题归纳为两个问题，noisy view dependency和dominant view dependency。而对于dominant view dependency，我们模型要去解决的就是如何平衡dominant view和非dominant view，这个其实也是解决Overfitting的问题。因此你要把审稿人的逻辑思维带到我们的motivation也就是noisy view dependency和dominant view dependency这两个问题本身上。

对于Limited Explanation of Regularization Impact的提问，可以根据实验部分Table3对于L_Sparce的消融实验和我们方法部分对于regularization techniques的介绍进一步阐释。





请你仔细阅读我们的CausalMVC这篇文章现在是ACM MM会议rebuttal阶段，我需要给审稿人的Rebuttal Questions写相应的回复。首先出于礼貌要感谢审稿人提出的问题。
对于Model Interpretability你需要从原理上进行合理的解释，不是敷衍地说我们会补充某个section或者会写在补充材料里。
对于Overfitting to Noise-Free Views这个提问，你要委婉的说明审稿人理解偏了，审稿人可能理解的是，对于noisy view dependency这个现象，如果说模型suffer from这些noisy view，那么模型可能会更关注于那些noise-free的view，导致Overfitting to Noise-Free Views。实际上我们已经已经将问题归纳为两个问题，noisy view dependency和dominant view dependency。而对于dominant view dependency，我们模型要去解决的就是如何平衡dominant view和非dominant view，这个其实也是解决Overfitting的问题。因此你要把审稿人的逻辑思维带到我们的motivation也就是noisy view dependency和dominant view dependency这两个问题本身上。

对于Limited Explanation of Regularization Impact的提问，可以根据实验部分Table3对于L_Sparce的消融实验和我们方法部分对于regularization techniques的介绍进一步阐释。







请你仔细阅读我们的CausalMVC这篇文章现在是ACM MM会议rebuttal阶段，我需要给审稿人的Rebuttal Questions写相应的回复。首先出于礼貌要感谢审稿人提出的问题。
对于"The paper lacks some analysis on the identifiability of content/style decomposition. How can the proposed loss functions ensure effective disentanglement of semantic factors?"这里再解释一下。

对于"The experiment results in figure 4 show improved performance of CausalMVC as the number of views increases. Could the authors clarify the underlying properties of the method that ensure this behavior?"，建议从我们这个方法的机制去解释，就是我们方法对我们方法在方法层面和技术层面和已有方法的区别。比如说利用不同视角的互补信息等等。

对于"The clustering results of the proposed method depend on pseudo-labels generated from multi-view representations. Could inaccurate pseudo-labels during initialization negatively affect the final clustering performance?"就是这里我们会补充实验，就是随机初始化若干次，结果影响不大。
对于"The experiment results of this article are excellent in MVC. Thus, I am interested in the reproducibility of the experiment results. I hope the authors can provide an anonymous code link in the reply letter, to show the reproduction of the results in Table 1, 2, 3, 4, as well as implement the corresponding hyperparameters for ten datasets."我们会在论文中稿之后开源代码。

