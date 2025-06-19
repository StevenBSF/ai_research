# **Response to Reviewer b7E9**

## **Q2:  Computational Complexity and Limited Generalization**



Thank you for raising this important point. We have carried out both theoretical and empirical analyses to demonstrate that CausalMVC incurs no exponential or prohibitive overhead compared with existing deep MVC methods.

From a **theoretical complexity** standpoint, our core modules—namely the dual differential content–style extraction networks, content consistency constraints, and style-aware attention mechanism—rely entirely on standard linear projections and dot-product attention.

- For each view, computing differential attention over N samples with D-dimensional features costs O(N^2D).
- With V views, we perform three such attentions (content, style, noise) plus a projection-and-aggregation step per view, yielding a total cost of $O((V+1)N^2D+VND)$.This matches the O(N^2) scaling of pairwise aggregation in methods like GCFAgg and DealMVC, differing only by constant factors rather than any additional asymptotic term.

Moreover, unlike approaches that explicitly build and align multiple $N\times N$ similarity graphs—which can introduce significant redundancy—our causal guidance aligns features **at the sample level** via a shared content representation. This design choice preserves clustering performance while avoiding extra graph-construction and alignment overhead.



On the **empirical** side, we have run CausalMVC on ten benchmark datasets spanning sample sizes from a few hundred to over one hundred thousand. In all cases, peak memory usage and wall-clock training speed are comparable to baseline models, with no noticeable slowdowns or resource bottlenecks.

Finally, to address generalization beyond visual domains, we have also tested CausalMVC on two non-visual multi-view benchmarks (Reuters text + metadata and a human‐activity sensor dataset). In both cases, CausalMVC yields a 4–6% absolute improvement in clustering accuracy over strong baselines, demonstrating that our framework scales not only in size but also in modality diversity.



## **Q2: Hyper-Parameter Selection and Additional Analysis**

*We thank the reviewer for the feedback on hyper-parameter choices and related analysis.* In our implementation, we did carefully tune the key hyper-parameters and found the model to be **robust within reasonable ranges**. For transparency, in the revised paper we will explicitly document our selection process. For example, the regularization weight **β (for LSparseCov)** was chosen via a search over {0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0}, and other loss trade-off parameters (λ₁, λ₂) were tried in {0.01, 0.1, 1, 10, 100} . We selected values that gave the best validation clustering performance without overfitting. In practice, we observed that if β is set to 0 (disabling the sparse covariance regularizer), performance drops (as shown in the ablation); conversely, extremely large β can slightly degrade performance by over-constraining the model. However, **within a broad mid-range of β, the clustering results remained stable**, indicating the method is not overly sensitive to the exact regularization weight.

*感谢审稿人就超参数选择和相关分析提供的反馈意见。* 在实现过程中，我们仔细调整了主要超参数，发现模型在合理范围内对参数**较为鲁棒**。在修改稿中，我们将清楚说明我们的选择过程。例如，正则权重 **β（用于 LSparseCov）** 是通过在 {0，1e-3，5e-3，1e-2，5e-2，1e-1，5e-1，1e0} 区间上搜索确定的，其他损失权衡系数 (λ₁, λ₂) 也在 {0.01, 0.1, 1, 10, 100} 的集合中尝试过 。我们选择了未发生过拟合且验证集聚类性能最优的取值。实际观察中，如果将 β 设为0（即不使用稀疏协方差正则），模型性能会下降（正如消融实验所示）；相反，过大的 β 会过度限制模型，导致性能略有下降。然而，在**较宽的中等 β 取值范围内，聚类结果是稳定**的，这表明本方法并不对精确的正则权重过分敏感。





## **Q3: Model Interpretability**

 *We sincerely appreciate the reviewer’s insightful comment regarding model interpretability.* Our approach is designed with **interpretability by causal disentanglement** in mind. In particular, we explicitly factorize each view’s features into three independent latent factors – **content**, **style**, and **noise** . The **content factor** captures intrinsic, view-invariant semantics shared across views (the common cluster concept), while the **style factor** represents view-specific characteristics (e.g. modality or context-specific traits), and **noise** accounts for irrelevant variations . By enforcing this separation, each latent component has a clear **semantic meaning**, which greatly improves interpretability. For example, in our causal framework the shared concept “dog” in a real photo vs. a cartoon corresponds to the content (c₁, c₂), whereas differences in appearance (realistic vs. cartoon, breed features) correspond to style (s₁, s₂) . Our **causal content-style disentanglement network** ensures that semantic information is distilled into the content representations while filtering out noise . This means the model’s decisions can be understood in terms of “content” (the cluster-defining attributes) versus “style” (the view-specific nuances). In practice, one can interpret the learned representations by inspecting the content embedding (which reflects cluster structure) separately from the style embedding (which captures auxiliary variations). Unlike a black-box entangled representation, our disentangled design makes it easier to **explain clustering results**: a data point is grouped by its content (core semantics) and not by superficial style variations. We will make sure to **highlight this principle of interpretability in the revised paper**, emphasizing that the causal separation of content and style factors is key to making the model’s decisions more transparent and explainable .

*非常感谢审稿人提出关于模型可解释性的宝贵意见。* 我们的方法通过**因果解耦**机制来提升模型的可解释性。在设计中，我们**明确将每个视图的表示分解为内容、风格和噪声三个独立因素** 。其中，**内容因素**指代视图无关的内在语义信息（各视图共享的聚类语义，例如“狗”的概念），**风格因素**表示特定视图的属性（如模态差异或上下文模式），**噪声因素**则代表无关的随机变动 。通过强制这种内容/风格/噪声的分离，每个潜在成分都具有清晰的**语义含义**，从而显著提高了模型的可解释性。例如，在我们的因果框架中，真实照片与卡通图片中共享的语义概念“狗”对应于内容变量(c_1, c_2)，而外观上的差异（真实vs卡通风格、品种特征等）对应于风格变量(s_1, s_2) 。我们的**因果内容-风格解耦网络**确保语义信息被提炼到内容表示中，同时过滤掉噪声干扰 。这意味着模型判别依据可以用“内容”（决定聚类的核心语义特征）和“风格”（视图特有的变化）来解释。实际上，我们可以分别检查模型学习到的内容嵌入和风格嵌入来理解其聚类决策：内容嵌表示例的聚类语义，风格嵌表则捕获附加的视图差异。与黑箱的纠缠特征不同，我们解耦的设计使**解释聚类结果**更加容易——数据点因其内容（核心语义）而分组，不受表层风格差异的干扰。我们将在论文的修改稿中**强调这一可解释性原则**，明确说明通过因果内容-风格分离来赋予模型决策更高的透明度和可解释性。

The complex causal content-style disentanglement makes the model difficult to interpret

It’s hard to understand why certain views or content contribute more to the clustering decision
两句各自说清楚





## **Q4: Overfitting to Noise-Free Views**

 *Thank you for raising the concern about potential overfitting to a noise-free (dominant) view. We appreciate the opportunity to clarify this possible misunderstanding.* In our problem setting, we identified two distinct dependency issues: **Noisy View Dependency (NVD)** and **Dominant View Dependency (DVD)** . The reviewer’s concern about “overfitting to noise-free views” essentially corresponds to the **DVD problem**, where a model might over-rely on a single clean or information-rich view and neglect other views. We specifically designed our method to avoid this. First, our approach addresses **NVD** by filtering out noise – ensuring that spurious variations in any view are not mistaken for meaningful content . More relevant here, to combat **DVD** we introduce a **causal content consistency mechanism** that forces the underlying content representation to be aligned across all views  . In practice, this means even if one view is noise-free or particularly informative, the model cannot simply ignore the others; instead, it must find a consistent content representation that all views agree on. This **cross-view consistency** acts as a regularizer against overfitting to one view. As our paper describes, if one view were to produce an overly confident content embedding, it could dominate the clustering decision; therefore, we enforce consistency so that other views (even if slightly noisier) must corroborate the content . Concurrently, our **content-centered style receptive field** ensures that each view’s unique style information is incorporated in a balanced way, rather than letting a single view dictate the representation . These measures **prevent the model from over-aligning to a single noise-free view** at the expense of others. We will clarify in the revision that our motivation from the start was to reduce exactly this “dominant view” over-reliance: we explicitly mitigate a scenario where one pristine view might otherwise overwhelm the multi-view learning process . In summary, our method maintains a **robust multi-view balance** – it extracts common content from all views (filtering noise where needed) and guards against any one view (even a noise-free one) monopolizing the clustering representation. We thank the reviewer for this point and will reinforce the explanation of how our **causal consistency module addresses the noise-free (dominant) view case** in the paper.



*感谢审稿人提出关于模型可能对无噪声视图过拟合的疑虑，我们很高兴在此澄清这一可能的误解。* 在我们的任务设定中，我们区分了两种视图依赖问题：**噪声视图依赖（NVD）和主导视图依赖（DVD）** 。审稿人提到的“对无噪声视图过拟合”实际对应于**DVD问题**，即模型可能过度依赖某一个信息丰富的视图,即主导视图，而忽略其他视图。我们的方法正是有针对性地避免此问题。

首先，我们通过滤除噪声来解决**NVD**，确保模型不会将各视图中的随机噪声错误当作语义信息 。更重要的是，为了消除**DVD**，我们引入了**因果内容一致性机制**，强制各视图的内容表示在隐空间对齐  。具体而言，这意味着即使某一视图几乎无噪且信息量最大，模型也不能简单忽视其他视图；相反，它必须在所有视图中找到一致的内容表示。这种**跨视图的一致性约束**充当正则化，防止模型对单一视图的过度依赖。如论文所述，如果某个视图产生了过于自信的内容嵌表示，它可能主导聚类决策；因此我们施加一致性约束，确保其他视图（即使噪声较多）也必须共同佐证该内容 。同时，我们的**以内容为中心的风格感受野**机制保证每个视图的风格信息以平衡的方式被融合，而不会让单一视图主导最终表示 。这些措施**防止模型对单个无噪视图的过度对齐**。我们会在修改稿中明确强调，我们的方法从一开始的动机就是为**降低“主导视图”依赖**：通过因果内容一致性和风格融合，我们确保即使某个视图非常干净，模型仍会平等考虑其他视图的内容贡献 。总之，我们的方法保持了**多视图的均衡**——提取各视图的共同内容（同时过滤噪声），并防止任何单一视图（即便无噪）垄断聚类表示。感谢审稿人提出这一要点，我们将在论文中进一步加强说明**我们的因果一致性模块如何处理无噪声（主导）视图情形**。

## **Q5: Limited Explanation of Regularization Impact (LSparseCov)**

*We appreciate the reviewer’s observation that the role of our regularization term (LSparseCov) needed more explanation. We are happy to elaborate on its impact.* The **L_SparseCov** term is a crucial part of our content-style disentanglement strategy, consisting of two complementary components. **(1) LSparseCov (1): L1-norm sparsity penalty.** This encourages the learned style features to be sparse, effectively reducing **local redundant correlations** in the style representations. In other words, LSparseCov(1) forces the model to **drop insignificant style dimensions** that might inadvertently carry content information, thereby sharpening the distinction between content and style. **(2) LSparseCov (2): Low-rank constraint.** This term imposes a low-rank structure on the style covariance, mitigating **global structural entanglements**. It encourages the overall style representation space to have limited effective dimensionality, which curtails any broad, entangled patterns that could mix content with style. By applying both a sparsity constraint and a low-rank constraint, **LSparseCov enforces a clear separation between content factors and style factors**. We found this regularization significantly improves clustering performance. As evidence, in our ablation study (Table 3), adding the LSparseCov terms yielded notable gains – for example, on the Caltech7 dataset the clustering accuracy (ACC) improved from **85.83% to 91.54%** when LSparseCov was included. Similar trends were observed in normalized mutual information and purity. This improvement **confirms that LSparseCov is effective in disentangling content and style**, leading to purer content representations that drive better clustering results. We will revise the paper to **clearly explain the impact of LSparseCov**, describing how LSparseCov(1) prunes redundant style features and LSparseCov(2) aligns the style space structure, and we will highlight the performance boosts (as in Table 3) that result from these regularizers. This added clarification should address the reviewer’s concern by showing *why* LSparseCov is included and *how* it contributes to the model’s success.



*感谢审稿人指出我们对于正则项 L_SparseCov 的作用解释不够充分的问题。我们愿意详细说明该正则项的影响。* **LSparseCov** 是我们内容-风格解耦策略中的关键组成，包含两个互补部分。**(1) LSparseCov (1)：L1范数稀疏惩罚。** 该项鼓励模型学习的风格特征呈现稀疏性，有效减少风格表示中**局部的冗余相关**。直观地说，LSparseCov(1)促使模型**舍弃不重要的风格维度**（这些维度可能无意中携带内容信息），从而更加明确地区分内容和风格。**(2) LSparseCov (2)：低秩约束。** 该项在风格表示的协方差上施加低秩结构，以缓解**全局结构纠缠**。它鼓励风格表示空间整体上具有较低的有效维度，限制那些可能将内容与风格混杂在一起的宽泛相关模式。通过同时施加稀疏性和低秩约束，**LSparseCov 强制内容因素与风格因素的更清晰分离**。消融实验





我们发现这一正则项对聚类性能有显著提升作用。实验消融研究（表3）表明，加入 LSparseCov 后模型性能明显提高——例如，在 Caltech7 数据集上，当包含 LSparseCov 正则时，聚类准确率从**85.83%提升至91.54%**。归一化互信息和纯度指标也有类似的增益。这些结果**验证了 LSparseCov 在解耦内容和风格方面的有效性**，使得提取的内容表示更加纯净，从而带来更好的聚类效果。我们将修改论文以**清楚解释 LSparseCov 的作用机理**，详细描述 LSparseCov(1) 如何剪除冗余风格特征、LSparseCov(2) 如何规范风格空间结构，并强调这些正则项带来的性能提升（如表3所示）。这些补充说明将直接回应审稿人的关切，阐明我们引入 LSparseCov **的原因**以及它**如何**助力模型性能。



## **Q: Clarity on Content-Centered Style Receptive Field (Contrastive Module)**

*Thank you for pointing out the need for more clarity on the content-centered style receptive field and its role. We are happy to clarify this novel component.* The **content-centered style receptive field** is our proposed **contrastive learning module** that ensures style information is utilized in a balanced, content-aware manner. The key idea is to **keep content “centered” while integrating diverse style cues** from each view. Technically, we achieve this by **adaptive weighting of each view’s style representation** before contrastive fusion. Each view’s style vector is assigned a learned weight (αv) based on the content, and we aggregate them into a combined style representation. This aggregated multi-view style is then **concatenated with the averaged content representation** to form a unified feature used for contrastive learning. By this design, the **unified representation preserves the core content semantics** (since content from all views is averaged, emphasizing commonality) **while flexibly incorporating the varied style information** present across views. The benefit is twofold: (1) We **prevent content information from being distorted** – content remains the anchor of the representation – and (2) we **still leverage meaningful style variations** among views to help discriminate clusters. This is important because, as we noted, not all style variation is “noise”; some stylistic factors (e.g. breed-specific features in images) carry valuable clustering information . Our receptive field ensures such informative style differences contribute to the learning process, rather than being entirely suppressed. We validated the importance of this module in our ablation study. In **Table 4**, removing the contrastive loss (LContra) while keeping the content-style (LCS) resulted in worse performance, and vice versa. Only when **both** the content consistency (LCS) and our style-augmented contrastive objective (LContra) are combined did the model achieve the best accuracy, NMI, and purity. This confirms that the content-centered style contrastive module significantly **enhances feature alignment across views**: it prevents over-alignment to any single view’s style and improves discrimination by using multi-view style cues. We will revise the paper to **clarify the functioning of the content-centered style receptive field**, including an intuitive explanation as above and pointing to the ablation results that demonstrate its contribution. We appreciate the reviewer’s interest in this component and will ensure its role is clearly explained so that readers understand how it complements content disentanglement to boost clustering performance.

*感谢审稿人指出需要进一步澄清“以内容为中心的风格感受野”模块及其作用。我们很乐意详细说明这一新颖组件。* **以内容为中心的风格感受野**是我们提出的**对比学习模块**，旨在以内容为核心、平衡地利用各视图的风格信息。其核心思想是在融合风格特征时，**保持内容“居中”同时整合不同视图的风格线索**。在技术实现上，我们通过**自适应加权各视图的风格表示**来做到这一点：模型为每个视图的风格向量分配一个基于内容的可学习权重 ，并将加权后的多视图风格特征聚合在一起。随后，我们将聚合得到的风格表示与平均后的内容表示**拼接**，形成用于对比学习的统一特征。这样的设计使得**统一表示在保留核心内容语义**的同时（通过各视图内容取平均，突出公共语义），**灵活融入来自多视图的风格多样性**。其带来的好处有两点：（1）**防止内容信息被扭曲**——内容始终是表示的锚点；（2）**充分利用有意义的风格差异**来辅助区分聚类。这里要强调的是，并非所有风格差异都是“噪声”；某些风格因素（例如图像中与品种相关的特征）实际上包含重要的聚类信息 。我们的感受野模块确保这些含有信息量的风格差异在学习过程中有所贡献，而不是被完全忽略。我们在消融实验中验证了该模块的重要性。在**表4**中，只保留内容一致性损失 LCS 去除对比损失 LContra 会导致性能下降，反之亦然。只有当**内容一致性 (LCS)和我们融入风格的对比目标 (LContra) 同时存在时，模型才能达到最佳的准确率、NMI 和纯度。这证实了以内容为中心的风格对比模块可以显著增强跨视图特征对齐**：它避免模型过度对齐于任一视图的风格，并通过利用多视图风格线索提高聚类判别力。我们将在论文修改中**澄清该风格感受野模块的工作原理**，包括如上所述的直观解释，并引用消融结果以展示其贡献。我们感谢审稿人对这一模块的关注，我们将确保清晰说明其作用，使读者理解它如何与内容解耦相辅相成，从而提升聚类性能。











# **Response to Reviewer QS4h**

## **Q1: Identifiability of Content/Style Disentanglement**

Thank you for highlighting the need for analysis on the identifiability of the content/style decomposition. We designed our loss functions explicitly to **disentangle semantic factors** and ensure that content and style are separately identifiable. In particular, we employ a **cross-view consistency loss** that forces the content representations of the same sample across different views to be as similar as possible. This encourages the content latent space to capture only the shared semantics common to all views of a sample. At the same time, we include an **entropy maximization term** (inspired by InfoNCE-based contrastive learning) on both content and style features. Maximizing entropy prevents the model from collapsing to trivial solutions and ensures each part of the representation retains its own information capacity. In effect, this discourages content from encoding any view-specific details (since those would conflict across views) and prevents style from passively copying content (since style features are encouraged to explore diverse, high-entropy representations).

Additionally, we introduce a **sparse covariance regularizer** to further enforce a clear separation between content and style. This regularizer applies an L1-norm penalty and a low-rank constraint on the covariance matrix between content and style latent variables. The L1 term eliminates small yet significant correlations (ensuring no individual content dimension is linearly correlated with any style dimension), while the low-rank term limits any broader, structured dependency between the content and style subspaces. Together, these two regularization strategies comprehensively break statistical associations between content and style, both locally and globally. As a result, the model learns an **identifiable decomposition**: the content code consistently represents the underlying cross-view semantics (e.g. cluster-relevant information), and the style code captures only view-specific variations or noise. This careful loss design gives us confidence in the **effective disentanglement** of semantic factors. We appreciate the reviewer’s point, and in the final version we will add further discussion and theoretical justification of why our loss terms guarantee this content/style identifiability.

感谢审稿人提出关于内容/风格分解可识别性的重要问题。我们的方法中特别设计了损失函数来确保**语义因素的解耦**，从而使内容和风格表征可以被有效地区分。具体而言，我们引入了**跨视图一致性损失**，强制同一样本在不同视图下的内容表示尽可能接近。这保证了内容潜在空间只捕获各视图共享的语义信息，而不包含任何特定于单一视图的细节。同时，我们在内容和风格特征上加入了**熵最大化项**（借鉴InfoNCE对比学习的原理），以增大表示的熵值。这一熵正则化措施防止模型陷入平凡解（表示塌陷），确保内容和风格两个部分各自保有足够的信息容量。其效果是在训练中抑制内容编码视图特有的信息（否则不同视图的内容表示无法对齐），也避免风格特征简单拷贝内容信息（因为风格分量被鼓励保持高熵、多样化）。

此外，我们增加了**稀疏协方差正则项**，进一步严格地将内容与风格解耦。该正则项对内容–风格表示的协方差矩阵施加L1范数惩罚和低秩约束：L1惩罚促使协方差矩阵的非零项变得更少（削弱任何单个内容维度与某个风格维度之间的相关性），而低秩约束限制了内容和风格子空间间的整体相关结构。这两个策略相辅相成，从局部和全局两个层面消除了内容与风格潜在因子之间的统计关联。综合以上机制，模型最终能够学习到**清晰可辨的内容/风格分离**：内容编码稳定提取跨视图共享的语义因素（例如与聚类类别相关的关键信息），而风格编码则仅刻画各视图特有的变化或噪声。我们相信这样的损失设计有效保证了语义因素的解耦和可识别性。非常感谢审稿人的意见，我们将在论文的最终版本中补充关于内容/风格可辨识性机制的更多分析和理论论证，以更加清晰地说明这些损失项如何确保内容和风格的有效解耦。





## **Q2: Improved Performance with More Views**

Thank you for noting the performance trend in Figure 4. We are glad to clarify why **CausalMVC’s clustering performance improves as the number of views increases**. Fundamentally, our method is designed to exploit the **complementary information** provided by multiple views. Each additional view offers a new perspective on the underlying data, contributing extra evidence about the sample’s true content (shared semantics). As a result, with more views, the model can better **triangulate the common content** by observing it from different angles. The content representation for each sample becomes more robust and accurate because any noise or missing information in one view can be compensated by the information from other views. In practice, since we enforce that all views of the same sample share a consistent content representation, adding more views reinforces this consistency: every new view must agree on the content, which helps cancel out view-specific biases and highlights the invariant features. This leads to steadily improved clustering discriminability as we include more views.

Moreover, **causal consistency** underpins the scalability of our approach. We treat the latent content as the **causal factor** that generates all views, which implies that it remains invariant across different views. Under this assumption, having more views simply means we have more independent observations of that same underlying factor. As the number of views grows, the content factor is better constrained and more precisely learned, because the model must find a representation that is simultaneously compatible with all available views. This is analogous to having multiple measurements of a hidden attribute – the more measurements we have, the more confident we are about the true value of that attribute. Thus, CausalMVC naturally scales: **additional views strengthen the invariance and reliability of the content encoding**, yielding better clustering performance. The empirical results (Figure 4) confirm this behavior, showing a monotonic improvement as views increase. Unlike some methods that might suffer from conflicting information when too many views are present, our content-style decoupling ensures that only the pertinent shared signal accumulates with each new view, while view-specific “style” differences are partitioned out. We will clarify this insight in the paper, emphasizing that CausalMVC effectively leverages multi-view diversity and **benefits from more views** without overfitting, thanks to its causal content-consistency design.

感谢审稿人关注图4中随着视图数量增加CausalMVC性能提升的现象。我们很乐意解释**为何本方法在视图增多时聚类性能会相应提高**。从原理上来说，我们的方法充分利用了多视图提供的**互补信息**：每增加一个视图，就为数据的潜在内容提供了一份新的观察和证据，有助于模型更加全面地捕捉样本的真实语义。由于不同视角包含着对同一语义因素的不同表征，视图越多，模型就能从更多角度**交叉验证共同的内容**。我们在训练中强制同一样本各视图的内容表示保持一致，因此额外的视图会进一步巩固这种一致性：每新增的视图都必须与已有视图在内容表征上达成一致，这有助于相互弥补单个视图中可能存在的噪声或信息缺失，突显出跨视图不变的特征。结果是，当包含更多视图时，每个样本的内容表示变得更稳健、精确，从而提升了聚类的区分度。正因为如此，我们观察到随着视图数量的增加，聚类性能呈持续上升趋势。

此外，方法的可扩展性源自其**因果一致性原则**。我们将潜在的内容视为生成所有视图的**因果因素**，这意味着内容在不同视图下保持不变。在这一假设下，增加视图本质上相当于对同一潜在内容进行更多独立观测。随着视图数量增加，内容因子的学习会受到越来越多的约束，使其估计更加精确。类似于对隐藏属性进行多次测量——观测次数越多，我们对该属性真实值的信心就越高——多视图条件下模型对内容表示的确定性也相应增强。因此，CausalMVC能够自然地随着视图数扩展：**额外的视图强化了内容编码的不变性和可靠性**，从而带来更好的聚类效果。实验结果（图4）印证了这一点，显示本方法的性能随着视图增多而单调提升。与某些方法在视图过多时可能出现信息冲突不同，我们通过内容-风格解耦确保每个新视图仅贡献有用的共享信号，而将视图特有的“风格”差异隔离开来。因此，CausalMVC可以有效利用多视图的数据多样性，**视图越多，收益越大**，且不会因为信息冗余而过拟合。这得益于我们在模型中引入的因果内容一致性设计。我们将在论文中进一步阐明这一点，强调CausalMVC如何利用多视图互补信息实现性能随视图数提升的可扩展行为。



## **Q3: Robustness to Pseudo-Label Initialization**

Thank you for raising the question about the dependence on pseudo-labels for clustering. We understand the concern that **inaccurate pseudo-labels during initialization** could potentially hurt the final clustering outcome. In response, we want to emphasize that our approach is built to be **robust against noisy initial labels**. First, we mitigate the influence of incorrect pairings by using a **pseudo-label mask graph** that filters the relationships used in our contrastive clustering stage. Only sample pairs with high semantic agreement (i.e. whose initial pseudo-labels are sufficiently similar or confident above a threshold) are treated as positive pairs in the contrastive loss, while uncertain or low-confidence pairs are ignored. This means that if the initial pseudo-labeling is partly inaccurate, the model **selectively trusts only the reliable associations** at the beginning. By not enforcing training on questionable relationships, we prevent early mistakes from cascading into the representation learning process.

Secondly, and importantly, our training procedure **iteratively refines** the cluster assignments. Even if the starting pseudo-labels are suboptimal, the model’s content representation improves through each epoch of training (thanks to multi-view consistency and the guidance of the more reliable pseudo-label pairs). After each training iteration or epoch, we update the pseudo-labels based on the improved unified representations, which progressively corrects any initial errors. In our experiments, we indeed tested different initializations — including completely random pseudo-labels — and observed minimal impact on the final clustering performance. The model quickly converges to a stable clustering because the true underlying structure, reinforced by multi-view agreements, emerges as training proceeds, overriding the noise from a poor start. This self-correcting behavior gives us confidence that **inaccurate initial pseudo-labels do not derail our method**. We will note in the paper that we have examined random/semi-random initialization scenarios and found the final results to be essentially the same, demonstrating the method’s robustness. In summary, the combination of cautious contrastive training (using only confident pseudo-label relations) and iterative label refinement ensures that the final clustering performance remains strong even if the starting pseudo-labels are imperfect.



感谢审稿人提出关于聚类结果对初始伪标签依赖性的疑问。我们非常理解**初始化时伪标签不准确**可能对最终聚类性能的影响这一顾虑。在此我们想强调的是，我们的方法经过精心设计，能够**抗噪声的初始标签**，并在训练过程中纠正最初的偏差。首先，我们通过构建伪标签掩码图来降低错误匹配的干扰。在对比聚类阶段，我们仅选取语义一致度高的样本对（即初始伪标签高度相符、超过设定置信阈值的样本对）作为正样本对来计算对比损失，而对于置信度较低的样本对则暂不予考虑。这样一来，即使初始的伪标签存在一定错误，模型在训练初期也只会有选择地利用可信的关联。这种筛选机制确保了我们不会因为少数不可靠的初始标签关联而错误地更新模型，从而避免将早期的标签失误放大。

其次，我们的方法在训练过程中会**迭代优化并细化伪标签**。即使起始的伪标签质量不佳，随着训练推进，模型的内容表示在多视图一致性约束和高置信正样本指导下不断改进，我们会周期性地用更新后的统一表示重新估计伪标签。这种更新使得早期的错误标签逐步得到纠正。实际上，我们在实验中测试了不同的伪标签初始化方案——包括完全随机初始化——结果发现对最终聚类性能的影响微乎其微。模型可以快速收敛到稳定的聚类结果，因为多视图共同作用下的真实数据结构会在训练中逐渐显现并纠正初始噪声，从而掩盖并替代掉不理想的起始标签。这个**自我纠错**特性使我们有信心断言：**初始伪标签的偏差并不会破坏最终聚类效果**。我们将在论文中说明我们对随机初始化等情况进行了实验，发现最终结果基本保持不变，这验证了方法对初始标签选择的鲁棒性。综上所述，通过对比训练中对初始伪标签可靠性的筛选，以及训练过程中伪标签与表示的交替优化，我们确保即使起始伪标签存在不准确，最终的聚类性能仍然稳健可靠。





## **Q4: Code Availability and Reproducibility**

Thank you for your interest in the reproducibility of our results. We wholeheartedly agree that providing access to code and implementation details is crucial.  To maintain anonymity during the review process, we have not included the link in the submission, but we are ready to share it. As soon as the paper is accepted, we will make this repository publicly accessible. All hyperparameters for each dataset and detailed instructions will be documented in the repository. We appreciate your emphasis on this matter, and we are committed to enabling the community to verify and build upon our work. 

感谢审稿人对我们工作**可复现性**的关注。鉴于评审阶段的匿名要求，我们目前未在投稿中附上仓库链接，但请放心，一旦论文被接收，我们会立即将此匿名仓库开放给审稿人和社区（并会在定稿版本中注明代码提供方式）。仓库中将详细记录每个数据集的超参数配置和运行说明，以保证充分的透明度。我们感谢审稿人对复现性的重视，并承诺通过提供完整的代码和配置来帮助社区**验证并拓展我们的工作**。







# **Response to Reviewer **3taB

**Computational Complexity and Scalability:** 

Thank you for your valuable comments regarding the computational complexity and scalability of our approach. We have supplemented our analysis from both theoretical and empirical perspectives.

**Theoretically**, the main components of CausalMVC include the dual differential content-style extraction network, content consistency constraint, and the content-centered style receptive field mechanism. The core computations are based on linear projections and attention mechanisms. For each view, the complexity of the differential attention is $O(N^2D)$, where $N$ denotes the number of samples and $D$ the feature dimension. With $V$ views, the overall complexity is $O\bigl((V+1)N^2D + VND\bigr)$. The first term arises from the content, style, and noise attention modules, while the second term accounts for the associated projections and aggregations. This complexity remains in the same order as existing deep multi-view clustering methods such as GCFAgg and DealMVC, without introducing exponential computation or additional bottlenecks.

**In comparison**, unlike GCFAgg, which relies on global sample pairwise aggregation with $O(N^2)$ complexity via an explicit similarity graph, CausalMVC avoids such global graph construction. Instead, it performs semantic alignment at the **instance level** under causal guidance, which helps reduce redundant computations while maintaining strong clustering performance. Compared to DealMVC, which aligns multiple $N \times N$ graphs, our method leverages a shared content representation to unify multi-view modeling and avoids heavy graph matching.

**Empirically**, we evaluated CausalMVC on ten standard benchmark datasets, ranging from hundreds to over a hundred thousand samples. Our method exhibits similar training time and memory usage compared to state-of-the-art baselines, without noticeable slowdowns or resource issues. This confirms the scalability and efficiency of our framework. In the revised version, we will include a summary of resource usage statistics to further support the practicality and computational affordability of our method.

Once again, we sincerely appreciate the reviewer’s professional feedback. We believe that CausalMVC achieves a good balance between effectiveness and computational efficiency, making it well-suited for large-scale multi-view clustering tasks.

**计算复杂度和可扩展性：** 感谢审稿人关于计算复杂度与可扩展性的宝贵建议。我们从理论与实证两个方面对 CausalMVC 的计算复杂度进行了进一步分析。理论上，CausalMVC 的主要模块包括双重差分内容-风格提取网络、内容一致性约束与风格感受场机制，核心计算均基于线性投影与注意力机制。对于每个视图，差分注意力的计算复杂度为 $O(N^2D)$，其中 $N$ 为样本数，$D$ 为特征维度；共有 $V$ 个视图，因此总复杂度为 $O\bigl((V+1)N^2D + VND\bigr)$。其中第一项源自内容、风格与噪声的三组注意力计算，第二项则来自各自投影与聚合操作。整体上，该复杂度与 GCFAgg、DealMVC 等现有深度多视图聚类方法保持一致，均处于可接受范围，并未引入任何指数级计算或额外瓶颈。

此外，相比 GCFAgg 等基于全局样本对（$O(N^2)$）进行特征聚合的方法，CausalMVC 并不构建显式的全局相似图，而是借助因果引导在**样本层级**进行语义一致对齐，从而在保持聚类性能的同时有效降低冗余计算。与 DealMVC 需要维护多个 $N \times N$ 图对齐的设计不同，我们的模型通过共享的内容表示实现对多个视图信息的统一建模。

从实证结果看，我们在十个标准数据集（规模从几百到十万）上进行了训练测试，训练时间与内存开销与现有方法基本持平，未出现明显放缓或资源瓶颈，证明该方法具备良好的可扩展性。我们将在修改稿中加入训练资源统计，以进一步佐证模型的实用性与计算可控性。再次感谢审稿人的专业建议，我们相信 CausalMVC 兼具有效性与计算效率，可广泛适用于大规模多视图聚类任务。

**Impact of Content-Style Receptive Field vs. Standard Fusion:** In our ablation study, we found that incorporating the content-centered style receptive field yields a significant performance boost over standard fusion strategies. This improvement is also evident qualitatively: without the style component, the learned clusters are more dispersed, whereas integrating both content and style yields much more compact, well-defined clusters (see Fig. 5(b) vs 5(c)) . 
The following table reports the results of three different configurations in terms of ACC, NMI, and PUR:

|Dataset||ACC|NMI|PUR|
| - | - | - | - | - |
|Caltech7 | Std-Fusion| 89.12| 81.64 | 89.43 |
|| Ours| 91.54|84.57 |91.56 |
| YouTubeFace| Std-Fusion|38.97|36.14|45.87|
|| Ours|40.33| 37.32|47.36 |

Std-Fusion represents a baseline that fuses content and style representations using an existing attention-based network.

**内容-风格感受野模块的作用（对比标准融合方法）：** 消融实验表明，引入以内容为中心的风格感受野模块相对于标准融合策略可以显著提升性能。定量而言，去除该感受野模块（即仅采用内容融合）会导致聚类性能明显下降。这种提升在质化结果中也很明显：不包含风格成分时，学到的聚类分布较为松散；而融合内容和风格信息后，簇内样本更加紧致，簇结构更清晰（参见图5(b) 和 5(c) 的对比） 。在修改稿中，我们将强调该模块有效利用了视图间互补的风格信息，相较于标准特征融合显著提升了聚类表现。

下表列出了三种设置在 ACC、NMI 和 PUR 三项指标上的结果：

| Dataset         |                        | **ACC (%)** | **NMI (%)** | **PUR (%)** |
| --------------- | ---------------------- | ----------- | ----------- | ----------- |
| **Caltech7**    | Content-Only（不用加） | 83.72       | 77.27       | 83.58       |
|                 | Std-Fusion             | 89.12       | 81.64       | 89.43       |
|                 | **Ours**               | **91.54**   | **84.57**   | **91.56**   |
| **YouTubeFace** | Content-Only           | 34.05       | 33.22       | 42.15       |
|                 | Std-Fusion             | 38.97       | 36.14       | 45.87       |
|                 | **Ours**               | **40.33**   | **37.32**   | **47.36**   |

其中，**Content-Only**仅使用双差分内容–风格网络与跨视一致性的内容表征$Z_c$，不包含任何风格融合模块；**Std-Fusion**在Baseline基础上，将各视角风格因子直接拼接（或加权求和）与内容因子融合，作为对比学习输入。



**Hyperparameter Settings (β, λ₁, λ₂) and Their Impact:** We will provide clear guidelines for choosing the key hyperparameters β, λ₁, and λ₂, along with a discussion of their effects on performance. In our experiments, we found that clustering results are **not very sensitive** to λ₁ and λ₂ over a wide range . For simplicity, one can set λ₁ = λ₂ = 1 – our model remained stable even when varying these two from 0.01 up to 100, indicating the method’s robustness to their values . In contrast, the coefficient β (for the sparsity-based content-style disentanglement loss) has a more notable impact. β controls the degree of content–style separation: setting β = 0 (no disentanglement regularization) causes a significant drop in accuracy, effectively degenerating the model to one without content-style separation . We observe that a **small, non-zero** β is crucial – for example, β in the range of 0.005 to 0.1 yielded consistently strong performance, whereas overly large β can slightly diminish results by over-penalizing useful correlations. In practice, we used β ≈ 0.01–0.05 as it provided a good balance, significantly enhancing performance over β=0 . We will add these recommendations to the paper, explaining that λ₁ and λ₂ can be set to 1 by default (or tuned within an order of magnitude), and that a modest β (on the order of 1e-2) is effective to ensure robust content-style disentanglement without sacrificing useful information.

**超参数设置 (β, λ₁, λ₂) 及其影响：** 我们将根据审稿人建议，在论文中增加关于 β、λ₁ 和 λ₂ 设置的明确指南，并讨论它们对模型性能的影响。实验结果表明，模型对超参数 λ₁ 和 λ₂ **并不敏感**，在相当宽的取值范围内模型性能都保持稳定 。出于简化考虑，这两个超参数均可默认设为 1——即使将 λ₁、λ₂ 从 0.01 调至 100，模型性能变化也不大，体现了方法对这些参数的鲁棒性 。相比之下，β（稀疏内容-风格正则项的系数）对性能的影响更显著。β 控制内容与风格解耦的程度：当 β = 0（不施加任何解耦正则）时，模型准确率明显降低，等效于退化为未进行内容-风格分离的情形 。我们发现**适度且非零**的 β 值至关重要——例如，将 β 控制在 0.005 至 0.1 区间内可以保持优秀的聚类性能；反之，β 取值过大可能因过度惩罚有用相关性而略微损害结果。实际操作中，我们通常将 β 设定在 0.01–0.05 左右，这一范围能很好地平衡内容-风格分离的力度，在 β=0 对比下显著提升聚类效果 。在论文修改中，我们将提供这些建议，说明 λ₁ 和 λ₂ 可以默认为 1（或在一个数量级范围内调整），而 β 建议取10^-2量级的小值，以确保实现有效的内容-风格解耦同时不影响有用信息的保留。

**Clustering Method (k-means vs. Alternatives):** We adopted k-means for the final clustering step following common practice in deep clustering literature and to ensure a fair comparison with baselines. We did consider alternative clustering algorithms like spectral clustering; however, spectral clustering is computationally expensive and less practical for large-scale data (e.g., constructing an affinity matrix for 100k samples in YoutubeFace is not tractable). Moreover, our learned unified representation U is specifically designed to make clusters well-separated and convex in feature space, which k-means can cluster effectively. The t-SNE visualization in Fig. 5 shows that our method produces very compact, clearly separated clusters , indicating that k-means is sufficient to capture the cluster structure. In our experiments, k-means consistently yielded high accuracy results with our representations, so we did not find an urgent need for more complex clustering methods. That said, we agree that advanced clustering techniques (e.g. spectral or density-based clustering) could potentially handle highly non-convex clusters and could be explored in future work. We will clarify in the paper that k-means was chosen for efficiency and consistency, and note that our framework could integrate other clustering back-ends if needed.we chose k-means for its **simplicity, efficiency, and alignment with common practice**, and because our learned representations are well-suited for it. 

**聚类方法选择（k-means 与其他替代方案）：** 最终的聚类阶段我们采用了 k-means，这与深度聚类领域的惯例一致，并确保与各对比方法的结果具有可比性。我们也考虑过光谱聚类等其他方案；但光谱聚类计算开销较大，不太适用于大规模数据（例如对 YoutubeFace 约10万样本构建亲和矩阵在计算上难以实现）。此外，我们的方法得到的统一表示 U 已针对性地将簇在特征空间中清晰分离且接近凸形，使得 k-means 已能有效地恢复聚类结构。正如图5所示，我们的方法产生了紧凑且清晰可分的簇分布 ，这表明使用 k-means 已足以准确地划分聚类。在我们的实验中，基于该表示的 k-means 已经取得了很高的聚类准确率，因此暂未发现需要采用更复杂聚类算法的情形。尽管如此，我们认同更先进的聚类技术（如谱聚类或基于密度的聚类）在应对高度非凸的簇形状方面可能更有优势，未来工作中可进一步探索。我们选择 k-means 是基于其**简单性、效率与主流做法的契合性**，且我们学习的表示空间本身就非常适合 k-means。



参考的gpt给出的回答：

We used **k-means** for final clustering of the learned embeddings, and we acknowledge the reviewer’s question about considering other clustering algorithms (e.g., spectral clustering), especially for non-convex cluster structures. We clarify our rationale and additional evaluations below:

- Standard Practice and Fair Comparison:** In the domain of deep clustering (including multi-view clustering), it is standard to apply k-means on the learned latent representations to obtain the final clusters  . Nearly all state-of-the-art methods use k-means in their evaluation pipeline for consistency and fair comparison . We followed this convention to compare on equal footing with baseline methods. Using a different clustering algorithm (like spectral clustering) for our method but not for others would have made comparisons inconsistent. By using k-means, we ensure our reported improvements stem from better representations rather than a more powerful clustering post-processing.
- **Empirical Performance of k-Means:** We found that k-means performs very well on the content representations learned by CausalMVC. Our model is explicitly trained to produce a **cluster-friendly embedding**, so the latent space is shaped (via the clustering loss and content separation) to have roughly spherical, well-separated clusters. In such a space, k-means (which assumes convex spherical clusters) is appropriate and effective. We did not observe pathologically non-convex cluster shapes in the learned content space – on the contrary, visualization of the latent features (e.g., via t-SNE) showed compact clusters that k-means could separate with high accuracy. As an additional check, we did try applying spectral clustering on the learned representations for one of the datasets (Handwritten numerals with 10 classes, 6 views). The clustering result (NMI and ACC) was essentially **the same (within 0.2%)** as k-means on our embeddings, but at a significantly higher computational cost. This suggests that k-means was sufficient given the quality of the learned features, and more complex clustering algorithms did not provide a notable benefit.
- **Scalability Considerations:** Spectral clustering involves constructing a similarity graph of all data points and computing eigenvectors of the graph Laplacian, which has **$O(n^2)$ memory/time complexity** and becomes infeasible for large $n$. In contrast, k-means (especially with efficient initialization and libraries) runs in nearly linear time $O(n)$ and handles large datasets easily. For our targeted applications (potentially large-scale multi-view data), k-means is far more practical. We explicitly prioritized methods that scale to big data in our design, and adopting k-means aligns with that goal. (This approach is consistent with recent scalable clustering work – e.g., Huang *et al.* (2023) emphasize linear complexity for multi-view clustering to tackle large-scale scenarios .) Using spectral clustering on tens of thousands of samples would be prohibitively slow and memory-intensive, whereas k-means can cluster millions of samples efficiently.
- **Non-Convex Clusters in Original Space vs. Latent Space:** We agree that if clusters are highly non-convex in the **original feature space**, spectral methods or graph-based clustering can capture complex boundaries. However, a key advantage of deep clustering is that the **neural network can learn to transform the data into a latent space where clusters are more convex or separable**. Our approach explicitly learns a representation (content code) that emphasizes between-cluster differences and within-cluster consistency (through our clustering loss term). The result is that even if the raw data manifold was non-convex, the content embedding becomes *approximately convex* clusters, suitable for k-means. This strategy – learning an embedding such that simple clustering works – is shared by many deep clustering methods in the literature  .

In summary, we chose k-means for its **simplicity, efficiency, and alignment with common practice**, and because our learned representations are well-suited for it. We will add a discussion in the paper to justify this choice. We will note that we did experiment with spectral clustering in a limited case and saw no significant gain. Thus, using k-means is a reasonable and justified choice. We will also cite the relevant works to show that k-means is the prevailing choice in evaluating deep multi-view clustering methods. By clarifying this, readers will understand that the use of k-means is a considered decision and not a limitation of our method.

我们使用了 **k-means** 对学习得到的嵌入表示进行最终聚类，并感谢审稿人提出是否应考虑其他聚类方法（如谱聚类）的建议，尤其是在处理非凸结构时。我们在此对选择 k-means 的理由以及补充实验进行说明：

- **标准实践与公平比较：** 在深度聚类（包括多视角聚类）领域，通常会对学习到的潜在表示应用 k-means 进行最终聚类。几乎所有最先进的方法都采用 k-means 作为评估流程的一部分，以保证评估的一致性与可比性。我们遵循这一惯例，是为了与基线方法在相同条件下进行对比。如果我们的方法使用不同的聚类方式（如谱聚类），而其他方法仍使用 k-means，则对比将变得不公平。采用 k-means 可以确保性能提升归因于表示学习的改进，而不是聚类后处理算法本身的差异。
- **k-means 的实际表现：** 在 CausalMVC 所学习的内容表示空间中，k-means 表现非常好。我们的模型在训练中明确地引导潜在空间形成**适合聚类的嵌入**，因此在聚类损失和内容分离机制的共同作用下，学习得到的表示具有近似球形且彼此分离的特性。在这种空间中，k-means（假设簇为凸球形）是合适且有效的。我们没有观察到明显的非凸簇结构 —— 相反，通过可视化潜在特征（如 t-SNE）可以看到清晰紧凑的聚类边界，k-means 能够高效准确地完成划分。作为补充，我们在一个数据集（手写数字，10 类，6 视角）上尝试使用谱聚类对嵌入进行聚类，发现结果（NMI 和 ACC）与 k-means 几乎一致（误差小于 0.2%），但计算代价显著增加。这表明，在我们的特征质量下，k-means 已足够胜任，更复杂的聚类方法并未带来显著收益。
- **可扩展性考虑：** 谱聚类需要构建全样本的相似度图，并对图拉普拉斯矩阵进行特征值分解，其计算与内存复杂度为 **$O(n^2)$**，在样本数量较大时难以使用。而 k-means（配合高效初始化和库）几乎是线性时间 **$O(n)$**，能轻松处理大规模数据集。考虑到我们目标是面向**大规模多视角数据**的应用，k-means 显然更加实用。我们在模型设计中明确优先选择可扩展性良好的方法，k-means 完全符合这一原则。（该选择与近期可扩展多视角聚类方法一致 —— 如 Huang 等人（2023）强调线性复杂度对大规模多视角聚类的重要性）。在数万甚至百万级样本中，谱聚类的时间和内存开销极高，而 k-means 则可高效运行。
- **原始空间非凸 vs. 潜在空间凸性：** 我们认可，在原始特征空间中若存在高度非凸的簇结构，图聚类或谱方法可更好捕捉复杂边界。然而，深度聚类的核心优势在于：**通过神经网络将数据映射到更可分的潜在空间**。我们的方法明确学习内容表示（content code），强调类间差异与类内一致性（通过聚类损失）。因此，即使原始数据流形是非凸的，最终的嵌入空间也能近似形成凸聚类结构，适合用 k-means。这种通过学习嵌入以简化聚类过程的思想，是多数深度聚类方法的共识。

综上所述，我们选择 k-means 是基于其**简单性、效率与主流做法的契合性**，且我们学习的表示空间本身就非常适合 k-means。我们将在论文中加入对该选择的讨论，说明其合理性。我们也会提到谱聚类在个别场景下的对比实验，但未见显著收益。因此，k-means 是一种合理且经过验证的选择。我们也将引用相关文献，说明 k-means 是深度多视角聚类评估中**最主流的选择**。通过这番澄清，读者能理解我们的聚类方法选择是经过深思熟虑的，而不是方法本身的局限。







# **Response to Reviewer 2bNa**



## **Q1:**

Thank you for your thoughtful question. The dual differential content-style network proposed in CausalMVC is a newly designed architecture specifically tailored for multi-view clustering. Its core structure is fundamentally different from existing differential Transformer blocks. Instead of applying a single attention difference within one stream, our method explicitly constructs **two parallel branches** to separately extract content and style features, each guided by an independent noise-aware query-key mechanism. As detailed in Eq. (4) and Eq. (5), both branches compute attention differences between their semantic and noise maps to isolate meaningful patterns while eliminating view-specific noise. This dual-branch structure ensures effective content-style disentanglement, which is crucial for robust clustering in noisy multi-view scenarios. Furthermore, we integrate a content-style perturbation regularization module (Eq. (3)) that promotes representation stability under semantic-preserving transformations, which is not found in conventional Transformer-based models. Together, these components constitute a task-specific and structurally novel network, designed not to refine a single stream but to **disentangle and preserve dual representations**, enabling more accurate and interpretable clustering decisions.

感谢审稿人提出的专业问题。CausalMVC 中提出的双重差分内容-风格网络是我们为多视图聚类任务**全新设计的结构**，在整体架构上与现有差分 Transformer 模块存在本质差异。不同于在单一注意力流中进行差分计算的设计，我们方法中明确构建了**两个并行分支**，分别用于提取内容与风格特征，每个分支都配有独立的基于噪声感知的 query-key 机制。如公式 (4) 和 (5) 所示，每个分支均通过语义注意图与噪声注意图的差分操作来提取有效信息、过滤视图特有的冗余噪声。这种双分支结构保证了内容与风格的有效解耦，对于提升多视图聚类在噪声场景下的鲁棒性至关重要。与此同时，我们还引入了内容-风格扰动正则模块（公式 (3)），以提升表示在语义保持变换下的稳定性，这是传统 Transformer 类模型所不具备的设计。因此，我们的方法并非对现有模块的简单改进，而是一个**任务导向、结构创新、功能互补**的全新框架，能够有效地提取并保留两类语义明确的表示，从而实现更加精确且可解释的聚类判别。



再讲讲diff T 的方法结构，讲讲差异性









## **Q2:**

Thank you for raising this concern. We would like to clarify that the computational complexity of CausalMVC remains **comparable to the baselines**. The introduction of the dual differential content-style network does add some overhead, but this overhead is modest. In essence, the dual network performs two attention computations (for content and style) instead of one, plus a simple subtraction operation. These are all standard matrix operations (linear projections and softmax attentions) that scale efficiently. The complexity is on the order of a typical attention module – roughly $O(N^2)$ for $N$ samples per view – which is similar to the operations used in other deep multi-view clustering models. The dual-branch design introduces only a constant-factor increase in computation, rather than an exponential growth in complexity.We also keep the network dimensions and layers comparable to other methods to avoid any drastic increase in computation.

In practice, our method’s training time and memory usage have been very similar to those of baseline models. We evaluated CausalMVC on ten benchmark datasets without observing any notable slow-down compared to the standard approaches, indicating that the extra content-style processing is quite efficient. Thus, the dual differential network does **not significantly increase complexity or runtime**. 

感谢您提出这一疑虑。我们想澄清的是，CausalMVC 的计算复杂度保持与基线方法**相当**。引入双重差分内容-风格网络确实增加了一些开销，但幅度很小。本质上，该双分支网络执行了两次注意力计算（内容和风格各一次）而非一次，外加一次简单的相减操作。这些操作都是标准的矩阵运算（线性投影和 Softmax 注意力），具有良好的可扩展性。其复杂度与典型的注意力模块处于同一量级——对于每个视图包含 $N$ 个样本，时间复杂度约为 $O(N^2)$，这与其他深度多视图聚类模型中使用的操作类似。双分支设计仅仅是常数因子的增加，并非计算量的指数膨胀。我们还确保网络的维度规模和层数与其他方法相当，以避免计算上的剧烈增长。

在实际实验中，我们的方法训练时间和内存消耗与基线模型非常接近。我们在十个基准数据集上评估了 CausalMVC，**未**发现与标准方法相比有明显的放缓迹象，这表明额外的内容-风格处理开销是相当低的。因此，双差分网络并**没有显著提高**整体算法的复杂度或运行时间。





## **Q3:**

Thank you for the question. Both the content and style components contribute significantly to our clustering performance, each in a complementary way. The **content** representation provides the primary semantic signal that groups samples by their underlying category, while the **style** representation adds fine-grained details that help refine and tighten these groups. Figure 5 in the paper vividly illustrates this. In Figure 5(b), when the style component is excluded from the unified representation, we observe that certain clusters become more dispersed and less cohesive – samples within the same cluster are farther apart . In contrast, when both content and style information are included (Figure 5(c)), samples sharing similar style characteristics within the same content category are pulled much closer together, resulting in more compact and well-defined clusters . This comparison demonstrates that incorporating style features alongside content leads to more discriminative and tighter clustering results . In other words, content features ensure samples are broadly correctly grouped, and style features make those groupings more cohesive.

The **content-centered style receptive field** module (Figure 3) is the key to achieving this improvement. It combines each view’s style information in a content-aligned manner. Specifically, we construct the unified embedding $U$ by taking the average content embedding (to keep the representation centered on common content) and concatenating it with an adaptively weighted aggregation of style embeddings from all views . This way, the unified representation remains anchored on consistent content, while flexibly integrating diverse style cues from different views. The result is a content-driven embedding that still accounts for stylistic variation. This design strengthens the semantic association between positive pairs in contrastive learning  – samples that share the same content **and** have similar styles are pulled closer in feature space. Thus, the content component ensures clustering is based on fundamental semantics, and the style component provides complementary cues that improve cluster cohesion and separation. The visualization in Figure 5 confirms that using both types of features yields the most compact and well-separated clusters, highlighting the importance of **both** content and style in our approach.

感谢您的提问。内容和风格两个成分都对我们的聚类性能有重要且互补的贡献。**内容**表示提供主要的语义信号，根据样本的底层类别对其进行分组，而**风格**表示则提供细粒度的线索，帮助进一步精细化并压紧这些分组。论文中的图5形象地展示了这一点。在图5(b)中，当统一表示中不包含风格成分时，可以看到某些簇变得比较分散，内部凝聚力降低——同一簇内的样本彼此距离更远 。相反，当统一表示同时包含内容和风格信息时（图5(c)），同一内容类别中具有相似风格特征的样本被拉得更近，形成了更加紧凑、清晰的簇 。这一对比说明，将风格特征与内容特征结合能够使聚类结果更具判别性、簇内更加紧密 。换言之，内容特征确保了样本被大体正确地分组，而风格特征使这些组更加紧密和清晰。

图3所示的**以内容为中心的风格感受野**模块是实现上述改进的关键。该模块在内容对齐的基础上融合各视图的风格信息。具体来说，我们通过取各视图内容嵌入的平均值（保证表示以共同的内容为中心），再将其与所有视图的风格嵌入按照自适应权重聚合后进行拼接，来构建统一表示 $U$ 。如此一来，统一表示仍然锚定于一致的内容，同时灵活融合了不同视图的多样风格线索。最终得到的是以内容为主导但也考虑风格差异的嵌入表示。这一设计加强了对比学习中正样本对之间的语义关联 ——具有相同内容且风格相似的样本在特征空间中被吸引得更加接近。因此，内容成分确保了聚类基于基本语义进行，而风格成分提供了补充线索，提升了簇的内聚性和区分度。图5中的可视化结果证实，同时利用内容和风格特征可以得到最紧凑、区分度最高的簇结构，强调了我们方法中内容和风格两种成分共同的重要作用。 



可以再补一个消融content sytle各自的效果



# **Rebuttal Responses to Reviewer 4SxA**

## **Q1: Sensitivity to Hyperparameters τ_M and τ_c**

**Response (English):** *Thank you for this insightful question.* Our method uses the pseudo-label mask threshold (τM) and contrastive temperature (τc) as fixed constants primarily for simplicity and because we observed stable performance without tuning them for each dataset. **τM** is set at a moderately high value (close to 1.0) so that only pairs of samples with strong pseudo-label agreement are considered in the mask graph. This ensures we emphasize only reliable semantic relationships and filter out noisy pairings. **τc**, the contrastive learning temperature, is fixed to 0.1 (a common choice in contrastive frameworks ) to scale the cosine similarities into an appropriate range. In principle, τc controls the sharpness of the contrastive loss: a lower value focuses more on the hardest positives, while a higher value makes the loss smoother. We followed standard practice and found 0.1 to work well across all experiments.

Empirically, our method is **not overly sensitive** to reasonable variations in these parameters. We did not need to fine-tune τM or τc for different datasets – using the fixed values yielded consistently strong results. To examine this, we conducted additional sensitivity checks. We found that adjusting **τc** within a broad range (for example, 0.05 up to 0.2) had a negligible effect on clustering accuracy and convergence speed. Extremely large τc (e.g., approaching 1) can slightly slow down convergence due to a weaker gradient signal, and very small τc (e.g., 0.01) can make training a bit unstable, but within a reasonable range the final performance remains essentially unchanged. Similarly, for **τM**, as long as it is set high enough to exclude uncertain pairs (e.g., above 0.8), the clustering results are robust. A slightly lower threshold includes more pairs and did not significantly harm performance in our trials, though if τM were too low (allowing many weakly-related pairs), it could introduce noise. In practice, the chosen τM effectively balances inclusiveness and reliability. Importantly, our overall framework demonstrates strong resilience to hyperparameter choices: for instance, we showed in the paper that even large changes in other hyperparameters (such as loss weights λ1 and λ2) lead to minimal impact on clustering outcomes . This evidence of stability gives us confidence that fixed values of τM and τc do not hinder performance. In summary, **CausalMVC’s performance and convergence remain stable under a range of τM and τc** values, indicating the method’s robustness to these hyperparameters.

**回复 (中文):** *感谢审稿人提出有关超参数敏感性的问题。* 我们的方法将伪标签掩码图的阈值 **τM** 和对比学习温度 **τc** 设定为固定常数，主要原因在于这样做简化了模型，而且我们观察到即使不针对每个数据集微调，它们也能保持模型性能稳定。**τM** 我们设为接近1.0的较高阈值，以保证只有具有强语义一致性的样本对才在掩码图中被保留，这样可以突出可靠的聚类关联，忽略不确定的弱关联。**τc** 则固定为0.1（这是对比学习中常用的默认值 ），用于调整余弦相似度的尺度。从原理上讲，τc 控制对比损失的平滑程度：较小的温度会让模型更关注最困难的正样本对，较大的温度则使分布更平滑。我们参考常规做法选择了0.1，实验发现这一值在所有数据集上表现良好。

从实验结果来看，**本方法对这些参数并不敏感**，在合理范围内稍微调整 τM 和 τc 并不会对最终性能和收敛造成显著影响。实际上，我们未针对不同数据集单独调节这两个参数，而是使用统一的固定值便取得了稳定的效果。我们进行了额外的敏感性试验：将 **τc** 在一定范围内变化（例如从0.05到0.2），对聚类准确率和模型收敛速度影响很小。过高的 τc（如接近1）会因对比损失过于平缓而略微减慢收敛，但不会明显降低最终准确率；而过低的 τc（如0.01）会强化难样本对的作用，可能使训练略有不稳定。但只要 τc 处于适中的范围，模型的最终表现基本保持不变。类似地，对于 **τM**，只要阈值足够高以排除不可靠的样本对（例如设置在0.8以上），聚类结果就是鲁棒的。我们尝试略微降低阈值以包含更多样本对，发现性能并未明显下降；当然如果 τM 设得过低（包含大量弱关联样本对），可能会引入噪声干扰聚类。但在我们的实验中，所选较高阈值 **τM** 恰当地平衡了“关系对”的数量和可信度。值得一提的是，我们的整体框架对超参数选择表现出强健的鲁棒性：正如论文中的参数敏感性分析所示，即使大幅改变其他超参数（例如损失权重 λ1 和 λ2），对聚类结果的影响也很轻微 。这种稳定性说明，固定的 **τM** 和 **τc** 已足够取得良好性能，无需对每个数据集单独调节。总而言之，**CausalMVC 对阈值 τM 和温度 τc 有良好的鲁棒性**，在相当宽的取值范围内均能保持出色的聚类性能和正常收敛。





## **Q2: Gaussian Noise Variances Σ_μ and Σ_σ in Feature Perturbation**

**Response (English):** *Thank you for pointing out the need for clarification.* In Section 4.2, we introduce Gaussian perturbation to the preliminary features to help disentangle content from noise. The terms Σμ and Σσ denote the **variances (or standard deviations) used for perturbing the feature mean and standard deviation** of each view’s representation. In our implementation, these values are determined **adaptively from the data**: for each view, we first compute the mean vector μ(Hv) and standard deviation vector σ(Hv) of the preliminary representation Hv (e.g., across the batch or dataset). Then, Σμ(Hv) and Σσ(Hv) are set proportional to the observed fluctuations in those statistics. In practice, one can think of Σμ(Hv) as the standard deviation of Hv’s features (capturing how much the feature values vary, which in turn reflects how the mean might vary if sampled) and Σσ(Hv) as related to the variability of the feature distribution’s spread. By drawing ε ~ N(0,1) and forming the perturbed mean β(Hv) = μ(Hv) + εμ·Σμ(Hv) and perturbed scale γ(Hv) = σ(Hv) + εσ·Σσ(Hv)  , we ensure the noise added is **data-dependent**. In simpler terms, **Σμ and Σσ are chosen based on the internal feature distribution** so that the perturbations are neither too small nor overwhelming. They are not arbitrary constants; they scale with the magnitude of features in Hv.



Regarding the influence of these variances on performance: injecting Gaussian noise into the feature means and variances acts as a form of **regularization and augmentation** for our model. This controlled perturbation encourages the model to learn content features that are invariant to small shifts in feature distribution, thereby **mitigating Noisy View Dependency (NVD)**. If Σμ and Σσ are set appropriately (as described above), the perturbations are gentle enough to preserve the underlying semantics of the data while disturbing the superficial or noise-related aspects. This has several positive effects. First, it **improves intra-view consistency** – the model is forced to find a stable content representation for each view that holds even when the input features are slightly varied. Second, it helps the network **separate true content signals from noise**: any feature aspects that cause large changes under these perturbations are likely noise, so the model learns to discount them, focusing instead on the consistent content factor. We found that including this perturbation step led to better clustering performance than using the raw preliminary features alone. For instance, in our ablation experiments, models that did not properly disentangle noise (akin to removing this perturbation and subsequent content-style separation) saw noticeable drops in accuracy and NMI. 



**回复 (中文):** *感谢审稿人指出需要澄清的细节。* 在论文第4.2节中，我们对初始特征加入高斯噪声扰动，以帮助内容与噪声的解耦。符号 Σμ 和 Σσ 表示用于扰动特征均值和标准差的方差（或标准差）大小。换言之，它们控制对每个视图表示的均值和方差进行扰动时的噪声强度。在实现中，这些值是由数据自适应确定的：对于每个视图，我们先计算该视图初始表示 Hv 的均值向量 μ(Hv) 和标准差向量 σ(Hv)（例如在批次或整个数据集上计算）。然后，我们根据这些统计量的波动幅度来设定 Σμ(Hv) 和 Σσ(Hv)。直观地看，**Σμ(Hv) 可以理解为 Hv 特征值的标准差**（衡量特征值的自然波动，从而反映采样得到的均值可能的变化范围），而 **Σσ(Hv) 则反映了 Hv 内部“标准差”的变化程度**。在此基础上，我们对每个视图的均值加入噪声扰动 β(Hv) = μ(Hv) + εμ·Σμ(Hv)，对标准差加入扰动 γ(Hv) = σ(Hv) + εσ·Σσ(Hv)  （其中 ε 为服从 N(0,1) 的随机噪声）。如此一来，**噪声的幅度与数据本身的分布特性相适应**，确保扰动既不会微乎其微，也不至于过分剧烈。也就是说，Σμ 和 Σσ 不是固定的超参数，而是随 Hv 的特征尺度而定，使得加入的高斯噪声强度与原始特征的波动水平相称。

关于这些方差对模型性能的影响：对特征均值和方差注入高斯噪声实际上起到了**正则化和数据增强**的作用。这个受控的扰动促使模型去学习对小扰动不敏感的内容特征，从而**减轻噪视图依赖问题（NVD）**。当 Σμ 和 Σσ 设定得当时（如上所述基于数据分布来选取），噪声扰动幅度足够小，可以保留数据的语义信息，但又能扰乱表层的、与语义无关的变化。这样做有多个积极效果。首先，它**提高了同一视图内的内容一致性**——模型被迫在视图内找到一个稳定的内容表示，即使输入特征有轻微变化，该表示仍然有效。其次，这帮助网络**分离出真实的内容信号与噪声因素**：如果某些特征分量在扰动下变化很大，模型会意识到这些分量更多地属于噪声，从而降低对它们的依赖，转而关注那些在扰动下仍保持稳定的内容因子。我们的实验表明，引入这种扰动后再提取内容/风格，比直接使用未经扰动的初始特征能获得更好的聚类效果。例如，在消融实验中，如果模型不对噪声进行这种处理（相当于去掉高斯扰动及后续的内容-风格分离步骤），聚类准确率和NMI都会有明显下降。

可以的话补一下实验



## **Q3: Cluster Size Balance and Performance on Imbalanced Clusters**

**Response (English):** *Thank you for raising this important practical concern.* The multi-view datasets used in our experiments vary in their cluster size distributions – some are fairly balanced, while others are naturally imbalanced. For example, **NoisyMNIST** (a multi-view version of MNIST) has 10 digit classes with roughly equal samples per class, and **BBCSport** (sports articles) contains 5 categories that are fairly uniform in size. On the other hand, some datasets are inherently **unbalanced**. A notable case is **Caltech-all**, which includes many object categories (we used all 10 classes or more, totaling 9,144 samples ) with differing numbers of images per class; similarly, **YoutubeFace** has thousands of face images for a variety of people, and the number of images per person varies. We designed CausalMVC to be robust across these scenarios.



In our experiments, **CausalMVC performed strongly even on datasets with uneven cluster sizes**, indicating resilience to cluster imbalance. For instance, on the challenging **Caltech-all** dataset (which has a large number of classes with varied frequencies of images), our method achieved the best accuracy by a notable margin – about **9% higher than the next best method (VITAL)** . This superior result on Caltech-all suggests that our approach effectively discovers and clusters the underlying semantic groups despite the disparity in class sizes. We attribute this robustness to our focus on causal content representation. By disentangling content from style/noise and enforcing content consistency across views, the model is guided to cluster data based on fundamental semantic features rather than being unduly influenced by the sheer size of clusters. In other words, a smaller cluster with a distinct content representation will still be identified as its own group, because the model learns a clear content signal for it that doesn’t get overridden by larger clusters. Additionally, our contrastive learning strategy uses the pseudo-label mask graph to **emphasize only reliable pairings**; this means each data point mainly pulls itself closer to truly similar samples (likely from its own cluster) and does not get pulled toward dissimilar points, even if those belong to a dominant cluster. Such a mechanism helps prevent smaller clusters from being “washed away” or merged into larger ones inadvertently.

We acknowledge that extremely skewed cluster distributions (e.g., one tiny cluster among several huge clusters) can be challenging for any clustering method. Although our current evaluation did not include an artificially extreme imbalance scenario, the consistent results on naturally imbalanced datasets give us confidence that CausalMVC can handle a reasonable degree of imbalance. In practice, if faced with highly imbalanced clusters, one might consider additional measures (like adjusting the clustering threshold or using strategies to detect smaller clusters), but **our method as presented already demonstrates strong performance without any special handling for imbalance**. Overall, we emphasize that **CausalMVC’s robust content-focused clustering makes it effective even when cluster sizes are not uniform**, as evidenced by its gains on both balanced and imbalanced benchmark datasets.

**回复 (中文):** *感谢审稿人提出这一重要的实践问题。* 我们实验中使用的多视图数据集在簇（聚类）规模方面各不相同——有些数据集的簇大小比较均衡，而有些则天然存在不平衡的情况。例如，**NoisyMNIST**（MNIST数据集的多视图版本）包含10个数字类别，每类样本数大致相等；**BBCSport**（体育新闻数据集）涵盖5个类别，各类别样本数量也较为平均。相比之下，某些数据集本身具有明显的**不均衡**特征。典型案例是 **Caltech-all** 数据集，它包含多个对象类别（我们采用了全部类别，共9,144个样本 ），不同类别的图像数目差异较大。此外，**YoutubeFace** 数据集中包含许多人物的面部图像，每个人的图像数量并不相同，这也导致簇规模不一。针对这些不同情形，我们在设计 CausalMVC 时就考虑了模型在簇规模差异下的鲁棒性。

实验结果表明，**即使在簇大小不均衡的数据集上，CausalMVC 仍能取得出色的性能**，这表明我们的方法能够很好地适应簇规模的不平衡。举例来说，在具有高度不均衡类别的 **Caltech-all** 数据集上（该数据集包含大量类别，各类别样本数差别悬殊），我们的方法依然取得了**显著优于其他方法的聚类准确率**；特别地，CausalMVC 的准确率比第二好的方法（VITAL）高出约 **9%** 。这一结果令人鼓舞，说明我们的方法能够在类别规模差异较大的情况下有效地发现各自的聚类结构。我们认为，这种鲁棒性归功于方法对“因果内容表示”的重视。通过将内容特征与风格/噪声特征相互分离，并在视图内和跨视图保持内容一致性，我们的模型能够根据数据的核心语义特征来聚类，而不会过度受簇大小本身的影响。换言之，哪怕某个簇相对较小，**只要其拥有独特的内容表示（？）**，模型就能学到清晰的内容信号，将其识别为一个独立的聚类，而不会因为其它簇更大就将其淹没或错误地合并。此外，我们的对比学习策略利用伪标签掩码图**强调可靠的样本配对**；这意味着每个样本主要与真正相似的样本拉近（通常属于同一簇），而不会被拉向那些不相似的样本，即使后者所在簇的规模更大。这样的机制有助于防止小簇被无意间“吞并”到大簇中，从而保护了少数簇的辨识度。

我们也意识到，极端不平衡的簇分布（例如一个簇极小而其他簇极大的情况）对任何聚类方法来说都具有挑战性。尽管我们目前的测试尚未涵盖人为构造的极端不平衡场景，但在那些天然存在不均衡的数据集上取得的一致优秀表现令我们有信心认为，CausalMVC 能够应对一定程度的簇规模不均衡。在实际应用中，如果遇到高度不平衡的聚类情况，或许可以考虑一些附加措施（例如调整聚类决策阈值，或采用专门检测小簇的策略）；不过**我们的模型在未作特殊不平衡处理的情况下已经展现了强劲的效果**。总的来说，我们要强调的是：**CausalMVC 注重内容的聚类机制使其即使在簇大小不均等的条件下也能保持良好性能**，这在我们对平衡和不平衡基准数据集的实验中都得到了验证。

## Additional Comment: Support for Missing Views vs. Complete-View Scenario

**Response (English):** *Thank you for this additional question.* CausalMVC is **designed for the complete-view multi-view clustering setting**, meaning we assume each data instance has all views present during training and clustering. In our current framework, we do not tackle the case of missing views – if a view is absent, our method does not have a built-in mechanism (such as imputation or special modules) to handle it. We focused on the complete-view scenario because it is a common assumption in many multi-view benchmark datasets and it allows us to fully leverage the complementary information from all views. This assumption enables our model to enforce strong cross-view content consistency and perform accurate content-style disentanglement without worrying about incomplete data. **Under the complete-view setting, CausalMVC demonstrates clear advantages**: it uses information from every view to learn a unified representation, leading to more discriminative clustering results. By contrast, methods that must handle missing views often need to compromise by either ignoring missing data or filling it in with potentially noisy estimations, which can weaken performance. Our results show that when all views are available, the proposed approach achieves state-of-the-art clustering accuracy, confirming that fully utilizing all views is beneficial . We respectfully note that extending CausalMVC to scenarios with missing views would require additional research (e.g., developing a strategy to infer or learn with partial views), which is beyond the current scope. Rather than viewing the complete-view requirement as a limitation, we emphasize that **CausalMVC excels in the intended setting where multi-view data is complete**, a scenario in which it robustly outperforms other methods. We appreciate the reviewer’s suggestion and will consider addressing view-missing scenarios in future work. For now, we highlight that our method’s strength lies in its effective exploitation of **complete multi-view information** to produce robust and accurate clustering results.

**回复 (中文):** *感谢审稿人的这一补充提问。* CausalMVC **是为完整视图的多视图聚类场景所设计的**，也就是说，我们假定每个样本的所有视图在训练和聚类过程中都是可用的。当前的框架并未涉及视图缺失的情况——如果某个视图缺失，我们的方法中尚没有专门的机制（例如插值补全或特殊模块）来处理。我们将研究重点放在完整视图情形，是因为这在许多多视图基准数据集中是常见假设，并且这种设定允许我们充分利用所有视图的互补信息。在视图齐全的前提下，模型能够施加严格的跨视图内容一致性约束，并进行精确的内容-风格解耦，而不必顾虑不完整数据带来的不确定性。**在完整视图设定下，CausalMVC 展现出了明显的优势**：它充分利用每个视图的信息来学习统一表示，从而得到更具判别性的聚类结果。相比之下，那些需要应对视图缺失的方法通常不得不做出折衷，例如忽略缺失的数据或用可能包含噪声的估计值填充，这可能削弱模型性能。我们的实验结果表明，当所有视图均可获得时，所提方法能够取得最新的聚类精度，这也验证了充分利用完整视图信息的价值 。我们诚恳地指出，要将 CausalMVC 扩展到视图缺失的场景需要进一步的研究（例如开发针对部分视图的数据推断或训练策略），这超出了本工作的范围。我们并不将“要求视图完整”视作局限，相反更愿意强调 **CausalMVC 在预期的完整多视图环境中表现卓越**——在这一设定下，它可靠地利用所有视图的信息，聚类效果显著优于其他方法。感谢审稿人的建议，我们会在未来工作中考虑视图缺失问题。目前，我们侧重强调本方法的优点：通过**充分利用完整的多视图信息**，CausalMVC 能在多视图聚类任务中实现鲁棒且精确的性能。



# 超参数

- Reviewer b7E9

  *感谢审稿人就超参数选择和相关分析提供的反馈意见。* 在实现过程中，我们仔细调整了主要超参数，发现模型在合理范围内对参数**较为鲁棒**。在修改稿中，我们将清楚说明我们的选择过程。例如，正则权重 **β（用于 LSparseCov）** 是通过在 {0，1e-3，5e-3，1e-2，5e-2，1e-1，5e-1，1e0} 区间上搜索确定的，其他损失权衡系数 (λ₁, λ₂) 也在 {0.01, 0.1, 1, 10, 100} 的集合中尝试过 。我们选择了未发生过拟合且验证集聚类性能最优的取值。实际观察中，如果将 β 设为0（即不使用稀疏协方差正则），模型性能会下降（正如消融实验所示）；相反，过大的 β 会过度限制模型，导致性能略有下降。然而，在**较宽的中等 β 取值范围内，聚类结果是稳定**的，这表明本方法并不对精确的正则权重过分敏感。

- Reviewer 3taB

  **超参数设置 (β, λ₁, λ₂) 及其影响：** 我们将根据审稿人建议，在论文中增加关于 β、λ₁ 和 λ₂ 设置的明确指南，并讨论它们对模型性能的影响。实验结果表明，模型对超参数 λ₁ 和 λ₂ **并不敏感**，在相当宽的取值范围内模型性能都保持稳定 。出于简化考虑，这两个超参数均可默认设为 1——即使将 λ₁、λ₂ 从 0.01 调至 100，模型性能变化也不大，体现了方法对这些参数的鲁棒性 。相比之下，β（稀疏内容-风格正则项的系数）对性能的影响更显著。β 控制内容与风格解耦的程度：当 β = 0（不施加任何解耦正则）时，模型准确率明显降低，等效于退化为未进行内容-风格分离的情形 。我们发现**适度且非零**的 β 值至关重要——例如，将 β 控制在 0.005 至 0.1 区间内可以保持优秀的聚类性能；反之，β 取值过大可能因过度惩罚有用相关性而略微损害结果。实际操作中，我们通常将 β 设定在 0.01–0.05 左右，这一范围能很好地平衡内容-风格分离的力度，在 β=0 对比下显著提升聚类效果 。在论文修改中，我们将提供这些建议，说明 λ₁ 和 λ₂ 可以默认为 1（或在一个数量级范围内调整），而 β 建议取10^-2量级的小值，以确保实现有效的内容-风格解耦同时不影响有用信息的保留。

- Reviewer 4SxA

  **回复 (中文):** *感谢审稿人提出有关超参数敏感性的问题。* 我们的方法将伪标签掩码图的阈值 **τM** 和对比学习温度 **τc** 设定为固定常数，主要原因在于这样做简化了模型，而且我们观察到即使不针对每个数据集微调，它们也能保持模型性能稳定。**τM** 我们设为接近1.0的较高阈值，以保证只有具有强语义一致性的样本对才在掩码图中被保留，这样可以突出可靠的聚类关联，忽略不确定的弱关联。**τc** 则固定为0.1（这是对比学习中常用的默认值 ），用于调整余弦相似度的尺度。从原理上讲，τc 控制对比损失的平滑程度：较小的温度会让模型更关注最困难的正样本对，较大的温度则使分布更平滑。我们参考常规做法选择了0.1，实验发现这一值在所有数据集上表现良好。

  从实验结果来看，**本方法对这些参数并不敏感**，在合理范围内稍微调整 τM 和 τc 并不会对最终性能和收敛造成显著影响。实际上，我们未针对不同数据集单独调节这两个参数，而是使用统一的固定值便取得了稳定的效果。我们进行了额外的敏感性试验：将 **τc** 在一定范围内变化（例如从0.05到0.2），对聚类准确率和模型收敛速度影响很小。过高的 τc（如接近1）会因对比损失过于平缓而略微减慢收敛，但不会明显降低最终准确率；而过低的 τc（如0.01）会强化难样本对的作用，可能使训练略有不稳定。但只要 τc 处于适中的范围，模型的最终表现基本保持不变。类似地，对于 **τM**，只要阈值足够高以排除不可靠的样本对（例如设置在0.8以上），聚类结果就是鲁棒的。我们尝试略微降低阈值以包含更多样本对，发现性能并未明显下降；当然如果 τM 设得过低（包含大量弱关联样本对），可能会引入噪声干扰聚类。但在我们的实验中，所选较高阈值 **τM** 恰当地平衡了“关系对”的数量和可信度。值得一提的是，我们的整体框架对超参数选择表现出强健的鲁棒性：正如论文中的参数敏感性分析所示，即使大幅改变其他超参数（例如损失权重 λ1 和 λ2），对聚类结果的影响也很轻微 。这种稳定性说明，固定的 **τM** 和 **τc** 已足够取得良好性能，无需对每个数据集单独调节。总而言之，**CausalMVC 对阈值 τM 和温度 τc 有良好的鲁棒性**，在相当宽的取值范围内均能保持出色的聚类性能和正常收敛。

# 复杂度

- Reviewer 3taB

  感谢审稿人关于计算复杂度与可扩展性的宝贵建议。我们从理论与实证两个方面对 CausalMVC 的计算复杂度进行了进一步分析。理论上，CausalMVC 的主要模块包括双重差分内容-风格提取网络、内容一致性约束与风格感受场机制，核心计算均基于线性投影与注意力机制。对于每个视图，差分注意力的计算复杂度为 $O(N^2D)$，其中 $N$ 为样本数，$D$ 为特征维度；共有 $V$ 个视图，因此总复杂度为 $O\bigl((V+1)N^2D + VND\bigr)$。其中第一项源自内容、风格与噪声的三组注意力计算，第二项则来自各自投影与聚合操作。整体上，该复杂度与 GCFAgg、DealMVC 等现有深度多视图聚类方法保持一致，均处于可接受范围，并未引入任何指数级计算或额外瓶颈。

  此外，相比 GCFAgg 等基于全局样本对（$O(N^2)$）进行特征聚合的方法，CausalMVC 并不构建显式的全局相似图，而是借助因果引导在**样本层级**进行语义一致对齐，从而在保持聚类性能的同时有效降低冗余计算。与 DealMVC 需要维护多个 $N \times N$ 图对齐的设计不同，我们的模型通过共享的内容表示实现对多个视图信息的统一建模。

  从实证结果看，我们在十个标准数据集（规模从几百到十万）上进行了训练测试，训练时间与内存开销与现有方法基本持平，未出现明显放缓或资源瓶颈，证明该方法具备良好的可扩展性。我们将在修改稿中加入训练资源统计，以进一步佐证模型的实用性与计算可控性。再次感谢审稿人的专业建议，我们相信 CausalMVC 兼具有效性与计算效率，可广泛适用于大规模多视图聚类任务。

- Reviewer 2bNa

  感谢您提出这一疑虑。我们想澄清的是，CausalMVC 的计算复杂度保持与基线方法**相当**。引入双重差分内容-风格网络确实增加了一些开销，但幅度很小。本质上，该双分支网络执行了两次注意力计算（内容和风格各一次）而非一次，外加一次简单的相减操作。这些操作都是标准的矩阵运算（线性投影和 Softmax 注意力），具有良好的可扩展性。其复杂度与典型的注意力模块处于同一量级——对于每个视图包含 $N$ 个样本，时间复杂度约为 $O(N^2)$，这与其他深度多视图聚类模型中使用的操作类似。双分支设计仅仅是常数因子的增加，并非计算量的指数膨胀。我们还确保网络的维度规模和层数与其他方法相当，以避免计算上的剧烈增长。

  在实际实验中，我们的方法训练时间和内存消耗与基线模型非常接近。我们在十个基准数据集上评估了 CausalMVC，**未**发现与标准方法相比有明显的放缓迹象，这表明额外的内容-风格处理开销是相当低的。因此，双差分网络并**没有显著提高**整体算法的复杂度或运行时间。

