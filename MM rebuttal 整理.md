**To Reviewer b7E9:**

**Q1.1\&Q2.1: Computational complexity.**

Due to limited response space to each reviewer, we have to answer this at Q2 of reviewer 2bNa. Please refer to reponse there.

**Q1.2: Dependence on hyperparameter tuning.**

Due to limited space, please refer to the response to Q3 of reviewer 3taB.

**Q1.3: Model interpretability.**

While our model involves complex causal disentanglement, it remains interpretable through an explicitly structured design. As detailed in Fig 2, each view’s representation is decomposed into content, style, and noise, following[5, 26], enabling us to trace their roles in clustering. Content ensures cross-view semantic consistency, style enhances intra-cluster compactness, and views with higher inter- and intra-cluster variance contribute more to the fused representation. Additionally, sparse regularization (Eq. 6) strengthens the separation of causal factors. 

**Q1.4: Overfitting to noise-free views.**

This concern aligns with the issue of Dominant View Dependency (DVD), where a model over-relies on semantically rich or noise-free views while undervaluing information from others. Our method is explicitly designed to address DVD. Through causal disentanglement, each view contributes both content and style independently. The style retains complementary information from less dominant views, while the content-centered receptive field and sparse regularization prevent any single view from dominating. These designs improve generalization in imbalanced, real-world scenarios.

**Q1.5\&Q2.4: explanations of $L_{SparseCov}$.** 

$L_{SparseCov}$ combines an L1 sparsity term to suppress irrelevant style dimensions with a low-rank constraint to limit broad entanglements between content and style. Together, they enforce a clear separation of content and style factors, as confirmed by the ablation results in Table 3.

**Q2.2&Q2.5&Q2.6: Hyperparameter settings, dataset, and ablation study expansion.**

We have conducted experiments on Caltech7 to evaluate the impact of increasing number of views in Figure 4, and the analysis of DVD is provided in Table 5. Experiments on other datasets and hyperparameter settings will be included in the supplementary material.

**Q2.3: Comparison with other causal methods.**

MVCRL is the causal method included in our experimental comparisons. To the best of our knowledge, other causal methods were not specifically designed for deep MVC and therefore are not applicable to this task.

---

**To Reviewer QS4h:**

**Q1: How can the proposed loss functions ensure effective disentanglement of semantic factors?**

Our model promotes disentanglement via a dual differential content-style network, which isolates semantics from noise by subtracting invariant noise mappings. Further, we enforce cross-view content consistency and include an entropy maximization term to prevent collapse and ensure content does not absorb view-specific signals. Additionally, our sparse covariance regularizer (L1 + low-rank) reduces both local and global dependencies between content and style, enhancing their separability. These combined constraints result in a well-structured and identifiable decomposition of latent factors.

**Q2: What properties of CausalMVC contribute to its improved performance with an increasing number of views?**

In principle, our method fully leverages the complementary information provided by multiple views: each additional view contributes a style representation of the current observation for the underlying content (see Eq.10), enabling the model to more comprehensively capture the true semantics of each sample. The scalability of our approach stems from its causal consistency. Adding more views essentially provides additional independent observations of the same latent content. As a result, the content representation of each sample becomes more robust and accurate, thereby enhancing final results.

**Q3: How does the quality of initial pseudo-labels impact the final clustering performance?**

Our approach is designed to be robust against inaccurate initial pseudo-labels. Specifically, we adopt a pseudo-label mask graph that filters out unreliable pairwise relationships during contrastive clustering. Only sample pairs with high semantic consistency are treated as positive pairs, while uncertain ones are ignored. This selective supervision prevents error propagation during early training. Moreover, the clustering assignments are iteratively refined: after each training epoch, pseudo-labels are updated based on the improved unified representations. This progressive refinement mechanism enables the model to correct early-stage label noise and gradually converge to reliable cluster structures.

**Q4: reproducibility.**

We will release the source code and trained model.

---

**To Reviewer 3taB:**

**Q1: Computational complexity.**

Please refer to the response to Q2 of reviewer 2bNa.

**Q2: The authors should quantify how much the content-style receptive field improves performance versus standard fusion methods in ablation studies.**

To quantify the effect of the content-centered style receptive field (CSR), we conducted an ablation comparing it to a standard fusion (Std-Fusion) baseline, which uses a typical attention-based mechanism to fuse content and style features without our receptive field design. As shown below, incorporating CSR consistently improves clustering performance across metrics (ACC, NMI, PUR) and datasets:

|Dataset||ACC|NMI|PUR|
|-|-|-|-|-|
|Caltech7|Std-Fusion|89.12|81.64|89.43|
||Ours|91.54|84.57|91.56|
|YouTubeFace|Std-Fusion|38.97|36.14|45.87|
||Ours|40.33|37.32|47.36|


Beyond these quantitative gains, we also observe qualitative improvements: as shown in Figs. 5(b) and 5(c) of the main paper, models without CSR produce more dispersed cluster structures, while CSR leads to more compact and semantically coherent clusters. This highlights its role in reinforcing view-specific style cues while preserving content alignment.

**Q3: Guidelines for setting key hyperparameters (e.g., $\beta$, $\lambda_1$, $\lambda_2$).**

We conducted sensitivity analyses and found that CausalMVC is relatively robust to a wide range of key hyperparameter settings. In particular, Figure 6 in the paper shows the model is relatively robust to variations in $\lambda_1$ and $\lambda_2$, with stable performance observed across a wide range. In contrast, $\beta$, which controls sparse covariance regularization, has a more noticeable effect: setting $\beta = 0$ degrades performance, while overly large values slightly constrain the model. In practice, we recommend using $\beta = 0.01$ and $\lambda_1 = \lambda_2 = 1$, which consistently yield strong results across datasets.

**Q4: Considered alternatives to k-means for non-convex clusters, such as spectral clustering?**

We chose k-means for its efficiency and standard usage in deep clustering. Our learned causal representation is designed to be convex and well-separated (see Fig. 5), which k-means handles effectively. While methods like spectral clustering can handle non-convexity, they are computationally impractical on large datasets. That said, CausalMVC is flexible and can support more advanced clustering heads in future work.

---

**To Reviewer 2bNa:**

**Q1: Difference from existing differential Transformers and advantages of dual differential content-style network.**

The dual differential content-style network is specifically designed for multi-view clustering and fundamentally differs from existing differential Transformers. Instead of computing attention differences within a single stream, it uses two parallel branches to extract content and style features independently, each guided by a noise-aware query-key mechanism (Eq. 4 and 5). This design enables targeted noise suppression and effective content-style disentanglement, which are crucial in noisy multi-view settings. Moreover, our model introduces a content-style perturbation regularization (Eq. 3) to improve representation stability under semantic-preserving transformations, a capability not addressed in conventional Transformer architectures. Our network explicitly separates and preserves dual representations, leading to more robust and interpretable clustering.

**Q2: Computational complexity.**

For each view, the computational complexity of the dual differential content-style network is $O(N^2D)$. Given V views in total, the overall complexity is $O((V+1)N^2D + VND)$, where the first term arises from the three attention mechanisms for content, style, and noise, while the second term comes from the projection and aggregation operations. This matches the complexity of existing deep MVC methods like DealMVC, without introducing additional overhead. On the NuswideOBJ dataset using an A6000 GPU, the average training time for 100 epochs is 388 seconds for DealMVC and 376 seconds for our method.

**Q3: Content and style contribution analysis.**

Both content and style components contribute to clustering performance in CausalMVC, but in complementary ways. Content representations capture the primary semantic structure, grouping samples by category, while style representations provide fine-grained intra-class variation, helping tighten clusters. As shown in Figure 5, excluding the style component (Fig. 5b) results in more dispersed clusters, whereas including both content and style (Fig. 5c) yields more compact and discriminative groupings. The table below quantifies each component’s contribution:

|Dataset||ACC|NMI|PUR|
| - | - | - | - | - |
|Caltech7 |S| 56.24 | 50.73 | 54.39 |
||C| 87.35 | 80.83 | 87.64 |
||C+S| 91.54|84.57 |91.56 |
| YouTubeFace|S|23.52|20.56|23.66|
||C| 35.63 | 32.47 | 41.24 |
||C+S|40.33| 37.32|47.36 |

---

**To Reviewer 4SxA:**

**Q1: sensitivity of $τ_M$ and $τ_c$.**

We set both $τ_M$ and $τ_c$ as fixed constants. Specifically, $τ_M$ is chosen close to 0.8 to retain only highly consistent sample pairs in the pseudo-label mask graph, ensuring reliable contrastive supervision. $τ_c$ is fixed at 0.1, as the model shows low sensitivity to its value.

The sensitivity analysis is as follows:

|τM|0|0.2|0.5|0.8|0.9|
|-|-|-|-|-|-|
|Wiki|45.12|55.87|60.45|62.98|61.22|
|YoutubeFace| 25.34|33.52|38.92|40.33|39.16|

|τc|0.05|0.10|0.20|0.50|1.0|
|-|-|-|-|-|-|
|Wiki|62.30|62.98|61.53|61.25|61.22|
|YoutubeFace|39.91|40.33|40.21|39.52|39.16|

**Q2: Explanations of $Σ_μ$ and $Σ_σ$.**

We introduce Gaussian perturbations to the feature means and standard deviations to improve robustness. The terms $Σ_μ(H^v)$ and $Σ_σ(H^v)$ are not learnable, but represent the inherent variability of $H^v$ itself. Specifically, $Σ_μ(H^v)$ is the standard deviation of per-sample mean vectors $μ(H^v_i)$, while $\sum_σ(H^v)$ reflects the variation in $σ(H^v_i)$ across the batch. These values act as noise scaling factors that adapt to the distribution of each view. By sampling noise as $ε·Σ$ with $ε ～ N(0,1)$, we ensure that the perturbation matches the uncertainty of the data. 

We illustrate below the impact of the scales of $Σ_μ$ and $Σ_σ$ on Caltech7:

|Scaling factor|perturbation scales|ACC|
|-|-|-|
|0|0|85.84|
|0.5|$0.5·ε·Σμ, 0.5·ε·Σσ$|90.37|
|1.0|$ε·Σμ, ε·Σσ$|91.54|
|1.5|$1.5·ε·Σμ, 1.5·ε·Σσ$|91.25|

**Q3: Cluster Imbalance Robustness.**

Several datasets in our benchmark, such as Caltech-all and YoutubeFace, contain naturally imbalanced clusters. CausalMVC performs strongly even under these condition. For example, on Caltech-all, we achieve a 8.94% ACC improvement over the next-best method. This resilience stems from our content-focused modeling and noise suppression, which prevent large clusters from dominating. Contrastive learning with a masked graph further ensures that each sample is influenced only by reliable neighbors, protecting smaller clusters from being absorbed.

**Additional Comment: Does the method handle missing views?**

CausalMVC currently operates under the complete-view assumption, focusing on fully leveraging all available views to maximize semantic alignment. We agree that handling missing views is important for practical applications. Extending our framework would require additional mechanisms, which we consider a valuable direction for future work.