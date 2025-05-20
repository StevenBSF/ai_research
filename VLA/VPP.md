最近更新很少，因为懒得写太详细的文章，之后也许可以更新更新零散的思考论述和自己的paper... 这次恰巧有个大作业要写报告，就把这篇最近的视频模型For具身智能的论文分析放上来了。

文章和网站链接：[A Generalist Robot Policy with Predictive Visual Representations](https://link.zhihu.com/?target=https%3A//video-prediction-policy.github.io/) , Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations。陈建宇老师团队和Berkeley的研究。



by [tanh](https://www.zhihu.com/people/thkkkhncs)

## **概览**

Video Prediction Policy（VPP）这篇论文聚焦于具身智能中的通用操作策略。它首先用不同的人类和机器人操作数据**预训练一个视频预测的扩散模型**（[Video Diffusion Models](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=Video+Diffusion+Models&zhida_source=entity), VDMs），然后利用VDMs提取出的视觉表征，去**训练后续的Diffusion Policy用于操作**。VPP在两项模拟器和真实世界Benchmark上consistently大幅超过现有方法，比如Calvin ABC-D benchmark上相对之前的SOTA有28.1%的提升，在真实世界灵巧操作任务上有28.8%的提升。

![img](https://pic1.zhimg.com/v2-28ad8fa499a0a01872ae7fa028f2aabc_1440w.jpg)



## **背景和动机**

VPP工作的大方向是打造一个能解决诸多task的通用机器人策略。这其中有两个重要的部分：**[action network](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=action+network&zhida_source=entity)**，和**vision encoder**。

- 对于打造更加先进的action network，现有的工作主要包括：利用预训练的VLM 、直接从diverse的机器人数据集中预训练、结合自回归或者diffusion架构，以及scale up action network。
- 至于vision encoder则主要focus在从egocentric的视频数据集中，为具身任务学习更加高效的视觉表征，通过对比学习的方法或者图像重建的方法。

这篇文章主要focus在**视觉表征也就是vision encoder**的学习，感觉上切入的是利用预训练视频生成模型的巨大潜力，将其vision encoder用于机器人策略中。以往的工作中的vision encoder一般都是用对比学习的方法，从两帧或者单帧重建来进行学习，然而没有很好地学会机器人操作过程中的物理规律、动力学和物体的变化，因为他们考虑的时间上下文很短。因此VPP考虑这样的视频生成模型：输入一帧H*W是当前图片，输出(T-1)个H*M的对未来T-1时间步的预测图片（视频），这样的视频生成模型的vision encoder就可以建模对未来的表征，称为“**predictive visual representations**”。其实从sora等视频生成模型的模型架构中也可以看出，对于long-context视频的生成，有一个包含long-context的vision encoder是多么重要，不只是对所有的图片做独立的VAE，而是把时间的上下文考虑进来一起做VAE，类似于TECO的方式（Temporally Consistent Transformers for Video Generation）。从下图中可以看出两种视觉表征的区别，文中用的**predictive visual representations**多了一个时间维度。

![img](https://pica.zhimg.com/v2-597ccc5bc66dc82efaa4ead02e364b82_1440w.jpg)

因此文中提到的key insights在于这样的**predictive visual representations对于下游的actoin learning非常有信息量，**包括物体的移动、机器人本身的运动等等。而且这一部分视频生成模型是可以从internet-scale video dataset和robotics dataset中通过一致的视频生成loss同时获益的，从而实现将互联网数据中蕴含的物理知识迁移到机器人策略中。我之前就思考过这个问题，一个模仿学习的模型策略，考虑的历史理应是越长越好（当然，recover的能力可以考虑从数据中获取）。所以具身操作的问题相当于一个视频实时回归的问题，要综合历史每一帧的图像，实时地在每一timestep得到一个连续的action。然而，如果这里图像的encoder只考虑单帧，会缺少上下文信息；如果视觉特征考虑历史所有帧，消耗资源又太大了。这也是为什么目前主流的真机策略模型比如ACT、DP、DP3、RDT都只考虑了少数几帧的历史，然后预测未来的多步action（即action chunking）。

于是文章就提出VPP，分为两步训练：

1. 基于一个预训练好的视频生成模型（[Stable Video Diffusion](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=Stable+Video+Diffusion&zhida_source=entity)），Finetune 一个text-guided视频预测模型（[TVP](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=TVP&zhida_source=entity)）。数据集来源是各种manipulation dataset：ego-centric人类操作的视频；开源机器人数据集（Open X-Embodiment）；自采集的机器人数据。这一步希望得到一个在manipulation domain中更强的controllable视频生成模型。
2. 做一个多任务的diffusion policy，基于TVP中提取的predictive visual representations。论文觉得视频模型的输出(T,H,W)维度还是太高，因此输入他前一层的表征（图中是这么画的），用一个[video former](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=video+former&zhida_source=entity)来distill必要的时间和空间信息。

## **具体方法**

![img](https://pic1.zhimg.com/v2-caff2e6fdd4ade6ca9ea8ff635834680_1440w.jpg)



对于视频模型TVP：选择1.5B的预训练好的Stable Video Diffusion作为基座模型。原先该模型没有语言instruction作为condition，只有初始帧s_0作为condition。于是TVP (就是前面说到的Text-guided Video Prediction Model)用cross-attention将CLIP的language feature L_emb结合进来。TVP还将视频模型输出的分辨率调整到了16*256*256（应该是16帧，256*256）。对于数据集D，模型用diffusion objective训练，考虑从(纯噪音x_t, L_emb, s_0)中重建视频x_0（x_0 \in D）的MSE。由于有三种数据集internet-based human manipulation、public robot manipulation data和self-collected datasets，所以loss对三种数据集进行加权平均。之后将TVP的权重freeze住，用于action的学习。

对于action learning：TVP模型作为一个vision encoder。但是这存在两个问题：

1. 由于TVP生成视频太慢了，会有开环控制的问题。
2. 文章认为原始像素格式的视频中有过多、无关的信息，会干扰有效决策。

解决的方法是只进行一步的预测。也就对应了前文所说的TVP主要是作为vision encoder而不是denoiser。文章的figure 5发现不用进行全部的denoise，**一步直接预测就可以有个大概的完成任务的信息了**。这样省很多时间。对于机器人多视角图像的问题，他们分别独立地预测每个视角的latent feature。

![img](https://picx.zhimg.com/v2-afb71b56f6e08c6c6c0852998b45db75_1440w.jpg)



TVP拿到的表征还是很高维，所以用了个video former将上面的latent feature通过spatial attention和temporal attention得到固定数量的token Q’’（ 模型架构图Fig3 的action learning左半部分）。最后用一个diffusion policy，将Q’’通过cross attention灌入[DiT blocks](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=DiT+blocks&zhida_source=entity)（模型架构图Fig3 的action learning右半部分），结合文本prompt、time noise去噪得到action。由于他们用的是灵巧手setting，有3维position (xyz)，3维rotation，12维finger，所以在action的loss上也对三个部分进行了加权。

## **效果**

论文在模拟环境CALVIN和[MetaWorld](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=MetaWorld&zhida_source=entity)，以及真机[Panda arm](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=Panda+arm&zhida_source=entity)和[XHand dexterous hand](https://zhida.zhihu.com/search?content_id=252079793&content_type=Article&match_order=1&q=XHand+dexterous+hand&zhida_source=entity)中进行了实验，主要为了回答以下几个问题：

1． 成功率；

2． video pretraining和internet manipulation datasets能否增强成功率；

3． predictive representation和其他表征相比；

4． video diffusion model的哪一层提供最有效的predictive visual representations。

论文比较的baseline特别多，包括RT-1, Diffusion Policy, Robo-Flamingo, Uni-Pi(同类), Susie(生成目标图片), GR-1, MDT(重建一个masked未来帧)。

数据：

- Something-Something-V2中的193,690 条**human manipulation** 轨迹
- 179,074条高质量的robotic manipulation 轨迹（来自open-X等）
- Downstream task datasets (包括模拟器数据集或者要部署的那个真机数据集)

资源：

- 视频模型训练: 2天，8张A100
- Action learning: 6-12小时，4张A100
- 部署：4090，7-10Hz

![img](https://pic3.zhimg.com/v2-a6dcafa4ebe3e272e350a09bdc26eef4_1440w.jpg)



![img](https://pic1.zhimg.com/v2-1812e5025b2147f4e619531e00735e80_1440w.jpg)



从成功率上来看是远超直接学习action的方法比如DP。

![img](https://pic3.zhimg.com/v2-fb917aa76ebd9e3a88fcfe681603832a_1440w.jpg)

![img](https://pic2.zhimg.com/v2-9c0a5b24fd03f73653d1c237047fd0f3_1440w.jpg)



四个问题从ablation study中得到的结论是：

1. VPP确实成功率高很多
2. 没有互联网的操作视频，会降一点avg.len，但是没有stable video diffusion (SVD)的pretrain，会降很多：**说明视频生成模型的预训练很重要，虽然里面的数据集完全不是机器人视角，而且很多与机器人操作无关。**
3. 比其他的visual representation比如stable-VAE式的强多了。
4. 最有效的predictive representations在上采样block的中间，而不是最终的pixels。

## **总结和感悟**

### **未来**：

文章强调视频生成模型对具身任务的潜力，还强调predictive visual representations对通用策略的重要性。我认为具身模型从internet-scale的视频中获益是必要的，比如2023年的数据显示Youtube每分钟有 500 小时的视频内容上传，相当于每天有 720,000 小时的新视频上传，对应每年2.6亿小时。Youtube总共至少有24亿小时的视频数据。如果折合成机器人操作数据要乘个0.1的话，youtube上的视频数据也能对应2.4亿小时，按照时薪25元来算，对应**60亿的采真机数据的capex**，这么大的capex+数据格式更换的风险+训练出来没效果的风险，估计没有哪个公司敢和全互联网的力量对抗。而且，这些internet-scale的视频数据不只是作为一张张图片去学visual encoder，而是作为视频去学有空间+时间两个维度的信息。

结合之前所说的完美的具身智能操作相当于一个“**视频实时回归**”任务，他的计算量是比简单的图像任务和视频理解还要大不少的，完美的policy和实时视频生成的计算量差不多，未来通用的方向我认为也是朝着这方向走的：综合历史多张图片（i.e. 视频），生成未来多帧的视频预测。这样也方便做post-training和RL，以及作为world model去大力sample、search，也方便利用人类的visual prompt。

### **局限性**：

1. 视频模型因为众所周知的原因，生成速度特别慢，这方面未来肯定会有大量工作来改进。
2. 多视角是个问题，机器人需要多视角才能更好地捕获全局信息，然而视频模型预测多视角还有待改进。
3. 预测出来的视频不符合物理规律怎么办？机械臂糊了怎么办等等。也是视频模型需要改进的点。
4. 文中的方法过于依赖特定任务action的采集。完成各种任务的Action trajectory可能非常难穷举，难以靠采这个数据实现通用。
5. 文章可以再探讨一下不同视频预训练模型（在不同的互联网数据上训练），对结果的影响。因为不只是作为visual representation，视频模型在具身智能上就已经很有意义了。
6. 一系列未来需要后人做改进的点：末端的触觉力反馈等等、动态操作任务等。