# GEVRM: GOAL-EXPRESSIVE VIDEO GENERATION MODEL FOR ROBUST VISUAL MANIPULATION

$p_{\Theta}(a_t, \tau_{t:T} \mid g, \tau_{0:t}) = p_{\phi}(\tau_{t:T} \mid g, \tau_{0:t})\, p_{\varphi}(a_t \mid \tau_{0:T}).$

模型分为两个阶段,对于当前时刻$t$,对于指令$g$和视频/图片序列$\tau_{0:t}$生成预测帧$\tau_{t:T}$.视频/图片序列$\tau_{0:t}$和预测帧$\tau_{t:T}$共同用于生成action.

我搜索到的:

1. **两阶段方法**：首先训练视频生成模型预测任务的未来视觉目标帧（即预测期望达到的图像状态），然后基于这些目标帧训练**条件在目标上的动作策略**。这种方法将视觉预测和动作规划解耦：视频模型提供**视觉目标**，策略模型负责实现目标 。
2. **端到端单模型方法**：将视觉预测与动作生成融合在同一个模型中，通过**统一的训练过程**同时学习“预测未来”的能力和“生成动作”的能力。模型在内部隐式地进行状态推演和决策，从观察直接输出动作，无需显式的中间目标帧。

![image-20250503132656609](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20250503132656609.png)

### ROBOT BEHAVIOR PLANNER

文章中提到对于视频/图像序列的表征提取,即Video spatio-temporal compression,是使用2D VAE和3D VAE的方法结合进行的,2D VAE是为了降低训练开销,3D VAE进一步捕捉时空信息关联.

language instruction是利用T5 encoder提取.

Random mask mechanism这里文章只提到the understanding of physical laws and object consistency,以及附录中的超参数设置,没有看到更多细节例证(?)

### ROBOT ACTION PREDICTION

文章中提到,分为 State alignment to simulate responses和Goal-guided action prediction两部分.

 State alignment to simulate responses此处使用的无监督对比学习,区分视频/图像序列的具体指令,似乎有提升空间?目前设计

![image-20250503152144474](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20250503152144474.png)

对于测试阶段算法部分起初阅读的时候产生了误解,原因是t = t+1的位置看错了.这段算法流程主要是在T时间范围内,对于目前状态的t时刻和指令生成M个未来帧,做出未来帧预测之后,进行相应的L_test次step,每次step算一次单位时间,对应t+1.对于当前的test的x_t进行encode并且二范,goal中第l个未来目标帧进行encode并且二范,生成相应的动作之后同时更新test环境的x_t+1.



- latent space的prompt token和text指令做对齐似乎可以考虑?
- 20-50帧的连续未来帧我认为还是太局限了,能不能想到关键帧?(不知道目前有没有相关工作)