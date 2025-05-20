## **出自[UC伯克](https://zhida.zhihu.com/search?content_id=235599568&content_type=Article&match_order=1&q=UC伯克&zhida_source=entity)利用[diffusion model](https://zhida.zhihu.com/search?content_id=235599568&content_type=Article&match_order=1&q=diffusion+model&zhida_source=entity)来完成[zero shot](https://zhida.zhihu.com/search?content_id=235599568&content_type=Article&match_order=1&q=zero+shot&zhida_source=entity)机器人操控**

**主页：**[SuSIE: Subgoal Synthesis via Image Editing (rail-berkeley.github.io)](https://link.zhihu.com/?target=https%3A//rail-berkeley.github.io/susie/)

题目：ZERO-SHOT ROBOTIC MANIPULATION WITH PRETRAINED IMAGE-EDITING DIFFUSION MODELS

## **1. 背景**

如果通用机器人要在非结构化的环境中应用的话，那么它们需要在新的场景中操控新的物体。但是现在很多方法都是用自己训练数据来训练一个策略，泛化到新的场景和物体的能力很弱。

## 2.Insight

人类去执行任务之前，在脑海里会有你自己想像完成这个任务是怎样的，然后在控制你的肌肉来完成任务。所以作者就像机器人操控问题解耦为两个阶段，第一阶段是用生成模型（stable diffusion）生成一张假设完成任务的图片，第二阶段是用low-level的控制策略使机器臂抵达生成的位置。

## 3.方法

3.1 微调[InstructPix2pix](https://zhida.zhihu.com/search?content_id=235599568&content_type=Article&match_order=1&q=InstructPix2pix&zhida_source=entity)模型

InstructPix2pix模型在测试时，输入是初始图片，语言指令，输出是想像中完成这个指令的子任务的图片。

微调数据：BridgeData V2(包含语言指令跟动作轨迹，共60k条，取45k)，和Somthing-Something dataset（人类操控物体的视频,无机器人动作轨迹,75k）

![img](https://picx.zhimg.com/v2-9e7ed8d0492263740c30991e3b92beb9_1440w.jpg)

如何让模型输出每个子任务的图片能够保持一定距离，让第二阶段的low-level策略能够实现，而且起码完成任务中的一小步骤。

调整超参数（玄学时间）：

![img](https://pic2.zhimg.com/v2-8d2d40835060163d0c61c7e24a0a268b_1440w.jpg)

3.2 训练一个goal-reaching策略

模型是diffusion policy，输入当前帧和未来帧，输出动作。所以策略是无关任务指令的（无语言输入）。

用diffusion model是因为它更能捕捉机器人数据中的多模态信息，有助于提高一系列任务的性能。而且预测的不是一个动作，而是后四步动作，执行的是这四个动作每个维度取平均作为最终的动作，目的是保持时间上的一致性和稳定动作；

训练数据是BridgeData V2（45k有语言跟动作，15k只有动作）。（疑问：这个45k有语言的数据怎么办）

## 4.实验

![img](https://pica.zhimg.com/v2-506f54df22221059042259547746d4f2_1440w.jpg)

最上面是SUSIE生成子任务的图片，并完成任务。中间是[RT-2-X模型](https://zhida.zhihu.com/search?content_id=235599568&content_type=Article&match_order=1&q=RT-2-X模型&zhida_source=entity)，下面是baseline，会提供最终完成任务ground truth goal image，但是不会给完成子任务的图片。

## 5.总结

提出一个机器人操控方法：输入任务指令，能先生成子任务图片，再用这个子任务图片来指导机器人完成low-level任务。这个的zero-shot指的是在微调InstructPix2pix模型和训练diffusion policy之后，**结合之后能够泛化到未见过的物体。**

缺点：微调生成模型和训练low-level策略是分开的，意味着生成模型是不关心low-level策略的能力的，如果训练数据有相同的，但是这里假设数据中任务可到达的地方，策略也能抵达，但这不现实，未来工作是如何联合训练让生成模型也能意识到低级策略的能力。

![image-20250512002313738](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20250512002313738.png)
