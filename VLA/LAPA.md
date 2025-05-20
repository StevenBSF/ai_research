## 问题

真实世界的robot数据集要求人体操作，这就很难scaling。因此作者们认为可以从大规模的互联网视频学习物理交互和人类行为。

但同样地，这也带来两个问题：

1. 视频没有action labels
2. 视频中的数据分布与机器人系统的本体和环境严重不同

**因此作者尝试去无监督地训robotic foundation model。**

![img](https://pic3.zhimg.com/v2-84dc6b64f1e817f48b7fa47acb1ff104_1440w.jpg)



## 方法

![img](https://picx.zhimg.com/v2-d27469aa7d96c3f5db8a6b0c5842c15f_1440w.jpg)



两个模型，一个[latent action quantization](https://zhida.zhihu.com/search?content_id=253550182&content_type=Article&match_order=1&q=latent+action+quantization&zhida_source=entity) and [latent pretraining](https://zhida.zhihu.com/search?content_id=253550182&content_type=Article&match_order=1&q=latent+pretraining&zhida_source=entity)。

方法猜测：首先为动作序列的每帧训练一个latent actions，第二步让vlm根据第一个观察预测这个latent actions。完成训练后就可以让这个vlm部署到真实环境里了。

## LATENT ACTION QUANTIZATION

![img](https://pic1.zhimg.com/v2-7894923dd3ddc89ba2f5e750f1de03c4_1440w.jpg)



输入当前时刻x_1和下一时刻x_2, 经过patch Embedding得到两个嵌入p_2，p_1，之后输入[spatial transformer](https://zhida.zhihu.com/search?content_id=253550182&content_type=Article&match_order=1&q=spatial+transformer&zhida_source=entity)。为了保持时序性，经过[causal transformer](https://zhida.zhihu.com/search?content_id=253550182&content_type=Article&match_order=1&q=causal+transformer&zhida_source=entity)得到最后的e_1和e_2的嵌入，两者的距离为d_1=e_2 - e_1, 离散化d_1得到z_1,通过[nsvq](https://zhida.zhihu.com/search?content_id=253550182&content_type=Article&match_order=1&q=nsvq&zhida_source=entity)防止码本崩溃，之后将d_1作为kv和当前时刻x_1（停止梯度更新）做cross-attention ，最后通过上采样得到未来时刻x_2。